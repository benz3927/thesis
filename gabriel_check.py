#!/usr/bin/env python3
"""
GABRIEL Robustness Check - gabriel.whatever() only
===================================================
Runs your exact v3 prompt through GABRIEL's infrastructure,
then compares against your original v3 scores.

Usage:
    python gabriel_check.py
"""

import pandas as pd
import numpy as np
import gabriel
import os
import glob
import asyncio
import re
import json
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY', '<your-api-key-here>')

TRANSCRIPTS_DIR = 'data/processed/Transcripts'
CACHE_DIR = 'data/cache'
V3_SCORES_FILE = f'{CACHE_DIR}/gpt_dissent_scores_v3.csv'
GABRIEL_SAVE_DIR = f'{CACHE_DIR}/gabriel_scores'
OUTPUT_FILE = f'{CACHE_DIR}/gabriel_vs_v3_comparison.csv'

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(GABRIEL_SAVE_DIR, exist_ok=True)

# ============================================================================
# DISTRICT MAPPING
# ============================================================================

DISTRICT_MAPPING = {
    'syron': 'Boston', 'minehan': 'Boston', 'rosengren': 'Boston',
    'mcdonough': 'New York', 'geithner': 'New York', 'dudley': 'New York',
    'boehne': 'Philadelphia', 'santomero': 'Philadelphia', 'plosser': 'Philadelphia', 'harker': 'Philadelphia',
    'jordan': 'Cleveland', 'pianalto': 'Cleveland', 'mester': 'Cleveland',
    'broaddus': 'Richmond', 'lacker': 'Richmond', 'barkin': 'Richmond',
    'forrestal': 'Atlanta', 'guynn': 'Atlanta', 'lockhart': 'Atlanta', 'bostic': 'Atlanta',
    'keehn': 'Chicago', 'moskow': 'Chicago', 'evans': 'Chicago',
    'melzer': 'St. Louis', 'poole': 'St. Louis', 'bullard': 'St. Louis',
    'stern': 'Minneapolis', 'kocherlakota': 'Minneapolis', 'kashkari': 'Minneapolis',
    'hoenig': 'Kansas City', 'george': 'Kansas City',
    'mcteer': 'Dallas', 'fisher': 'Dallas', 'kaplan': 'Dallas',
    'parry': 'San Francisco', 'yellen': 'San Francisco', 'williams': 'San Francisco', 'daly': 'San Francisco',
}

def get_district(speaker):
    if pd.isna(speaker):
        return None
    speaker_lower = str(speaker).lower()
    for name, district in DISTRICT_MAPPING.items():
        if name in speaker_lower:
            return district
    return None

# ============================================================================
# LOAD TRANSCRIPTS
# ============================================================================

print("=" * 70)
print("GABRIEL ROBUSTNESS CHECK (whatever only)")
print("=" * 70)

print("\n[1] Loading transcripts...")
transcript_files = glob.glob(f'{TRANSCRIPTS_DIR}/*_t.csv')
print(f"    Found {len(transcript_files)} files")

all_data = []
for file_path in transcript_files:
    try:
        df = pd.read_csv(file_path)
        filename = os.path.basename(file_path)
        date_str = filename.split('_')[0]
        date = pd.to_datetime(date_str, format='%Y%m%d')
        text_col = 'clean_transcript_text' if 'clean_transcript_text' in df.columns else 'transcript_text'
        speaker_col = 'Speaker' if 'Speaker' in df.columns else 'speaker'
        if text_col not in df.columns or speaker_col not in df.columns:
            continue
        df = df[[speaker_col, text_col]].rename(columns={speaker_col: 'speaker', text_col: 'text'})
        df['date'] = date
        df = df.dropna(subset=['text', 'speaker'])
        all_data.append(df)
    except Exception:
        continue

transcripts = pd.concat(all_data, ignore_index=True)
transcripts['district'] = transcripts['speaker'].apply(get_district)

grouped = (
    transcripts
    .groupby(['speaker', 'date', 'district'])
    .agg({'text': lambda x: ' '.join(x)})
    .reset_index()
)
grouped['year'] = grouped['date'].dt.year
grouped = grouped[~grouped['speaker'].str.upper().str.contains('CHAIR', na=False)].copy().reset_index(drop=True)

def truncate_text(text, max_chars=7000):
    if len(text) <= max_chars:
        return text
    return (text[:2000] + "\n[...]\n"
            + text[len(text)//2 - 1000:len(text)//2 + 1000]
            + "\n[...]\n" + text[-2000:])

grouped['text_truncated'] = grouped['text'].apply(truncate_text)

# Create identifier for merging back
grouped['identifier'] = grouped.apply(
    lambda row: f"{row['speaker']}_{row['date'].strftime('%Y%m%d')}", axis=1
)

print(f"    Speaker-meetings: {len(grouped):,}")

# ============================================================================
# BUILD PROMPTS
# ============================================================================

print("\n[2] Building prompts...")

def build_v3_prompt(text, speaker):
    return f"""You are analyzing FOMC meeting transcripts. Score {speaker}'s policy stance on a scale from -10 to +10.

SCORING SCALE:

-10 to -6: STRONG DISSENT FOR TIGHTER POLICY (HAWKISH)
- Wants higher interest rates than proposed
- "We should raise rates more" / "Policy is too easy"
- "Inflation is a serious concern" / "We're behind the curve"
- "Too accommodative" / "We need to act more aggressively"

-5 to -1: MILD HAWKISH LEAN
- Some concern policy is too loose
- "I worry about inflation" / "Perhaps we should do more"
- Supports proposal but would prefer slightly tighter

0: AGREEMENT / NEUTRAL
- "I support the proposal" / "I agree with the Chairman"
- No clear directional preference
- Purely procedural discussion

+1 to +5: MILD DOVISH LEAN
- Some concern policy is too tight
- "I worry about growth" / "Unemployment is concerning"
- Supports proposal but would prefer slightly easier

+6 to +10: STRONG DISSENT FOR LOOSER POLICY (DOVISH)
- Wants lower interest rates than proposed
- "We should cut rates" / "Policy is too tight"
- "Growth is weakening" / "We're moving too fast"
- "Too restrictive" / "We should wait" / "Premature"

IMPORTANT:
- Focus on POLICY PREFERENCE relative to the committee proposal
- Discussing regional weakness alone = +1 to +3, not strong dissent
- Most members agree (score 0) most of the time

Text to analyze:
\"\"\"{text[:7000]}\"\"\"

Based on the text above, score {speaker}'s policy direction.
Reply with ONLY a single integer from -10 to +10."""

prompts = [
    build_v3_prompt(row['text_truncated'], row['speaker'])
    for _, row in grouped.iterrows()
]
identifiers = [
    f"{row['speaker']}_{row['date'].strftime('%Y%m%d')}"
    for _, row in grouped.iterrows()
]
print(f"    Built {len(prompts)} prompts")

# ============================================================================
# RUN GABRIEL.WHATEVER
# ============================================================================

async def run_whatever():
    print("\n[3] Running gabriel.whatever()...")
    results = await gabriel.whatever(
        prompts=prompts,
        identifiers=identifiers,
        save_dir=f'{GABRIEL_SAVE_DIR}/whatever_v2',
        model='gpt-4o-mini',
        reset_files=False,
    )
    return results

results = asyncio.run(run_whatever())

print(f"\n    Columns: {results.columns.tolist()}")
print(f"    Shape: {results.shape}")
print(f"    First 5 raw responses:")
print(results[['Identifier', 'Response']].head(5).to_string())

# ============================================================================
# PARSE SCORES - handle ["0"] JSON format
# ============================================================================

print("\n[4] Parsing scores...")

def parse_score(val):
    """Parse response which may be '0', '["0"]', '-3', '["-3"]', etc."""
    try:
        s = str(val).strip()
        if s in ('nan', 'None', ''):
            return np.nan

        # Try JSON parse first (handles '["0"]' format)
        try:
            parsed = json.loads(s)
            if isinstance(parsed, list) and len(parsed) > 0:
                s = str(parsed[0]).strip()
            elif isinstance(parsed, (int, float)):
                return max(-10, min(10, int(parsed)))
        except (json.JSONDecodeError, TypeError):
            pass

        # Direct int
        s_clean = s.replace('+', '')
        try:
            return max(-10, min(10, int(s_clean)))
        except ValueError:
            pass

        # Extract integer from text
        match = re.search(r'(-?\d+)', s)
        if match:
            return max(-10, min(10, int(match.group(1))))
        return np.nan
    except Exception:
        return np.nan

results['gabriel_score'] = results['Response'].apply(parse_score)

print(f"    Parsed: {results['gabriel_score'].notna().sum()} / {len(results)}")
print(f"    NaN: {results['gabriel_score'].isna().sum()}")
print(f"\n    Distribution:")
print(results['gabriel_score'].describe())

# ============================================================================
# MERGE WITH V3 SCORES - use Identifier to align properly
# ============================================================================

print("\n[5] Merging with v3 scores (using identifiers)...")

v3 = pd.read_csv(V3_SCORES_FILE)
v3['date'] = pd.to_datetime(v3['date'])

# Merge gabriel scores into grouped using Identifier
gabriel_lookup = results[['Identifier', 'gabriel_score']].copy()
gabriel_lookup = gabriel_lookup.rename(columns={'Identifier': 'identifier'})

merged = grouped[['speaker', 'date', 'district', 'year', 'identifier']].merge(
    gabriel_lookup,
    on='identifier',
    how='left'
)

# Merge v3 scores
merged = merged.merge(
    v3[['speaker', 'date', 'gpt_dissent_direction']].rename(
        columns={'gpt_dissent_direction': 'v3_score'}
    ),
    on=['speaker', 'date'],
    how='left'
)

print(f"    Rows: {len(merged)}")
print(f"    v3 non-null: {merged['v3_score'].notna().sum()}")
print(f"    gabriel non-null: {merged['gabriel_score'].notna().sum()}")

# Sanity check: print a few matched pairs
print(f"\n    Sample matched pairs:")
sample = merged.dropna(subset=['v3_score', 'gabriel_score']).head(10)
print(f"    {'speaker':<25} {'date':<12} {'v3':>5} {'gabriel':>8}")
for _, row in sample.iterrows():
    print(f"    {row['speaker']:<25} {str(row['date'])[:10]:<12} {row['v3_score']:>5.0f} {row['gabriel_score']:>8.0f}")

# ============================================================================
# COMPARE SCORES
# ============================================================================

print("\n" + "=" * 70)
print("[6] SCORE COMPARISON")
print("=" * 70)

valid = merged.dropna(subset=['v3_score', 'gabriel_score'])
print(f"\n    Observations with both scores: {len(valid):,}")

if len(valid) > 0:
    r = valid['v3_score'].corr(valid['gabriel_score'])
    print(f"    Correlation (v3 vs gabriel): r = {r:.3f}")

    print(f"\n    {'':>20} {'v3':>10} {'gabriel':>10}")
    for stat in ['mean', 'std', 'min', '25%', '50%', '75%', 'max']:
        v = valid['v3_score'].describe()[stat]
        g = valid['gabriel_score'].describe()[stat]
        print(f"    {stat:>20} {v:>10.2f} {g:>10.2f}")

    # Direction agreement
    valid = valid.copy()
    valid['v3_dir'] = np.sign(valid['v3_score'])
    valid['gab_dir'] = np.sign(valid['gabriel_score'])
    agree = (valid['v3_dir'] == valid['gab_dir']).mean()
    print(f"\n    Direction agreement (sign match): {agree:.1%}")
else:
    print("    No valid observations to compare!")

# ============================================================================
# REGRESSION (same approach as your regression script)
# ============================================================================

print("\n" + "=" * 70)
print("[7] REGRESSION COMPARISON")
print("=" * 70)

try:
    import statsmodels.api as sm

    unemp = pd.read_csv('data/cache/regional_unemployment_all.csv')
    unemp['date'] = pd.to_datetime(unemp['date'])
    unemp['year_month'] = unemp['date'].dt.to_period('M')

    # Compute national average and gap (same as your regression script)
    nat_unemp = unemp.groupby('year_month')['unemployment_rate'].mean().rename('nat_unemp')
    unemp = unemp.merge(nat_unemp, on='year_month')
    unemp['unemployment_gap'] = unemp['unemployment_rate'] - unemp['nat_unemp']

    merged['year_month'] = merged['date'].dt.to_period('M')

    reg_data = merged.merge(
        unemp[['year_month', 'district', 'unemployment_rate', 'unemployment_gap']],
        on=['year_month', 'district'],
        how='inner'
    )

    # District presidents only, exclude NY
    reg_data = reg_data[
        (reg_data['district'].notna()) &
        (reg_data['district'] != 'New York')
    ]

    print(f"\n    Regression observations: {len(reg_data)}")

    reg_data['post_greenspan'] = (reg_data['date'] >= pd.Timestamp('2006-02-01')).astype(int)

    for period_name, period_data in [
        ('FULL SAMPLE', reg_data),
        ('POST-GREENSPAN (2006+)', reg_data[reg_data['post_greenspan'] == 1]),
        ('GREENSPAN ERA (pre-2006)', reg_data[reg_data['post_greenspan'] == 0]),
    ]:
        print(f"\n    === {period_name} (N={len(period_data)}) ===\n")
        for score_col, label in [
            ('v3_score', 'Custom GPT (v3)'),
            ('gabriel_score', 'GABRIEL whatever()'),
        ]:
            subset = period_data.dropna(subset=[score_col, 'unemployment_gap'])
            if len(subset) < 30:
                print(f"    {label}: not enough data ({len(subset)} obs)\n")
                continue
            X = sm.add_constant(subset['unemployment_gap'])
            y = subset[score_col]
            model = sm.OLS(y, X).fit(cov_type='cluster',
                                      cov_kwds={'groups': subset['speaker']})
            print(f"    {label}:")
            print(f"      unemployment_gap coef = {model.params['unemployment_gap']:.4f}")
            print(f"      std error             = {model.bse['unemployment_gap']:.4f}")
            print(f"      t-stat                = {model.tvalues['unemployment_gap']:.3f}")
            print(f"      p-value               = {model.pvalues['unemployment_gap']:.4f}")
            print(f"      R-squared             = {model.rsquared:.4f}")
            print(f"      N                     = {int(model.nobs)}")
            print()

except FileNotFoundError as e:
    print(f"    File not found: {e}")

# ============================================================================
# SAVE
# ============================================================================

print("\n[8] Saving...")
merged.to_csv(OUTPUT_FILE, index=False)
print(f"    Saved: {OUTPUT_FILE}")
print("\nDONE")