#!/usr/bin/env python3
"""
CLAUDE Dissent Scoring - ROBUSTNESS CHECK
Exact same prompt as GPT v8, using Claude API
"""

import pandas as pd
import numpy as np
from anthropic import Anthropic
from tqdm import tqdm
import os
import glob
from dotenv import load_dotenv, find_dotenv
import time

_ = load_dotenv(find_dotenv())
client = Anthropic()

CACHE_DIR = 'data/cache'
TRANSCRIPTS_DIR = 'data/processed/Transcripts'
CHECKPOINT_FILE = f'{CACHE_DIR}/claude_dissent_v8_checkpoint.csv'
OUTPUT_FILE = f'{CACHE_DIR}/claude_dissent_scores_v8.csv'

os.makedirs(CACHE_DIR, exist_ok=True)

print("=" * 70)
print("CLAUDE DISSENT SCORING - v8 PROMPT ROBUSTNESS CHECK")
print("=" * 70)

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

print("\n[1] Loading transcripts...")

transcript_files = glob.glob(f'{TRANSCRIPTS_DIR}/*_t.csv')
print(f"    Found {len(transcript_files)} files")

all_data = []
for file_path in tqdm(transcript_files, desc="    Loading"):
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
    except:
        continue

transcripts = pd.concat(all_data, ignore_index=True)
transcripts['district'] = transcripts['speaker'].apply(get_district)
print(f"    Loaded {len(transcripts):,} statements")

# ============================================================================
# AGGREGATE TO SPEAKER-MEETING
# ============================================================================

print("\n[2] Aggregating to speaker-meeting level...")

grouped = (
    transcripts
    .groupby(['speaker', 'date', 'district'])
    .agg({'text': lambda x: ' '.join(x)})
    .reset_index()
)
grouped['year'] = grouped['date'].dt.year
print(f"    Speaker-meetings: {len(grouped):,}")

# ============================================================================
# CHECK FOR CHECKPOINT
# ============================================================================

start_idx = 0
if os.path.exists(CHECKPOINT_FILE):
    checkpoint = pd.read_csv(CHECKPOINT_FILE)
    start_idx = len(checkpoint)
    print(f"    Resuming from checkpoint: {start_idx}/{len(grouped)}")
    checkpoint['date'] = pd.to_datetime(checkpoint['date'])
    grouped = grouped.merge(
        checkpoint[['speaker', 'date', 'claude_dissent_direction']],
        on=['speaker', 'date'],
        how='left'
    )
else:
    grouped['claude_dissent_direction'] = np.nan

# ============================================================================
# SCORING FUNCTION - EXACT SAME V8 PROMPT, CLAUDE MODEL
# ============================================================================

def score_dissent_direction_claude(text, speaker):
    """Exact v8 prompt through Claude instead of GPT-4o"""

    if "CHAIR" in speaker.upper():
        return 0.0

    # Same end-weighted truncation as v8
    if len(text) > 6000:
        text = text[:1500] + "\n[...]\n" + text[-4500:]

    # EXACT SAME PROMPT AS V8
    prompt = f"""You are analyzing FOMC meeting transcripts. Score {speaker}'s policy stance on a scale from -10 to +10.

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

EXAMPLES:

Score: -8
Speaker argues inflation expectations are becoming unanchored, says the committee is "behind the curve," urges an immediate 50bp hike instead of the proposed 25bp, and warns that delay risks credibility.

Score: 0
Speaker reviews regional economic conditions, notes mixed signals, says "I support the Chairman's recommendation," and discusses communication strategy without expressing a policy preference.

Score: +6
Speaker emphasizes rising unemployment in their district, worries the proposed rate path is too aggressive, says "we should be patient" and "the risks to growth concern me more than inflation at this point."

Score: -2
Speaker generally supports the proposal but adds "I could have supported a somewhat firmer action" or notes inflation risks deserve close monitoring.

Score: +9
Speaker explicitly opposes the proposed tightening, argues for a rate cut or extended pause, says policy is "premature" or "too restrictive," and warns of recession risk.

IMPORTANT:
- Focus on POLICY PREFERENCE relative to the committee proposal
- Discussing regional weakness alone = +1 to +3, not strong dissent
- Most members agree (score 0) most of the time

Text to analyze:
\"\"\"{text[:7000]}\"\"\"

Based on the text above, score {speaker}'s policy direction.
Reply with ONLY a single integer from -10 to +10."""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=5,
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )
        score_str = response.content[0].text.strip()
        score_str = score_str.replace('+', '')
        score = int(''.join(c for c in score_str if c.isdigit() or c == '-'))
        return max(-10, min(10, score))
    except Exception as e:
        print(f"    Error: {e}")
        return np.nan

# ============================================================================
# TEST SAMPLES
# ============================================================================

print("\n[3] Testing Claude on 10 samples...")
print("=" * 70)

sample = grouped.sample(10, random_state=42)
for _, row in sample.iterrows():
    score = score_dissent_direction_claude(row['text'], row['speaker'])
    direction = "HAWKISH" if score < 0 else "DOVISH" if score > 0 else "NEUTRAL"
    print(f"{row['speaker']:<20} | {row['date'].strftime('%Y-%m-%d')} | Score: {score:+3d} ({direction})")

# ============================================================================
# TEST ON KNOWN DISSENTERS
# ============================================================================

print("\n" + "=" * 70)
print("[4] Testing on KNOWN dissenters...")
print("=" * 70)

votes = pd.read_excel("data/FOMC_Dissents_Data.xlsx", skiprows=3)
votes["date"] = pd.to_datetime(votes["FOMC Meeting"])

dissent_records = []
for _, row in votes.iterrows():
    for col, direction in [("Dissenters Tighter", "tighter"),
                            ("Dissenters Easier", "easier"),
                            ("Dissenters Other/Indeterminate", "other")]:
        if pd.notna(row.get(col)):
            for name in str(row[col]).split(", "):
                dissent_records.append({
                    "date": row["date"],
                    "name": name.strip().upper(),
                    "direction": direction
                })

dissent_df = pd.DataFrame(dissent_records)
dissent_df = dissent_df[(dissent_df['date'].dt.year >= 1994) & (dissent_df['date'].dt.year <= 2020)]

print("\nScoring known dissent meetings:\n")
correct = 0
total = 0
for _, dissent in dissent_df.sample(10, random_state=123).iterrows():
    match = grouped[
        (grouped['date'] == dissent['date']) &
        (grouped['speaker'].str.upper().str.contains(dissent['name']))
    ]
    if len(match) > 0:
        row = match.iloc[0]
        score = score_dissent_direction_claude(row['text'], row['speaker'])

        if dissent['direction'] == 'tighter':
            correct_dir = score < 0
        elif dissent['direction'] == 'easier':
            correct_dir = score > 0
        else:
            correct_dir = None

        if correct_dir is not None:
            total += 1
            if correct_dir:
                correct += 1
            check = "✓" if correct_dir else "✗"
        else:
            check = "?"

        print(f"  {dissent['name']:<12} {dissent['date'].strftime('%Y-%m-%d')} | Voted: {dissent['direction']:<7} | Score: {score:+3d} {check}")

if total > 0:
    print(f"\n  Direction accuracy (excl. other): {correct}/{total} = {correct/total:.1%}")

# ============================================================================
# CONTINUE?
# ============================================================================

remaining = len(grouped) - start_idx
cost_est = remaining * 0.003
print(f"\n[5] Full run: {remaining} remaining, ~${cost_est:.2f}")
proceed = input("Continue? (y/n): ")

if proceed.lower() != 'y':
    print("Exiting.")
    exit()

# ============================================================================
# SCORE ALL
# ============================================================================

print("\n[6] Scoring all...")

for i in tqdm(range(start_idx, len(grouped)), initial=start_idx, total=len(grouped)):
    if pd.notna(grouped.loc[grouped.index[i], 'claude_dissent_direction']):
        continue

    row = grouped.iloc[i]
    score = score_dissent_direction_claude(row['text'], row['speaker'])
    grouped.loc[grouped.index[i], 'claude_dissent_direction'] = score

    if (i + 1) % 200 == 0:
        save_cols = ['speaker', 'date', 'district', 'year', 'claude_dissent_direction']
        grouped[save_cols].dropna(subset=['claude_dissent_direction']).to_csv(CHECKPOINT_FILE, index=False)
        print(f"\n    Checkpoint: {i+1}/{len(grouped)}")

    time.sleep(0.05)

# ============================================================================
# SAVE
# ============================================================================

print("\n[7] Saving results...")

output = grouped[['speaker', 'date', 'district', 'year', 'claude_dissent_direction']].copy()
output['date'] = output['date'].dt.strftime('%Y-%m-%d')
output.to_csv(OUTPUT_FILE, index=False)

print(f"    Saved: {OUTPUT_FILE}")
print(f"    Rows: {len(output):,}")

if os.path.exists(CHECKPOINT_FILE):
    os.remove(CHECKPOINT_FILE)

# ============================================================================
# COMPARE WITH GPT v8
# ============================================================================

print("\n" + "=" * 70)
print("[8] COMPARING WITH GPT v8")
print("=" * 70)

gpt_v8_file = f'{CACHE_DIR}/gpt_dissent_scores_v8.csv'
if os.path.exists(gpt_v8_file):
    gpt_v8 = pd.read_csv(gpt_v8_file)
    gpt_v8['date'] = pd.to_datetime(gpt_v8['date'])

    output_compare = output.copy()
    output_compare['date'] = pd.to_datetime(output_compare['date'])

    merged = output_compare.merge(
        gpt_v8[['speaker', 'date', 'gpt_dissent_direction']],
        on=['speaker', 'date'],
        how='inner'
    )

    valid = merged.dropna(subset=['claude_dissent_direction', 'gpt_dissent_direction'])
    r = valid['claude_dissent_direction'].corr(valid['gpt_dissent_direction'])

    print(f"\n    Matched observations: {len(valid):,}")
    print(f"    Correlation (Claude vs GPT v8): r = {r:.3f}")

    print(f"\n    {'':>20} {'GPT v8':>10} {'Claude':>10}")
    for stat in ['mean', 'std', 'min', '25%', '50%', '75%', 'max']:
        g = valid['gpt_dissent_direction'].describe()[stat]
        c = valid['claude_dissent_direction'].describe()[stat]
        print(f"    {stat:>20} {g:>10.2f} {c:>10.2f}")

    # Direction agreement
    valid = valid.copy()
    valid['gpt_dir'] = np.sign(valid['gpt_dissent_direction'])
    valid['claude_dir'] = np.sign(valid['claude_dissent_direction'])
    agree = (valid['gpt_dir'] == valid['claude_dir']).mean()
    print(f"\n    Direction agreement: {agree:.1%}")
else:
    print("    GPT v8 scores not found — run comparison manually.")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"\nClaude score distribution:")
print(output['claude_dissent_direction'].describe())

print(f"\nDirection counts:")
output['direction_cat'] = pd.cut(output['claude_dissent_direction'],
                                  bins=[-11, -0.5, 0.5, 11],
                                  labels=['Hawkish', 'Neutral', 'Dovish'])
print(output['direction_cat'].value_counts())

print(f"\n✅ Done! CSV saved to: {OUTPUT_FILE}")