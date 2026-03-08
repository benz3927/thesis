#!/usr/bin/env python3
"""
GPT Dissent Scoring v3 - WITH DIRECTION (looser vs tighter)
"""

import pandas as pd
import numpy as np
from openai import OpenAI
from tqdm import tqdm
import os
import glob
from dotenv import load_dotenv, find_dotenv
import time

_ = load_dotenv(find_dotenv())
client = OpenAI()

CACHE_DIR = 'data/cache'
TRANSCRIPTS_DIR = 'data/processed/Transcripts'
CHECKPOINT_FILE = f'{CACHE_DIR}/gpt_dissent_v3_checkpoint.csv'
OUTPUT_FILE = f'{CACHE_DIR}/gpt_dissent_scores_v3.csv'

os.makedirs(CACHE_DIR, exist_ok=True)

print("="*70)
print("GPT DISSENT SCORING v3 - WITH DIRECTION")
print("="*70)

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
        checkpoint[['speaker', 'date', 'gpt_dissent_direction']], 
        on=['speaker', 'date'], 
        how='left'
    )
else:
    grouped['gpt_dissent_direction'] = np.nan

# ============================================================================
# DIRECTIONAL SCORING FUNCTION
# ============================================================================

def score_dissent_direction(text, speaker):
    """GPT scoring with direction: -10 (tighter) to +10 (looser)"""
    
    # Chairs = 0
    if "CHAIR" in speaker.upper():
        return 0.0
    
    # Sample text if too long
    if len(text) > 6000:
        text = text[:2000] + "\n[...]\n" + text[len(text)//2-1000:len(text)//2+1000] + "\n[...]\n" + text[-2000:]

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

IMPORTANT:
- Focus on POLICY PREFERENCE relative to the committee proposal
- Discussing regional weakness alone = +1 to +3, not strong dissent
- Most members agree (score 0) most of the time

Text to analyze:
\"\"\"{text[:7000]}\"\"\"

Based on the text above, score {speaker}'s policy direction.
Reply with ONLY a single integer from -10 to +10."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=5,
            temperature=0
        )
        score_str = response.choices[0].message.content.strip()
        # Handle negative numbers
        score_str = score_str.replace('+', '')
        score = int(''.join(c for c in score_str if c.isdigit() or c == '-'))
        return max(-10, min(10, score))
    except:
        return 0.0

# ============================================================================
# TEST SAMPLES
# ============================================================================

print("\n[3] Testing directional prompt on 10 samples...")
print("="*70)

sample = grouped.sample(10, random_state=42)
for _, row in sample.iterrows():
    score = score_dissent_direction(row['text'], row['speaker'])
    direction = "HAWKISH" if score < 0 else "DOVISH" if score > 0 else "NEUTRAL"
    print(f"{row['speaker']:<20} | {row['date'].strftime('%Y-%m-%d')} | Score: {score:+3d} ({direction})")

# ============================================================================
# TEST ON KNOWN DISSENTERS
# ============================================================================

print("\n" + "="*70)
print("[4] Testing on KNOWN dissenters...")
print("="*70)

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
        score = score_dissent_direction(row['text'], row['speaker'])
        
        # Check direction using SIGN, not threshold
        if dissent['direction'] == 'tighter':
            correct_dir = score < 0
            expected = "−"
        elif dissent['direction'] == 'easier':
            correct_dir = score > 0
            expected = "+"
        else:
            correct_dir = None  # "other" - skip
            expected = "?"
        
        actual = "−" if score < 0 else "+" if score > 0 else "0"
        
        if correct_dir is not None:
            total += 1
            if correct_dir:
                correct += 1
            check = "✓" if correct_dir else "✗"
        else:
            check = "?"
        
        print(f"  {dissent['name']:<12} {dissent['date'].strftime('%Y-%m-%d')} | Voted: {dissent['direction']:<7} | Score: {score:+3d} {check}")

print(f"\n  Direction accuracy (excl. other): {correct}/{total} = {correct/total:.1%}")

# ============================================================================
# CONTINUE?
# ============================================================================

remaining = len(grouped) - start_idx
print(f"\n[5] Full run: {remaining} remaining, ~${remaining * 0.002:.2f}")
proceed = input("Continue? (y/n): ")

if proceed.lower() != 'y':
    print("Exiting.")
    exit()

# ============================================================================
# SCORE ALL
# ============================================================================

print("\n[6] Scoring all...")

for i in tqdm(range(start_idx, len(grouped)), initial=start_idx, total=len(grouped)):
    if pd.notna(grouped.loc[grouped.index[i], 'gpt_dissent_direction']):
        continue
        
    row = grouped.iloc[i]
    score = score_dissent_direction(row['text'], row['speaker'])
    grouped.loc[grouped.index[i], 'gpt_dissent_direction'] = score
    
    if (i + 1) % 200 == 0:
        save_cols = ['speaker', 'date', 'district', 'year', 'gpt_dissent_direction']
        grouped[save_cols].dropna(subset=['gpt_dissent_direction']).to_csv(CHECKPOINT_FILE, index=False)
        print(f"\n    Checkpoint: {i+1}/{len(grouped)}")
    
    time.sleep(0.03)

# ============================================================================
# SAVE
# ============================================================================

print("\n[7] Saving results...")

output = grouped[['speaker', 'date', 'district', 'year', 'gpt_dissent_direction']].copy()
output['date'] = output['date'].dt.strftime('%Y-%m-%d')
output.to_csv(OUTPUT_FILE, index=False)

print(f"    Saved: {OUTPUT_FILE}")
print(f"    Rows: {len(output):,}")

if os.path.exists(CHECKPOINT_FILE):
    os.remove(CHECKPOINT_FILE)

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*70)
print("SUMMARY")
print("="*70)

print(f"\nScore distribution:")
print(output['gpt_dissent_direction'].describe())

print(f"\nDirection counts:")
output['direction_cat'] = pd.cut(output['gpt_dissent_direction'], 
                                  bins=[-11, -0.5, 0.5, 11], 
                                  labels=['Hawkish', 'Neutral', 'Dovish'])
print(output['direction_cat'].value_counts())

print(f"\n✅ Done! CSV saved to: {OUTPUT_FILE}")