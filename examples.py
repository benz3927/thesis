#!/usr/bin/env python3
"""
Extract example scored transcripts for Appendix D
Run: python examples_for_appendix.py
"""

import pandas as pd
import glob
import os

CACHE_DIR = 'data/cache'
TRANSCRIPTS_DIR = 'data/processed/Transcripts'

# Load v8 scores
v8 = pd.read_csv(f'{CACHE_DIR}/gpt_dissent_scores_v8.csv')
v8['date'] = pd.to_datetime(v8['date'])

# Load dissent data
votes = pd.read_excel("data/FOMC_Dissents_Data.xlsx", skiprows=3)
votes["date"] = pd.to_datetime(votes["FOMC Meeting"])

dissent_records = []
for _, row in votes.iterrows():
    for col, direction in [("Dissenters Tighter", "tighter"),
                            ("Dissenters Easier", "easier")]:
        if pd.notna(row.get(col)):
            for name in str(row[col]).split(", "):
                dissent_records.append({
                    "date": row["date"],
                    "name": name.strip().upper(),
                    "direction": direction
                })

dissent_df = pd.DataFrame(dissent_records)
dissent_df = dissent_df[(dissent_df['date'].dt.year >= 1994) & (dissent_df['date'].dt.year <= 2020)]

# Match dissents
def match_dissent(row):
    speaker_upper = row['speaker'].upper()
    for _, d in dissent_df[dissent_df['date'] == row['date']].iterrows():
        if d['name'] in speaker_upper:
            return d['direction']
    return 'none'

v8['dissent_direction'] = v8.apply(match_dissent, axis=1)

# Load actual transcript text
def get_transcript_text(speaker, date):
    date_str = date.strftime('%Y%m%d')
    pattern = f"{TRANSCRIPTS_DIR}/{date_str}*_t.csv"
    files = glob.glob(pattern)
    if not files:
        return None
    df = pd.read_csv(files[0])
    text_col = 'clean_transcript_text' if 'clean_transcript_text' in df.columns else 'transcript_text'
    speaker_col = 'Speaker' if 'Speaker' in df.columns else 'speaker'
    last_name = speaker.split()[-1].upper()
    matches = df[df[speaker_col].str.upper().str.contains(last_name, na=False)]
    if len(matches) == 0:
        return None
    full_text = ' '.join(matches[text_col].dropna().tolist())
    # Return last 500 chars (policy go-around, most relevant)
    return full_text[-500:] if len(full_text) > 500 else full_text

print("=" * 70)
print("EXAMPLE SCORED TRANSCRIPTS FOR APPENDIX D")
print("=" * 70)

# 1. Strong hawk who dissented
hawks = v8[(v8['gpt_dissent_direction'] <= -6) & (v8['dissent_direction'] == 'tighter')]
hawks = hawks.sort_values('gpt_dissent_direction')

print("\n" + "=" * 70)
print("EXAMPLE 1: STRONG HAWKISH DISSENT")
print("=" * 70)
if len(hawks) > 0:
    ex = hawks.iloc[0]
    text = get_transcript_text(ex['speaker'], ex['date'])
    print(f"Speaker: {ex['speaker']}")
    print(f"Date: {ex['date'].strftime('%B %d, %Y')}")
    print(f"District: {ex['district']}")
    print(f"GPT Score: {ex['gpt_dissent_direction']:+.0f}")
    print(f"Vote: Dissented for tighter policy")
    print(f"\nExcerpt (last ~500 chars of remarks):")
    print(f'"{text}"' if text else "  [transcript not found]")

# 2. Strong dove who dissented
doves = v8[(v8['gpt_dissent_direction'] >= 6) & (v8['dissent_direction'] == 'easier')]
doves = doves.sort_values('gpt_dissent_direction', ascending=False)

print("\n" + "=" * 70)
print("EXAMPLE 2: STRONG DOVISH DISSENT")
print("=" * 70)
if len(doves) > 0:
    ex = doves.iloc[0]
    text = get_transcript_text(ex['speaker'], ex['date'])
    print(f"Speaker: {ex['speaker']}")
    print(f"Date: {ex['date'].strftime('%B %d, %Y')}")
    print(f"District: {ex['district']}")
    print(f"GPT Score: {ex['gpt_dissent_direction']:+.0f}")
    print(f"Vote: Dissented for easier policy")
    print(f"\nExcerpt:")
    print(f'"{text}"' if text else "  [transcript not found]")

# 3. Neutral / agreement
neutrals = v8[(v8['gpt_dissent_direction'] == 0) & (v8['dissent_direction'] == 'none')]
neutrals = neutrals.sample(1, random_state=42)

print("\n" + "=" * 70)
print("EXAMPLE 3: AGREEMENT / NEUTRAL")
print("=" * 70)
if len(neutrals) > 0:
    ex = neutrals.iloc[0]
    text = get_transcript_text(ex['speaker'], ex['date'])
    print(f"Speaker: {ex['speaker']}")
    print(f"Date: {ex['date'].strftime('%B %d, %Y')}")
    print(f"District: {ex['district']}")
    print(f"GPT Score: {ex['gpt_dissent_direction']:+.0f}")
    print(f"Vote: Agreed with majority")
    print(f"\nExcerpt:")
    print(f'"{text}"' if text else "  [transcript not found]")

# 4. Hidden dissent - high score but voted with majority
hidden = v8[(v8['gpt_dissent_direction'].abs() >= 5) & (v8['dissent_direction'] == 'none')]
hidden = hidden.sort_values('gpt_dissent_direction')

print("\n" + "=" * 70)
print("EXAMPLE 4: HIDDEN DISSENT (hawkish speech, no dissent vote)")
print("=" * 70)
if len(hidden) > 0:
    ex = hidden.iloc[0]
    text = get_transcript_text(ex['speaker'], ex['date'])
    print(f"Speaker: {ex['speaker']}")
    print(f"Date: {ex['date'].strftime('%B %d, %Y')}")
    print(f"District: {ex['district']}")
    print(f"GPT Score: {ex['gpt_dissent_direction']:+.0f}")
    print(f"Vote: Agreed with majority (no dissent)")
    print(f"\nExcerpt:")
    print(f'"{text}"' if text else "  [transcript not found]")

print("\n" + "=" * 70)
print("Done! Copy the examples above into Appendix D.")
print("=" * 70)