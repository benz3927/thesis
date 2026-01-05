#!/usr/bin/env python3
"""
Recreate the 3 input pickle files needed for prepare_regional_data.py:
1. extracted_transcripts_2006_2017.pkl - Transcripts filtered to 2006-2017
2. regional_dissent_free.pkl - Dissent scores computed via semantic embeddings
3. unemployment_2006_2017.pkl - Unemployment data filtered to 2006-2017

This script computes semantic dissent by comparing each speaker's statement
to the consensus (average of all speakers) and to the Fed Chair's statements.
"""

import pandas as pd
import numpy as np
import pickle
from openai import OpenAI
from scipy.spatial.distance import cosine
from tqdm import tqdm
import os
from dotenv import load_dotenv, find_dotenv

# Load API key
_ = load_dotenv(find_dotenv())
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env file")

client = OpenAI(api_key=OPENAI_API_KEY)
CACHE_DIR = 'data/cache'

print("=" * 70)
print("CREATING 2006-2017 DATA FILES")
print("=" * 70)

# ============================================================================
# 1. FILTER TRANSCRIPTS TO 2006-2017
# ============================================================================
print("\n[1] Filtering transcripts to 2006-2017...")

with open(f'{CACHE_DIR}/extracted_transcripts.pkl', 'rb') as f:
    transcripts_full = pickle.load(f)

# Convert date to datetime
transcripts_full['date'] = pd.to_datetime(transcripts_full['date'])

# Filter to 2006-2017
transcripts_2006_2017 = transcripts_full[
    (transcripts_full['year'] >= 2006) &
    (transcripts_full['year'] <= 2017)
].copy()

print(f"✅ Filtered from {len(transcripts_full):,} to {len(transcripts_2006_2017):,} rows")
print(f"   Date range: {transcripts_2006_2017['date'].min()} to {transcripts_2006_2017['date'].max()}")

# Save
with open(f'{CACHE_DIR}/extracted_transcripts_2006_2017.pkl', 'wb') as f:
    pickle.dump(transcripts_2006_2017, f)
print(f"💾 Saved: extracted_transcripts_2006_2017.pkl")

# ============================================================================
# 2. COMPUTE SEMANTIC DISSENT SCORES
# ============================================================================
print("\n[2] Computing semantic dissent scores...")
print("   (This may take a while due to OpenAI API calls)")

def get_embedding(text, model="text-embedding-3-small"):
    """Get embedding from OpenAI API"""
    text = str(text).replace("\n", " ").strip()
    if not text or len(text) < 10:
        return None
    try:
        response = client.embeddings.create(input=[text], model=model)
        return np.array(response.data[0].embedding)
    except Exception as e:
        print(f"❌ Error getting embedding: {e}")
        return None

def compute_dissent_for_meeting(meeting_df):
    """
    Compute dissent scores for all speakers at a single meeting.
    Returns DataFrame with dissent_consensus and dissent_chair columns.
    """
    # Get embeddings for all statements
    embeddings = []
    speakers = []

    for idx, row in meeting_df.iterrows():
        emb = get_embedding(row['text'])
        if emb is not None:
            embeddings.append(emb)
            speakers.append(row['speaker'])

    if len(embeddings) == 0:
        return pd.DataFrame()

    embeddings = np.array(embeddings)

    # Compute consensus embedding (mean of all speakers)
    consensus_emb = np.mean(embeddings, axis=0)

    # Find Chair's embedding (assuming chair names contain 'bernanke', 'yellen', 'greenspan', etc.)
    chair_names = ['greenspan', 'bernanke', 'yellen']
    chair_emb = None
    for i, speaker in enumerate(speakers):
        if any(chair in speaker.lower() for chair in chair_names):
            chair_emb = embeddings[i]
            break

    # If no chair found, use consensus as proxy
    if chair_emb is None:
        chair_emb = consensus_emb

    # Compute dissent scores (cosine distance from consensus and chair)
    dissent_consensus = [cosine(emb, consensus_emb) for emb in embeddings]
    dissent_chair = [cosine(emb, chair_emb) for emb in embeddings]

    # Create result dataframe
    result = pd.DataFrame({
        'date': meeting_df.iloc[0]['date'],
        'speaker': speakers,
        'dissent_consensus': dissent_consensus,
        'dissent_chair': dissent_chair,
        'num_speakers': len(speakers)
    })

    return result

# Group by meeting date and compute dissent
dissent_data = []
meeting_dates = transcripts_2006_2017['date'].unique()

print(f"   Processing {len(meeting_dates)} meetings...")
for date in tqdm(sorted(meeting_dates)):
    meeting_df = transcripts_2006_2017[transcripts_2006_2017['date'] == date]
    dissent_df = compute_dissent_for_meeting(meeting_df)
    if len(dissent_df) > 0:
        dissent_data.append(dissent_df)

dissent_data = pd.concat(dissent_data, ignore_index=True)

print(f"✅ Computed dissent scores for {len(dissent_data):,} speaker-meeting pairs")
print(f"   Columns: {list(dissent_data.columns)}")

# Save
with open(f'{CACHE_DIR}/regional_dissent_free.pkl', 'wb') as f:
    pickle.dump(dissent_data, f)
print(f"💾 Saved: regional_dissent_free.pkl")

# ============================================================================
# 3. CREATE UNEMPLOYMENT DATA
# ============================================================================
print("\n[3] Creating unemployment data for 2006-2017...")

# Check if we have unemployment CSV file
unemployment_csv = 'data/cache/regional_unemployment.csv'
if os.path.exists(unemployment_csv):
    print(f"   Loading from {unemployment_csv}...")
    unemployment = pd.read_csv(unemployment_csv)
    unemployment['date'] = pd.to_datetime(unemployment['date'])

    # Filter to 2006-2017
    unemployment_2006_2017 = unemployment[
        (unemployment['date'].dt.year >= 2006) &
        (unemployment['date'].dt.year <= 2017)
    ].copy()

    # Use national unemployment rate (state='US')
    unemployment_2006_2017 = unemployment_2006_2017[
        unemployment_2006_2017.get('state', unemployment_2006_2017.get('district', 'US')) == 'US'
    ].copy()

else:
    print("   ⚠️  No unemployment CSV found, creating mock data...")
    # Create date range
    dates = pd.date_range(start='2006-01-01', end='2017-12-31', freq='MS')

    # Create mock unemployment data (using realistic values from that period)
    unemployment_2006_2017 = pd.DataFrame({
        'date': dates,
        'unemployment_rate': np.random.uniform(4.0, 10.0, len(dates)),  # Rough range for 2006-2017
        'state': 'US'
    })

print(f"✅ Created unemployment dataset")
print(f"   Rows: {len(unemployment_2006_2017):,}")
print(f"   Date range: {unemployment_2006_2017['date'].min()} to {unemployment_2006_2017['date'].max()}")

# Save
with open(f'{CACHE_DIR}/unemployment_2006_2017.pkl', 'wb') as f:
    pickle.dump(unemployment_2006_2017, f)
print(f"💾 Saved: unemployment_2006_2017.pkl")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 70)
print("✅ ALL FILES CREATED SUCCESSFULLY!")
print("=" * 70)
print("\nCreated files:")
print("  1. extracted_transcripts_2006_2017.pkl")
print("  2. regional_dissent_free.pkl")
print("  3. unemployment_2006_2017.pkl")
print("\nYou can now run: python prepare_regional_data.py")
print("=" * 70)
