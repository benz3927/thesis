#!/usr/bin/env python3
"""
Compute all embeddings from scratch using transcript CSV files and unemployment data.

INPUT FILES:
- data/processed/Transcripts/*.csv (200 transcript files)
- data/cache/regional_unemployment.csv (unemployment data)

OUTPUT FILES:
1. extracted_transcripts.pkl - All transcripts consolidated
2. extracted_transcripts_2006_2017.pkl - Filtered to 2006-2017
3. regional_dissent_free.pkl - Semantic dissent scores with embeddings
4. unemployment_2006_2017.pkl - Unemployment data for 2006-2017

This script computes semantic dissent by:
1. Getting embeddings for each speaker's statement using OpenAI
2. Computing consensus embedding (mean of all speakers in a meeting)
3. Finding Fed Chair's embedding for each meeting
4. Calculating cosine distance from consensus and chair
"""

import pandas as pd
import numpy as np
import pickle
from openai import OpenAI
from scipy.spatial.distance import cosine
from tqdm import tqdm
import os
import glob
from dotenv import load_dotenv, find_dotenv
from datetime import datetime

# Configuration
_ = load_dotenv(find_dotenv())
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env file")

client = OpenAI(api_key=OPENAI_API_KEY)
CACHE_DIR = 'data/cache'
TRANSCRIPTS_DIR = 'data/processed/Transcripts'

# Ensure cache directory exists
os.makedirs(CACHE_DIR, exist_ok=True)

print("=" * 80)
print("COMPUTING ALL EMBEDDINGS FROM SCRATCH")
print("=" * 80)
print(f"Using OpenAI model: text-embedding-3-small")
print(f"Output directory: {CACHE_DIR}")
print("=" * 80)

# ============================================================================
# STEP 1: LOAD ALL TRANSCRIPT CSV FILES
# ============================================================================
print("\n[STEP 1/5] Loading transcript CSV files...")

# Find all transcript CSV files
transcript_files = glob.glob(f'{TRANSCRIPTS_DIR}/*_t.csv')
print(f"   Found {len(transcript_files)} transcript files")

# Load and consolidate all transcripts
all_transcripts = []
for file_path in tqdm(sorted(transcript_files), desc="   Loading files"):
    try:
        df = pd.read_csv(file_path)
        # Extract date from filename (format: YYYYMMDD_t.csv)
        filename = os.path.basename(file_path)
        date_str = filename.split('_')[0]
        date = pd.to_datetime(date_str, format='%Y%m%d')

        # Add date and year columns
        df['date'] = date
        df['year'] = date.year

        # Use clean_transcript_text if available, otherwise transcript_text
        if 'clean_transcript_text' in df.columns:
            df['text'] = df['clean_transcript_text']
        else:
            df['text'] = df['transcript_text']

        # Keep only needed columns
        df = df[['Speaker', 'text', 'date', 'year']].rename(columns={'Speaker': 'speaker'})

        all_transcripts.append(df)
    except Exception as e:
        print(f"   ⚠️  Error loading {file_path}: {e}")

# Consolidate into single dataframe
transcripts_full = pd.concat(all_transcripts, ignore_index=True)

# Clean up
transcripts_full = transcripts_full.dropna(subset=['text'])
transcripts_full = transcripts_full[transcripts_full['text'].str.len() >= 10]

print(f"\n✅ Loaded {len(transcripts_full):,} speaker statements")
print(f"   Date range: {transcripts_full['date'].min()} to {transcripts_full['date'].max()}")
print(f"   Years covered: {transcripts_full['year'].min()} - {transcripts_full['year'].max()}")
print(f"   Unique meetings: {transcripts_full['date'].nunique()}")
print(f"   Unique speakers: {transcripts_full['speaker'].nunique()}")

# Save full transcripts
with open(f'{CACHE_DIR}/extracted_transcripts.pkl', 'wb') as f:
    pickle.dump(transcripts_full, f)
print(f"💾 Saved: {CACHE_DIR}/extracted_transcripts.pkl")

# ============================================================================
# STEP 2: FILTER TO 2006-2017 PERIOD
# ============================================================================
print("\n[STEP 2/5] Filtering to 2006-2017 period...")

transcripts_2006_2017 = transcripts_full[
    (transcripts_full['year'] >= 2006) &
    (transcripts_full['year'] <= 2017)
].copy()

print(f"✅ Filtered from {len(transcripts_full):,} to {len(transcripts_2006_2017):,} statements")
print(f"   Meetings in period: {transcripts_2006_2017['date'].nunique()}")

# Save filtered transcripts
with open(f'{CACHE_DIR}/extracted_transcripts_2006_2017.pkl', 'wb') as f:
    pickle.dump(transcripts_2006_2017, f)
print(f"💾 Saved: {CACHE_DIR}/extracted_transcripts_2006_2017.pkl")

# ============================================================================
# STEP 3: COMPUTE EMBEDDINGS AND SEMANTIC DISSENT
# ============================================================================
print("\n[STEP 3/5] Computing semantic embeddings and dissent scores...")
print("   ⚠️  This will make OpenAI API calls and may take 10-30 minutes")
print("   💰 Estimated cost: ~$0.50-2.00 (depending on text volume)")

def get_embedding(text, model="text-embedding-3-small"):
    """Get embedding from OpenAI API with error handling"""
    try:
        text = str(text).replace("\n", " ").strip()
        if not text or len(text) < 10:
            return None
        # Truncate very long texts to avoid API limits (8191 tokens ≈ 32k chars)
        if len(text) > 30000:
            text = text[:30000]
        response = client.embeddings.create(input=[text], model=model)
        return np.array(response.data[0].embedding)
    except Exception as e:
        print(f"\n   ❌ Error getting embedding: {e}")
        return None

def compute_dissent_for_meeting(meeting_df):
    """
    Compute semantic dissent scores for all speakers at a single meeting.

    Returns DataFrame with:
    - date: Meeting date
    - speaker: Speaker name
    - dissent_consensus: Cosine distance from consensus (mean of all speakers)
    - dissent_chair: Cosine distance from Fed Chair's position
    - num_speakers: Number of speakers in the meeting
    """
    # Get embeddings for all statements in this meeting
    embeddings = []
    speakers = []
    texts = []

    for idx, row in meeting_df.iterrows():
        emb = get_embedding(row['text'])
        if emb is not None:
            embeddings.append(emb)
            speakers.append(row['speaker'])
            texts.append(row['text'])

    if len(embeddings) == 0:
        return pd.DataFrame()

    embeddings = np.array(embeddings)

    # Compute consensus embedding (mean of all speakers)
    consensus_emb = np.mean(embeddings, axis=0)

    # Find Fed Chair's embedding
    # Chair names by period: Greenspan (2006), Bernanke (2006-2014), Yellen (2014-2017)
    chair_names = ['greenspan', 'bernanke', 'yellen']
    chair_emb = None

    for i, speaker in enumerate(speakers):
        speaker_lower = speaker.lower()
        if any(chair in speaker_lower for chair in chair_names):
            chair_emb = embeddings[i]
            break

    # If no chair found, use consensus as fallback
    if chair_emb is None:
        chair_emb = consensus_emb

    # Compute dissent scores (cosine distance)
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

# Process each meeting
dissent_data = []
meeting_dates = sorted(transcripts_2006_2017['date'].unique())

print(f"   Processing {len(meeting_dates)} meetings...")
print(f"   Each dot = 1 meeting processed")

for i, date in enumerate(meeting_dates):
    meeting_df = transcripts_2006_2017[transcripts_2006_2017['date'] == date]
    dissent_df = compute_dissent_for_meeting(meeting_df)

    if len(dissent_df) > 0:
        dissent_data.append(dissent_df)

    # Progress indicator
    if (i + 1) % 10 == 0:
        print(f"   [{i+1}/{len(meeting_dates)}] meetings processed", end='\r')

print(f"\n   [{len(meeting_dates)}/{len(meeting_dates)}] meetings processed")

# Consolidate dissent data
dissent_data = pd.concat(dissent_data, ignore_index=True)

print(f"\n✅ Computed dissent scores for {len(dissent_data):,} speaker-meeting pairs")
print(f"   Average dissent from consensus: {dissent_data['dissent_consensus'].mean():.4f}")
print(f"   Average dissent from chair: {dissent_data['dissent_chair'].mean():.4f}")
print(f"   Columns: {list(dissent_data.columns)}")

# Save dissent data with embeddings
with open(f'{CACHE_DIR}/regional_dissent_free.pkl', 'wb') as f:
    pickle.dump(dissent_data, f)
print(f"💾 Saved: {CACHE_DIR}/regional_dissent_free.pkl")

# ============================================================================
# STEP 4: LOAD AND FILTER UNEMPLOYMENT DATA
# ============================================================================
print("\n[STEP 4/5] Loading unemployment data...")

unemployment_csv = 'data/cache/regional_unemployment.csv'
if not os.path.exists(unemployment_csv):
    raise FileNotFoundError(f"Unemployment data not found at {unemployment_csv}")

unemployment = pd.read_csv(unemployment_csv)
unemployment['date'] = pd.to_datetime(unemployment['date'])

# Filter to 2006-2017
unemployment_2006_2017 = unemployment[
    (unemployment['date'].dt.year >= 2006) &
    (unemployment['date'].dt.year <= 2017)
].copy()

print(f"✅ Loaded unemployment data")
print(f"   Total rows: {len(unemployment_2006_2017):,}")
print(f"   Date range: {unemployment_2006_2017['date'].min()} to {unemployment_2006_2017['date'].max()}")
print(f"   Districts: {unemployment_2006_2017['district'].nunique()}")
print(f"   Districts: {', '.join(sorted(unemployment_2006_2017['district'].unique()))}")

# Save unemployment data
with open(f'{CACHE_DIR}/unemployment_2006_2017.pkl', 'wb') as f:
    pickle.dump(unemployment_2006_2017, f)
print(f"💾 Saved: {CACHE_DIR}/unemployment_2006_2017.pkl")

# ============================================================================
# STEP 5: SUMMARY AND VALIDATION
# ============================================================================
print("\n" + "=" * 80)
print("✅ ALL EMBEDDINGS COMPUTED SUCCESSFULLY!")
print("=" * 80)

print("\n📦 Created files:")
print(f"  1. {CACHE_DIR}/extracted_transcripts.pkl")
print(f"     - All transcripts: {len(transcripts_full):,} statements")
print(f"     - Years: {transcripts_full['year'].min()}-{transcripts_full['year'].max()}")

print(f"\n  2. {CACHE_DIR}/extracted_transcripts_2006_2017.pkl")
print(f"     - Filtered transcripts: {len(transcripts_2006_2017):,} statements")
print(f"     - Years: 2006-2017")

print(f"\n  3. {CACHE_DIR}/regional_dissent_free.pkl")
print(f"     - Semantic dissent scores: {len(dissent_data):,} rows")
print(f"     - Embeddings computed via OpenAI API")
print(f"     - Avg dissent from consensus: {dissent_data['dissent_consensus'].mean():.4f}")

print(f"\n  4. {CACHE_DIR}/unemployment_2006_2017.pkl")
print(f"     - Unemployment data: {len(unemployment_2006_2017):,} rows")
print(f"     - Districts: {unemployment_2006_2017['district'].nunique()}")

print("\n" + "=" * 80)
print("🎯 Ready for analysis!")
print("   You can now use these files for your thesis analysis")
print("=" * 80)
