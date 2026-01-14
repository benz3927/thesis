#!/usr/bin/env python3
"""
Step 1: Compute and store embeddings for all transcript text.
Run this ONCE - it's expensive (~$1-2 in API costs).

This script:
1. Loads all FOMC transcript CSV files
2. Maps speakers to Federal Reserve districts
3. Computes embeddings for each text chunk via OpenAI API
4. Saves everything for later analysis

OUTPUT: data/cache/transcript_embeddings.pkl
Contains: date, year, speaker, text, district, is_bank_president, embedding (1536-dim vector)
"""

import pandas as pd
import numpy as np
import pickle
from openai import OpenAI
from tqdm import tqdm
import os
import glob
from dotenv import load_dotenv, find_dotenv
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

# Load environment variables
_ = load_dotenv(find_dotenv())
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env file")

client = OpenAI(api_key=OPENAI_API_KEY)

# Directories
CACHE_DIR = 'data/cache'
TRANSCRIPTS_DIR = 'data/processed/Transcripts'

# Ensure cache directory exists
os.makedirs(CACHE_DIR, exist_ok=True)

print("="*80)
print("STEP 1: COMPUTING ALL EMBEDDINGS")
print("="*80)
print(f"Using OpenAI model: text-embedding-3-small")
print(f"Output directory: {CACHE_DIR}")
print("="*80)

# ============================================================================
# DISTRICT MAPPING CONFIGURATION
# ============================================================================

# Regional Bank Presidents by district (using last names to match data)
# This is based on Federal Reserve Bank structure (2006-2017 period)
DISTRICT_MAPPING = {
    # Boston (1st District)
    'minehan': 'Boston',
    'rosengren': 'Boston',
    
    # New York (2nd District)
    'geithner': 'New York',
    'dudley': 'New York',
    
    # Philadelphia (3rd District)
    'plosser': 'Philadelphia',
    'harker': 'Philadelphia',
    
    # Cleveland (4th District)
    'pianalto': 'Cleveland',
    'mester': 'Cleveland',
    
    # Richmond (5th District)
    'lacker': 'Richmond',
    'barkin': 'Richmond',
    
    # Atlanta (6th District)
    'lockhart': 'Atlanta',
    'bostic': 'Atlanta',
    'guynn': 'Atlanta',
    
    # Chicago (7th District)
    'moskow': 'Chicago',
    'evans': 'Chicago',
    
    # St. Louis (8th District)
    'poole': 'St. Louis',
    'bullard': 'St. Louis',
    
    # Minneapolis (9th District)
    'stern': 'Minneapolis',
    'kocherlakota': 'Minneapolis',
    'kashkari': 'Minneapolis',
    
    # Kansas City (10th District)
    'hoenig': 'Kansas City',
    'george': 'Kansas City',
    
    # Dallas (11th District)
    'fisher': 'Dallas',
    'kaplan': 'Dallas',
    
    # San Francisco (12th District)
    'yellen': 'San Francisco',
    'williams': 'San Francisco',
    'daly': 'San Francisco',
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_embedding(text, model="text-embedding-3-small"):
    """
    Get embedding from OpenAI API with error handling and text preprocessing.
    
    Args:
        text: Text to embed
        model: OpenAI embedding model to use
        
    Returns:
        numpy array of embedding (1536-dim) or None if failed
    """
    try:
        # Clean and validate text
        text = str(text).replace("\n", " ").strip()
        
        if not text or len(text) < 10:
            return None
            
        # Truncate very long texts to avoid API limits 
        # (8191 tokens ≈ 32k chars for safety)
        if len(text) > 30000:
            text = text[:30000]
            
        # Call OpenAI API
        response = client.embeddings.create(input=[text], model=model)
        return np.array(response.data[0].embedding)
        
    except Exception as e:
        print(f"\n   ❌ Error getting embedding: {e}")
        return None


def map_speaker_to_district(speaker_name):
    """
    Map speaker name to district using partial matching.
    Handles formats like "CHAIR YELLEN", "MR LOCKHART", "VICE CHAIRMAN DUDLEY"
    
    Args:
        speaker_name: Speaker name from CSV (e.g., "CHAIR YELLEN")
        
    Returns:
        District name (e.g., "San Francisco") or None
    """
    if pd.isna(speaker_name):
        return None
    
    # Convert to lowercase for matching
    speaker_lower = str(speaker_name).lower().strip()
    
    # Check if any last name from DISTRICT_MAPPING appears in the speaker name
    for last_name, district in DISTRICT_MAPPING.items():
        if last_name in speaker_lower:
            return district
    
    return None


# ============================================================================
# STEP 1: LOAD ALL TRANSCRIPT CSV FILES
# ============================================================================

print("\n[STEP 1/4] Loading transcript CSV files...")

# Find all transcript CSV files
transcript_files = glob.glob(f'{TRANSCRIPTS_DIR}/*_t.csv')
print(f"   Found {len(transcript_files)} transcript files")

if len(transcript_files) == 0:
    raise FileNotFoundError(f"No transcript files found in {TRANSCRIPTS_DIR}")

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
        elif 'transcript_text' in df.columns:
            df['text'] = df['transcript_text']
        else:
            print(f"   ⚠️  Warning: No text column found in {file_path}")
            continue
        
        # Keep only needed columns and rename for consistency
        if 'Speaker' in df.columns:
            df = df[['Speaker', 'text', 'date', 'year']].rename(columns={'Speaker': 'speaker'})
        elif 'speaker' in df.columns:
            df = df[['speaker', 'text', 'date', 'year']]
        else:
            print(f"   ⚠️  Warning: No speaker column found in {file_path}")
            continue
        
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

# ============================================================================
# STEP 2: FILTER TO 2006-2017 PERIOD & MAP DISTRICTS
# ============================================================================

print("\n[STEP 2/4] Filtering to 2006-2017 and mapping speakers to districts...")

# Filter to analysis period
transcripts_2006_2017 = transcripts_full[
    (transcripts_full['year'] >= 2006) &
    (transcripts_full['year'] <= 2017)
].copy().reset_index(drop=True)

print(f"✅ Filtered from {len(transcripts_full):,} to {len(transcripts_2006_2017):,} statements")
print(f"   Meetings in period: {transcripts_2006_2017['date'].nunique()}")

# Map speakers to districts using the flexible matching function
print(f"\n   Mapping speakers to Federal Reserve districts...")

transcripts_2006_2017['district'] = transcripts_2006_2017['speaker'].apply(map_speaker_to_district)
transcripts_2006_2017['is_bank_president'] = transcripts_2006_2017['district'].notna()

# Statistics
num_bank_president_statements = transcripts_2006_2017['is_bank_president'].sum()
num_bank_presidents = transcripts_2006_2017[transcripts_2006_2017['is_bank_president']]['speaker'].nunique()
num_districts = transcripts_2006_2017['district'].nunique()

print(f"✅ District mapping complete:")
print(f"   Regional Bank President statements: {num_bank_president_statements:,} ({num_bank_president_statements/len(transcripts_2006_2017)*100:.1f}%)")
print(f"   Unique bank presidents: {num_bank_presidents}")
print(f"   Districts represented: {num_districts}")

if num_districts > 0:
    print(f"   Districts: {', '.join(sorted(transcripts_2006_2017['district'].dropna().unique()))}")
    
    # Show sample of mapped data
    print(f"\n   Sample of mapped speakers:")
    sample = (
        transcripts_2006_2017[transcripts_2006_2017['is_bank_president']]
        [['speaker', 'district']]
        .drop_duplicates()
        .head(15)
    )
    for _, row in sample.iterrows():
        print(f"      {row['speaker']:30s} → {row['district']}")
else:
    print(f"\n   ⚠️  WARNING: No bank presidents mapped!")
    print(f"   This is a problem. Check the speaker names and mapping.")
    exit()

# ============================================================================
# STEP 3: COMPUTE EMBEDDINGS
# ============================================================================

print("\n[STEP 3/4] Computing embeddings via OpenAI API...")
print(f"   ⚠️  This will make ~{len(transcripts_2006_2017):,} API calls")
print(f"   ⚠️  Estimated time: 10-30 minutes")
print(f"   💰 Estimated cost: $0.50-2.00")
print(f"   📊 Each embedding is 1536-dimensional")

# Ask for confirmation
user_input = input("\n   Proceed with embedding computation? (yes/no): ")
if user_input.lower() not in ['yes', 'y']:
    print("\n   ❌ Aborted by user")
    exit()

print(f"\n   Computing embeddings...")

embeddings = []
failed_count = 0

for idx, row in tqdm(transcripts_2006_2017.iterrows(), 
                     total=len(transcripts_2006_2017),
                     desc="   Progress"):
    emb = get_embedding(row['text'])
    embeddings.append(emb)
    
    if emb is None:
        failed_count += 1

# Add embeddings to dataframe
transcripts_2006_2017['embedding'] = embeddings

# Remove rows where embedding failed
original_count = len(transcripts_2006_2017)
transcripts_2006_2017 = transcripts_2006_2017[
    transcripts_2006_2017['embedding'].notna()
].copy()

print(f"\n✅ Embedding computation complete!")
print(f"   Successfully embedded: {len(transcripts_2006_2017):,} statements")
print(f"   Failed: {failed_count} ({failed_count/original_count*100:.1f}%)")
print(f"   Embedding dimension: {transcripts_2006_2017['embedding'].iloc[0].shape[0]}")

# Verify embedding statistics
sample_embedding = transcripts_2006_2017['embedding'].iloc[0]
print(f"\n   Embedding statistics (first embedding as sample):")
print(f"      Mean: {np.mean(sample_embedding):.4f}")
print(f"      Std:  {np.std(sample_embedding):.4f}")
print(f"      Min:  {np.min(sample_embedding):.4f}")
print(f"      Max:  {np.max(sample_embedding):.4f}")

# ============================================================================
# STEP 4: SAVE RESULTS
# ============================================================================

print("\n[STEP 4/4] Saving results...")

# Verify we have all required columns
required_columns = ['date', 'year', 'speaker', 'text', 'district', 
                   'is_bank_president', 'embedding']
missing_columns = [col for col in required_columns if col not in transcripts_2006_2017.columns]

if missing_columns:
    print(f"   ⚠️  Warning: Missing columns: {missing_columns}")

# Save embeddings
output_file = f'{CACHE_DIR}/transcript_embeddings.pkl'
with open(output_file, 'wb') as f:
    pickle.dump(transcripts_2006_2017, f)

print(f"✅ Saved: {output_file}")
print(f"   File size: {os.path.getsize(output_file) / 1024 / 1024:.1f} MB")
print(f"   Rows: {len(transcripts_2006_2017):,}")
print(f"   Columns: {list(transcripts_2006_2017.columns)}")

# Also save a version without embeddings for inspection
transcripts_no_embedding = transcripts_2006_2017.drop(columns=['embedding'])
csv_file = f'{CACHE_DIR}/transcript_metadata.csv'
transcripts_no_embedding.to_csv(csv_file, index=False)
print(f"✅ Saved metadata (without embeddings): {csv_file}")

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================

print("\n" + "="*80)
print("✅ EMBEDDING COMPUTATION COMPLETE!")
print("="*80)

print(f"\n📊 Dataset Summary:")
print(f"   Total statements: {len(transcripts_2006_2017):,}")
print(f"   Time period: {transcripts_2006_2017['date'].min()} to {transcripts_2006_2017['date'].max()}")
print(f"   Unique meetings: {transcripts_2006_2017['date'].nunique()}")
print(f"   Unique speakers: {transcripts_2006_2017['speaker'].nunique()}")

print(f"\n📍 Regional Bank Presidents:")
bank_pres_data = transcripts_2006_2017[transcripts_2006_2017['is_bank_president']]
print(f"   Statements: {len(bank_pres_data):,} ({len(bank_pres_data)/len(transcripts_2006_2017)*100:.1f}%)")
print(f"   Unique speakers: {bank_pres_data['speaker'].nunique()}")
print(f"   Districts: {bank_pres_data['district'].nunique()}")

print(f"\n   Statements by district:")
district_counts = bank_pres_data['district'].value_counts().sort_index()
for district, count in district_counts.items():
    print(f"      {district:15s}: {count:4,} statements")

print(f"\n🎯 Next Steps:")
print(f"   1. Run: python compute_semantic_dissent.py")
print(f"   2. Run: python compute_concept_similarity.py")
print(f"   3. Run: python regional_analysis.py")

print("\n" + "="*80)