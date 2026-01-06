#!/usr/bin/env python3
"""
Compute disagreement/dissent embeddings for FOMC transcripts.

This script computes semantic similarity scores by:
1. Getting embeddings for disagreement/dissent concepts
2. Getting embeddings for each speaker's statement
3. Computing similarity scores to measure dissent tone

Author: Benjamin Zhao
Date: January 2026
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

# Configuration
_ = load_dotenv(find_dotenv())
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found in .env file")

client = OpenAI(api_key=OPENAI_API_KEY)
CACHE_DIR = 'data/cache'
TRANSCRIPTS_DIR = 'data/processed/Transcripts'

os.makedirs(CACHE_DIR, exist_ok=True)

print("=" * 80)
print("COMPUTING DISAGREEMENT/DISSENT EMBEDDINGS")
print("Research Question: Do bank presidents from high-unemployment districts")
print("                   express more dissent in FOMC meetings?")
print("=" * 80)

# ============================================================================
# DISAGREEMENT/DISSENT CONCEPTS
# ============================================================================

DISAGREEMENT_CONCEPTS = {
    'express_disagreement': 'I disagree with the proposed policy decision',
    'voice_concerns': 'I have concerns about this approach and believe we should reconsider',
    'advocate_alternative': 'I think we should pursue a different course of action instead',
    'challenge_consensus': 'I question whether the committee consensus is appropriate',
    'dissenting_view': 'My view differs from the majority and I want to express my dissent',
    'reservation': 'I have reservations about this decision and am uncomfortable with it',
    'oppose': 'I oppose this policy choice and believe it is a mistake',
}

print("\nDisagreement concepts to measure:")
for concept, text in DISAGREEMENT_CONCEPTS.items():
    print(f"  - {concept}: '{text}'")

# ============================================================================
# STEP 1: LOAD TRANSCRIPTS
# ============================================================================
print("\n[STEP 1/4] Loading transcript CSV files...")

transcript_files = glob.glob(f'{TRANSCRIPTS_DIR}/*_t.csv')
print(f"   Found {len(transcript_files)} transcript files")

all_transcripts = []
for file_path in tqdm(sorted(transcript_files), desc="   Loading files"):
    try:
        df = pd.read_csv(file_path)
        filename = os.path.basename(file_path)
        date_str = filename.split('_')[0]
        date = pd.to_datetime(date_str, format='%Y%m%d')

        df['date'] = date
        df['year'] = date.year

        if 'clean_transcript_text' in df.columns:
            df['text'] = df['clean_transcript_text']
        else:
            df['text'] = df['transcript_text']

        df = df[['Speaker', 'text', 'date', 'year']].rename(columns={'Speaker': 'speaker'})
        all_transcripts.append(df)
    except Exception as e:
        print(f"   ⚠️  Error loading {file_path}: {e}")

transcripts_full = pd.concat(all_transcripts, ignore_index=True)
transcripts_full = transcripts_full.dropna(subset=['text'])
transcripts_full = transcripts_full[transcripts_full['text'].str.len() >= 10]

print(f"\n✅ Loaded {len(transcripts_full):,} speaker statements")

# Save full transcripts
with open(f'{CACHE_DIR}/extracted_transcripts.pkl', 'wb') as f:
    pickle.dump(transcripts_full, f)
print(f"💾 Saved: {CACHE_DIR}/extracted_transcripts.pkl")

# ============================================================================
# STEP 2: FILTER TO 2006-2017
# ============================================================================
print("\n[STEP 2/4] Filtering to 2006-2017 period...")

transcripts_2006_2017 = transcripts_full[
    (transcripts_full['year'] >= 2006) &
    (transcripts_full['year'] <= 2017)
].copy()

print(f"✅ Filtered to {len(transcripts_2006_2017):,} statements")

with open(f'{CACHE_DIR}/extracted_transcripts_2006_2017.pkl', 'wb') as f:
    pickle.dump(transcripts_2006_2017, f)
print(f"💾 Saved: {CACHE_DIR}/extracted_transcripts_2006_2017.pkl")

# ============================================================================
# STEP 3: COMPUTE CONCEPT EMBEDDINGS
# ============================================================================
print("\n[STEP 3/4] Computing disagreement concept embeddings...")

def get_embedding(text, model="text-embedding-3-small"):
    """Get embedding from OpenAI API"""
    try:
        text = str(text).replace("\n", " ").strip()
        if not text or len(text) < 10:
            return None
        if len(text) > 30000:
            text = text[:30000]
        response = client.embeddings.create(input=[text], model=model)
        return np.array(response.data[0].embedding)
    except Exception as e:
        print(f"\n   ❌ Error: {e}")
        return None

# Get embeddings for disagreement concepts
print("   Computing disagreement concept embeddings...")
disagreement_embeddings = {}
for concept, text in DISAGREEMENT_CONCEPTS.items():
    disagreement_embeddings[concept] = get_embedding(text)

print(f"✅ Computed {len(disagreement_embeddings)} disagreement concept embeddings")

# ============================================================================
# STEP 4: COMPUTE STATEMENT SCORES (THIS IS THE SLOW PART)
# ============================================================================
print("\n[STEP 4/4] Computing disagreement scores for all statements...")
print(f"   Processing {len(transcripts_2006_2017):,} statements")
print(f"   ⚠️  This will take 10-30 minutes and cost ~$0.50-2.00")
print(f"   Progress:")

disagreement_scores = []

for idx, row in tqdm(transcripts_2006_2017.iterrows(), total=len(transcripts_2006_2017)):
    if pd.isna(row['text']) or len(row['text']) < 50:
        disagreement_scores.append(np.nan)
        continue

    # Get embedding for this statement
    statement_emb = get_embedding(row['text'])
    
    if statement_emb is None:
        disagreement_scores.append(np.nan)
        continue

    # Compute similarity to disagreement concepts
    disagreement_sims = []
    for concept_emb in disagreement_embeddings.values():
        sim = 1 - cosine(statement_emb, concept_emb)
        disagreement_sims.append(sim)
    disagreement_scores.append(np.mean(disagreement_sims))

# Add scores to dataframe
transcripts_2006_2017['disagreement_score'] = disagreement_scores

print(f"\n✅ Computed scores for {len(transcripts_2006_2017):,} statements")
print(f"   Avg disagreement score: {pd.Series(disagreement_scores).mean():.4f}")
print(f"   Score range: {pd.Series(disagreement_scores).min():.4f} to {pd.Series(disagreement_scores).max():.4f}")

# Save with scores
with open(f'{CACHE_DIR}/transcripts_with_disagreement_scores_2006_2017.pkl', 'wb') as f:
    pickle.dump(transcripts_2006_2017, f)
print(f"💾 Saved: {CACHE_DIR}/transcripts_with_disagreement_scores_2006_2017.pkl")

# ============================================================================
# VALIDATION: Show high and low scoring examples
# ============================================================================
print("\n" + "=" * 80)
print("VALIDATION: Sample High and Low Disagreement Scores")
print("=" * 80)

valid_scores = transcripts_2006_2017[transcripts_2006_2017['disagreement_score'].notna()]

print("\n🔴 HIGHEST disagreement scores (should express dissent):")
top_dissent = valid_scores.nlargest(5, 'disagreement_score')
for i, (_, row) in enumerate(top_dissent.iterrows(), 1):
    print(f"\n[{i}] Score: {row['disagreement_score']:.4f}")
    print(f"    Speaker: {row['speaker']}")
    print(f"    Date: {row['date']}")
    print(f"    Text: {row['text'][:250]}...")

print("\n🟢 LOWEST disagreement scores (should express agreement):")
bottom_dissent = valid_scores.nsmallest(5, 'disagreement_score')
for i, (_, row) in enumerate(bottom_dissent.iterrows(), 1):
    print(f"\n[{i}] Score: {row['disagreement_score']:.4f}")
    print(f"    Speaker: {row['speaker']}")
    print(f"    Date: {row['date']}")
    print(f"    Text: {row['text'][:250]}...")

# Check if "disagree" keyword correlates with scores
disagree_mentions = valid_scores[valid_scores['text'].str.contains('disagree', case=False, na=False)]
if len(disagree_mentions) > 0:
    print(f"\n📊 Validation: Statements mentioning 'disagree':")
    print(f"   Count: {len(disagree_mentions)}")
    print(f"   Mean score: {disagree_mentions['disagreement_score'].mean():.4f}")
    print(f"   Overall mean: {valid_scores['disagreement_score'].mean():.4f}")
    if disagree_mentions['disagreement_score'].mean() > valid_scores['disagreement_score'].mean():
        print("   ✅ Statements with 'disagree' have HIGHER scores (GOOD)")
    else:
        print("   ⚠️  Statements with 'disagree' have LOWER scores (PROBLEM)")

print("\n" + "=" * 80)
print("✅ EMBEDDINGS COMPLETE!")
print("=" * 80)
print(f"\n📦 Created file with {len(transcripts_2006_2017):,} scored statements")
print(f"   Ready for regression analysis:")
print(f"   DV: disagreement_score (semantic measure of dissent)")
print(f"   IV: district unemployment rate (actual economic data)")