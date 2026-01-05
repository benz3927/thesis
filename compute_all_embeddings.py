#!/usr/bin/env python3
"""
Compute all embeddings from scratch for unemployment-dissent analysis.

This script computes semantic similarity scores by:
1. Getting embeddings for unemployment and dissent concepts
2. Getting embeddings for each speaker's statement
3. Computing similarity scores to measure discussion topics
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
print("COMPUTING UNEMPLOYMENT-DISSENT EMBEDDINGS")
print("=" * 80)

# ============================================================================
# SEMANTIC CONCEPTS
# ============================================================================

UNEMPLOYMENT_CONCEPTS = {
    'high_unemployment': 'Unemployment is elevated and labor market conditions are weak',
    'labor_market_slack': 'There is substantial slack in the labor market',
    'job_losses': 'Job losses have increased and employment has declined',
    'weak_hiring': 'Hiring has slowed and employers are cautious',
    'regional_weakness': 'Economic conditions in my region have deteriorated',
}

DISSENT_CONCEPTS = {
    'dovish_dissent': 'Policy should be more accommodative to support employment',
    'hawkish_dissent': 'Policy should be tighter to address inflation risks',
    'disagree_pace': 'I disagree with the pace of policy adjustment',
    'regional_concern': 'Conditions in my district warrant different policy',
}

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
print("\n[STEP 3/4] Computing concept embeddings...")

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

# Get embeddings for unemployment concepts
print("   Computing unemployment concept embeddings...")
unemp_embeddings = {}
for concept, text in UNEMPLOYMENT_CONCEPTS.items():
    unemp_embeddings[concept] = get_embedding(text)

# Get embeddings for dissent concepts
print("   Computing dissent concept embeddings...")
dissent_embeddings = {}
for concept, text in DISSENT_CONCEPTS.items():
    dissent_embeddings[concept] = get_embedding(text)

print(f"✅ Computed {len(unemp_embeddings)} unemployment concept embeddings")
print(f"✅ Computed {len(dissent_embeddings)} dissent concept embeddings")

# ============================================================================
# STEP 4: COMPUTE STATEMENT SCORES (THIS IS THE SLOW PART)
# ============================================================================
print("\n[STEP 4/4] Computing semantic scores for all statements...")
print(f"   Processing {len(transcripts_2006_2017):,} statements")
print(f"   ⚠️  This will take 10-30 minutes and cost ~$0.50-2.00")
print(f"   Progress:")

unemp_scores = []
dissent_scores = []

for idx, row in tqdm(transcripts_2006_2017.iterrows(), total=len(transcripts_2006_2017)):
    if pd.isna(row['text']) or len(row['text']) < 50:
        unemp_scores.append(np.nan)
        dissent_scores.append(np.nan)
        continue

    # Get embedding for this statement
    statement_emb = get_embedding(row['text'])
    
    if statement_emb is None:
        unemp_scores.append(np.nan)
        dissent_scores.append(np.nan)
        continue

    # Compute similarity to unemployment concepts
    unemp_sims = []
    for concept_emb in unemp_embeddings.values():
        sim = 1 - cosine(statement_emb, concept_emb)
        unemp_sims.append(sim)
    unemp_scores.append(np.mean(unemp_sims))

    # Compute similarity to dissent concepts
    dissent_sims = []
    for concept_emb in dissent_embeddings.values():
        sim = 1 - cosine(statement_emb, concept_emb)
        dissent_sims.append(sim)
    dissent_scores.append(np.mean(dissent_sims))

# Add scores to dataframe
transcripts_2006_2017['unemployment_discussion_score'] = unemp_scores
transcripts_2006_2017['dissent_tone_score'] = dissent_scores

print(f"\n✅ Computed scores for {len(transcripts_2006_2017):,} statements")
print(f"   Avg unemployment score: {pd.Series(unemp_scores).mean():.4f}")
print(f"   Avg dissent score: {pd.Series(dissent_scores).mean():.4f}")

# Save with scores
with open(f'{CACHE_DIR}/transcripts_with_scores_2006_2017.pkl', 'wb') as f:
    pickle.dump(transcripts_2006_2017, f)
print(f"💾 Saved: {CACHE_DIR}/transcripts_with_scores_2006_2017.pkl")

print("\n" + "=" * 80)
print("✅ EMBEDDINGS COMPLETE!")
print("=" * 80)
print(f"\n📦 Created file with {len(transcripts_2006_2017):,} scored statements")
print(f"   Ready for regression analysis with unemployment data")