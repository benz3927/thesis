#!/usr/bin/env python3
"""
Step 2: Compute semantic dissent scores using pre-computed embeddings.
This is FAST - no API calls needed!

INPUT: data/cache/transcript_embeddings.pkl
OUTPUT: data/cache/semantic_dissent_scores.pkl

Computes dissent using weighted average aggregation:
- Aggregates statement embeddings to speaker-meeting level
- Weights by word count (longer statements = more weight)
- Computes dissent as distance from meeting consensus
"""

import pandas as pd
import numpy as np
import pickle
from scipy.spatial.distance import cosine
from tqdm import tqdm
import os

CACHE_DIR = 'data/cache'

print("="*80)
print("STEP 2: COMPUTING SEMANTIC DISSENT")
print("="*80)

# ============================================================================
# LOAD PRE-COMPUTED EMBEDDINGS
# ============================================================================

print("\n[1] Loading embeddings...")
with open(f'{CACHE_DIR}/transcript_embeddings.pkl', 'rb') as f:
    transcripts = pickle.load(f)

print(f"✅ Loaded {len(transcripts):,} statements with embeddings")
print(f"   Columns: {list(transcripts.columns)}")
print(f"   Embedding dimension: {transcripts['embedding'].iloc[0].shape[0]}")
print(f"   Date range: {transcripts['date'].min()} to {transcripts['date'].max()}")

# ============================================================================
# AGGREGATE TO SPEAKER-MEETING LEVEL
# ============================================================================

print("\n[2] Aggregating statements to speaker-meeting level...")

# Add word count for weighting
transcripts['word_count'] = transcripts['text'].str.split().str.len()

def aggregate_speaker_embedding(group_df):
    """
    Aggregate statement-level embeddings to one speaker-meeting embedding.
    Uses weighted average by word count.
    
    Args:
        group_df: All statements by one speaker in one meeting
        
    Returns:
        Dictionary with aggregated data
    """
    # Get embeddings as array
    embeddings = np.array(group_df['embedding'].tolist())
    
    # Get weights (word counts normalized)
    weights = group_df['word_count'].values
    weights = weights / weights.sum()  # Normalize to sum to 1
    
    # Weighted average of embeddings
    aggregated_emb = np.average(embeddings, axis=0, weights=weights)
    
    return {
        'embedding': aggregated_emb,
        'total_words': group_df['word_count'].sum(),
        'num_statements': len(group_df),
        'district': group_df['district'].iloc[0],
        'is_bank_president': group_df['is_bank_president'].iloc[0],
        'speaker': group_df['speaker'].iloc[0],
        'date': group_df['date'].iloc[0]
    }

# Group by speaker-meeting and aggregate
print("   Aggregating embeddings by speaker-meeting...")
speaker_meeting_list = []

grouped = transcripts.groupby(['date', 'speaker'])
for (date, speaker), group_df in tqdm(grouped, desc="   Progress"):
    agg_data = aggregate_speaker_embedding(group_df)
    speaker_meeting_list.append(agg_data)

speaker_meeting_df = pd.DataFrame(speaker_meeting_list)

print(f"\n✅ Aggregated to {len(speaker_meeting_df):,} speaker-meeting pairs")
print(f"   Average statements per speaker-meeting: {speaker_meeting_df['num_statements'].mean():.1f}")
print(f"   Average words per speaker-meeting: {speaker_meeting_df['total_words'].mean():.0f}")

# ============================================================================
# COMPUTE DISSENT SCORES
# ============================================================================

print("\n[3] Computing dissent scores by meeting...")

def compute_dissent_for_meeting(meeting_df):
    """
    Compute semantic dissent scores for all speakers at a single meeting.
    
    Dissent measures:
    1. dissent_consensus: Distance from mean of all speakers in meeting
    2. dissent_chair: Distance from Fed Chair's position
    
    Args:
        meeting_df: All speakers in one meeting (with aggregated embeddings)
        
    Returns:
        DataFrame with dissent scores added
    """
    # Get embeddings for all speakers in this meeting
    embeddings = np.array(meeting_df['embedding'].tolist())
    
    # Consensus embedding (mean of all speakers)
    consensus_emb = np.mean(embeddings, axis=0)
    
    # Find Fed Chair's embedding
    chair_names = ['greenspan', 'bernanke', 'yellen']
    chair_emb = None
    
    for idx, row in meeting_df.iterrows():
        speaker_lower = row['speaker'].lower()
        if any(chair in speaker_lower for chair in chair_names):
            chair_emb = embeddings[meeting_df.index.get_loc(idx)]
            break
    
    # If no chair found, use consensus as fallback
    if chair_emb is None:
        chair_emb = consensus_emb
    
    # Compute dissent scores (cosine distance)
    dissent_consensus = [cosine(emb, consensus_emb) for emb in embeddings]
    dissent_chair = [cosine(emb, chair_emb) for emb in embeddings]
    
    # Add to dataframe
    meeting_df = meeting_df.copy()
    meeting_df['dissent_consensus'] = dissent_consensus
    meeting_df['dissent_chair'] = dissent_chair
    meeting_df['num_speakers_in_meeting'] = len(meeting_df)
    
    return meeting_df

# Process each meeting
meeting_dates = sorted(speaker_meeting_df['date'].unique())
all_results = []

for date in tqdm(meeting_dates, desc="   Processing meetings"):
    meeting_df = speaker_meeting_df[speaker_meeting_df['date'] == date].copy()
    dissent_df = compute_dissent_for_meeting(meeting_df)
    all_results.append(dissent_df)

# Consolidate
dissent_data = pd.concat(all_results, ignore_index=True)

print(f"\n✅ Computed dissent scores for {len(dissent_data):,} speaker-meeting pairs")

# ============================================================================
# STATISTICS & VALIDATION
# ============================================================================

print("\n[4] Dissent statistics...")

print(f"\n   Overall statistics:")
print(f"      Mean dissent from consensus: {dissent_data['dissent_consensus'].mean():.4f}")
print(f"      Std dissent from consensus:  {dissent_data['dissent_consensus'].std():.4f}")
print(f"      Min dissent from consensus:  {dissent_data['dissent_consensus'].min():.4f}")
print(f"      Max dissent from consensus:  {dissent_data['dissent_consensus'].max():.4f}")

print(f"\n   Dissent from Chair:")
print(f"      Mean: {dissent_data['dissent_chair'].mean():.4f}")
print(f"      Std:  {dissent_data['dissent_chair'].std():.4f}")

# Statistics by bank presidents vs. others
if 'is_bank_president' in dissent_data.columns:
    bank_pres = dissent_data[dissent_data['is_bank_president'] == True]
    others = dissent_data[dissent_data['is_bank_president'] == False]
    
    print(f"\n   Bank Presidents:")
    print(f"      N = {len(bank_pres)}")
    print(f"      Mean dissent: {bank_pres['dissent_consensus'].mean():.4f}")
    
    print(f"\n   Board Governors/Others:")
    print(f"      N = {len(others)}")
    print(f"      Mean dissent: {others['dissent_consensus'].mean():.4f}")
    
    diff = bank_pres['dissent_consensus'].mean() - others['dissent_consensus'].mean()
    print(f"\n   Difference: {diff:.4f}")
    if abs(diff) > 0.01:
        direction = "MORE" if diff > 0 else "LESS"
        print(f"   → Bank presidents show {direction} semantic dissent on average")

# Distribution by meeting
meeting_stats = dissent_data.groupby('date')['dissent_consensus'].agg(['mean', 'std', 'count'])
print(f"\n   Dissent by meeting:")
print(f"      Average speakers per meeting: {meeting_stats['count'].mean():.1f}")
print(f"      Most contentious meeting: {meeting_stats['mean'].idxmax()} (avg dissent: {meeting_stats['mean'].max():.4f})")
print(f"      Most consensus meeting: {meeting_stats['mean'].idxmin()} (avg dissent: {meeting_stats['mean'].min():.4f})")

# Top dissenters
if len(dissent_data) > 0:
    top_dissenters = dissent_data.nlargest(10, 'dissent_consensus')[
        ['date', 'speaker', 'dissent_consensus', 'district', 'total_words']
    ]
    print(f"\n   Top 10 most dissenting speaker-meetings:")
    print(top_dissenters.to_string(index=False))

# ============================================================================
# SAVE RESULTS
# ============================================================================

print("\n[5] Saving results...")

output_file = f'{CACHE_DIR}/semantic_dissent_scores.pkl'
with open(output_file, 'wb') as f:
    pickle.dump(dissent_data, f)

print(f"✅ Saved: {output_file}")
print(f"   Rows: {len(dissent_data):,}")
print(f"   Columns: {list(dissent_data.columns)}")

# Also save as CSV (without embeddings) for inspection
dissent_csv = dissent_data.drop(columns=['embedding'])
csv_file = f'{CACHE_DIR}/semantic_dissent_scores.csv'
dissent_csv.to_csv(csv_file, index=False)
print(f"✅ Saved CSV (without embeddings): {csv_file}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("✅ SEMANTIC DISSENT COMPUTATION COMPLETE!")
print("="*80)

print(f"\n📊 Output Summary:")
print(f"   Speaker-meeting pairs: {len(dissent_data):,}")
print(f"   Meetings covered: {dissent_data['date'].nunique()}")
print(f"   Unique speakers: {dissent_data['speaker'].nunique()}")
print(f"   Date range: {dissent_data['date'].min()} to {dissent_data['date'].max()}")

print(f"\n📈 Key Findings:")
print(f"   Average dissent from consensus: {dissent_data['dissent_consensus'].mean():.4f}")
print(f"   Range: [{dissent_data['dissent_consensus'].min():.4f}, {dissent_data['dissent_consensus'].max():.4f}]")

if 'is_bank_president' in dissent_data.columns:
    bank_pres_mean = dissent_data[dissent_data['is_bank_president']]['dissent_consensus'].mean()
    print(f"   Bank presidents mean dissent: {bank_pres_mean:.4f}")

print(f"\n🎯 Next Steps:")
print(f"   1. Run: python compute_concept_similarity.py")
print(f"   2. Run: python regional_analysis.py")

print("\n" + "="*80)