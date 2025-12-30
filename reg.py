#!/usr/bin/env python3
"""
Regional Context Analysis: Do Regional Bank Presidents with Higher Local 
Unemployment Dissent More? 

Uses semantic embeddings to measure:
1. How much each speaker discusses unemployment/economic weakness
2. Whether speakers from high-unemployment regions dissent more
3. Interaction between regional conditions and dissent behavior

Author: Benjamin Zhao
Date: November 2025
"""

import pandas as pd
import numpy as np
from openai import OpenAI
from scipy.spatial.distance import cosine
import statsmodels.api as sm
import pickle
from tqdm import tqdm
import os
from dotenv import load_dotenv, find_dotenv

# ============================================================================
# CONFIGURATION
# ============================================================================

OUTPUT_DIR = '/Users/CS/Documents/GitHub/thesis/data/processed/'
CACHE_DIR = '/Users/CS/Documents/GitHub/thesis/data/cache/'

# Load environment variables and initialize OpenAI client
_ = load_dotenv(find_dotenv())
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set.")

client = OpenAI(api_key=OPENAI_API_KEY)
print(f"✅ OpenAI API key loaded successfully")

# ============================================================================
# SEMANTIC CONCEPTS FOR ANALYSIS
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

print("="*70)
print("REGIONAL UNEMPLOYMENT VS. DISSENT ANALYSIS")
print("="*70)

# ============================================================================
# HELPER FUNCTION: GET OPENAI EMBEDDINGS
# ============================================================================

def get_embedding(text, model="text-embedding-3-small"):
    """Get embedding from OpenAI API"""
    text = text.replace("\n", " ")
    response = client.embeddings.create(input=[text], model=model)
    return np.array(response.data[0].embedding)

# ============================================================================
# LOAD DATA
# ============================================================================

print("\n[1] Loading data...")

# Load transcript data with speaker-level information
try:
    with open(f'{CACHE_DIR}/fomc_transcripts_speakers.pkl', 'rb') as f:
        transcripts_df = pickle.load(f)
    print(f"✅ Loaded {len(transcripts_df)} speaker turns from transcripts")
except:
    print("⚠️  Could not find speaker-level transcript data")
    print("    Expected file: fomc_transcripts_speakers.pkl")
    print("    Columns needed: date, speaker, text, district, is_dissent")
    raise

# Load regional unemployment data
try:
    regional_unemp = pd.read_csv(f'{CACHE_DIR}/regional_unemployment.csv')
    regional_unemp['date'] = pd.to_datetime(regional_unemp['date'])
    print(f"✅ Loaded regional unemployment data")
except:
    print("⚠️  Could not find regional unemployment data")
    print("    Expected file: regional_unemployment.csv")
    print("    Columns needed: date, district, unemployment_rate")
    raise

# Diagnostic: Check date ranges before merge
print(f"\n🔍 Pre-merge diagnostics:")
print(f"   Transcript dates: {transcripts_df['date'].min()} to {transcripts_df['date'].max()}")
print(f"   Unemployment dates: {regional_unemp['date'].min()} to {regional_unemp['date'].max()}")
print(f"   Transcript districts: {sorted(transcripts_df['district'].unique())}")
print(f"   Unemployment districts: {sorted(regional_unemp['district'].unique())}")

# FIX 1: Drop unemployment_rate from transcripts if it exists
if 'unemployment_rate' in transcripts_df.columns:
    transcripts_df = transcripts_df.drop('unemployment_rate', axis=1)
    print("✅ Dropped existing unemployment_rate column from transcripts")

# FIX 2: Create year-month keys for better merge coverage
# Check if year_month already exists in transcripts
if 'year_month' not in transcripts_df.columns:
    transcripts_df['year_month'] = transcripts_df['date'].dt.to_period('M')
    print("✅ Created year_month column in transcripts")

regional_unemp['year_month'] = regional_unemp['date'].dt.to_period('M')
print("✅ Created year_month column in unemployment data")

# Merge on year-month and district (more forgiving than exact date match)
df = pd.merge(transcripts_df, 
              regional_unemp[['year_month', 'district', 'unemployment_rate']], 
              on=['year_month', 'district'], 
              how='left')

# FIX 3: Convert is_dissent to integer for logistic regression
df['is_dissent'] = df['is_dissent'].astype(int)
print("✅ Converted is_dissent to integer (0/1)")

# Debug: Check merge results
print(f"\n📊 Post-merge diagnostics:")
print(f"   Total rows after merge: {len(df)}")
print(f"   Rows with unemployment_rate: {df['unemployment_rate'].notna().sum()}")
print(f"   Missing unemployment_rate: {df['unemployment_rate'].isna().sum()}")
if df['unemployment_rate'].notna().sum() > 0:
    print(f"   Unemployment rate range: {df['unemployment_rate'].min():.1f}% to {df['unemployment_rate'].max():.1f}%")
    print(f"   Mean unemployment rate: {df['unemployment_rate'].mean():.1f}%")

# Filter to only Regional Bank Presidents
bank_presidents = df[df['is_bank_president'] == True].copy()

print(f"\n✅ Working with {len(bank_presidents)} statements from Regional Bank Presidents")
print(f"   Statements with unemployment data: {bank_presidents['unemployment_rate'].notna().sum()}")
print(f"   Unique speakers: {bank_presidents['speaker'].nunique()}")
print(f"   Date range: {bank_presidents['date'].min()} to {bank_presidents['date'].max()}")

# ============================================================================
# COMPUTE SEMANTIC SCORES
# ============================================================================

print("\n[2] Computing semantic embeddings...")

# Check if cached embeddings exist
embeddings_cache_file = f'{CACHE_DIR}/semantic_scores_cache.pkl'

if os.path.exists(embeddings_cache_file):
    print("\n   📦 Loading cached semantic scores...")
    with open(embeddings_cache_file, 'rb') as f:
        cached_scores = pickle.load(f)

    bank_presidents['unemployment_discussion_score'] = cached_scores['unemployment_discussion_score']
    bank_presidents['dissent_tone_score'] = cached_scores['dissent_tone_score']
    print(f"✅ Loaded cached scores for {len(bank_presidents)} statements")

else:
    print("\n   🔄 Computing new embeddings (this will be cached for future runs)...")

    # Get embeddings for concept anchors
    print("\n   Encoding unemployment concepts...")
    unemp_embeddings = {}
    for concept, text in UNEMPLOYMENT_CONCEPTS.items():
        unemp_embeddings[concept] = get_embedding(text)

    print("   Encoding dissent concepts...")
    dissent_embeddings = {}
    for concept, text in DISSENT_CONCEPTS.items():
        dissent_embeddings[concept] = get_embedding(text)

    # Compute similarity scores for each speaker turn
    print("\n   Scoring speaker statements...")

    unemp_scores = []
    dissent_scores = []

    for idx, row in tqdm(bank_presidents.iterrows(), total=len(bank_presidents)):
        if pd.isna(row['text']) or len(row['text']) < 50:
            unemp_scores.append(np.nan)
            dissent_scores.append(np.nan)
            continue

        # Get embedding for this statement
        statement_emb = get_embedding(row['text'])

        # Compute average similarity to unemployment concepts
        unemp_sims = []
        for concept_emb in unemp_embeddings.values():
            sim = 1 - cosine(statement_emb, concept_emb)
            unemp_sims.append(sim)
        unemp_scores.append(np.mean(unemp_sims))

        # Compute average similarity to dissent concepts
        dissent_sims = []
        for concept_emb in dissent_embeddings.values():
            sim = 1 - cosine(statement_emb, concept_emb)
            dissent_sims.append(sim)
        dissent_scores.append(np.mean(dissent_sims))

    bank_presidents['unemployment_discussion_score'] = unemp_scores
    bank_presidents['dissent_tone_score'] = dissent_scores

    # Cache the scores for future runs
    print(f"\n   💾 Saving scores to cache...")
    cached_scores = {
        'unemployment_discussion_score': unemp_scores,
        'dissent_tone_score': dissent_scores
    }
    with open(embeddings_cache_file, 'wb') as f:
        pickle.dump(cached_scores, f)

    print(f"✅ Computed and cached semantic scores for all statements")

# ============================================================================
# AGGREGATE TO SPEAKER-MEETING LEVEL
# ============================================================================

print("\n[3] Aggregating to speaker-meeting level...")

speaker_meeting = bank_presidents.groupby(['date', 'speaker', 'district']).agg({
    'unemployment_discussion_score': 'mean',
    'dissent_tone_score': 'mean',
    'is_dissent': 'max',  # Did they formally dissent?
    'unemployment_rate': 'first',
    'text': 'count'  # Number of speaking turns
}).reset_index()

speaker_meeting.rename(columns={'text': 'num_turns'}, inplace=True)

# Remove observations with missing data
analysis_df = speaker_meeting.dropna(subset=['unemployment_discussion_score', 
                                              'dissent_tone_score',
                                              'unemployment_rate'])

print(f"✅ {len(analysis_df)} speaker-meeting observations for analysis")
print(f"   Date range: {analysis_df['date'].min()} to {analysis_df['date'].max()}")
print(f"   Unemployment range: {analysis_df['unemployment_rate'].min():.1f}% to {analysis_df['unemployment_rate'].max():.1f}%")

# ============================================================================
# ANALYSIS 1: DOES REGIONAL UNEMPLOYMENT PREDICT UNEMPLOYMENT DISCUSSION?
# ============================================================================

print("\n" + "="*70)
print("ANALYSIS 1: Regional Unemployment → Unemployment Discussion")
print("="*70)

X1 = analysis_df[['unemployment_rate']]
X1 = sm.add_constant(X1)
y1 = analysis_df['unemployment_discussion_score']

model1 = sm.OLS(y1, X1).fit(cov_type='HC1')

print(f"\nModel: Unemployment_Discussion = β₀ + β₁*Regional_Unemployment")
print(model1.summary())

print(f"\n{'Interpretation:':>20}")
if model1.pvalues['unemployment_rate'] < 0.05:
    direction = "MORE" if model1.params['unemployment_rate'] > 0 else "LESS"
    print(f"  ✅ Bank presidents from high-unemployment regions discuss")
    print(f"     unemployment {direction} (p={model1.pvalues['unemployment_rate']:.4f})")
else:
    print(f"  ❌ No significant relationship (p={model1.pvalues['unemployment_rate']:.4f})")

# ============================================================================
# ANALYSIS 2: DOES REGIONAL UNEMPLOYMENT PREDICT DISSENT?
# ============================================================================

print("\n" + "="*70)
print("ANALYSIS 2: Regional Unemployment → Formal Dissent")
print("="*70)

# Logistic regression for binary dissent outcome
import statsmodels.formula.api as smf

dissent_model = smf.logit('is_dissent ~ unemployment_rate', 
                          data=analysis_df).fit()

print(f"\nModel: P(Dissent) = logit(β₀ + β₁*Regional_Unemployment)")
print(dissent_model.summary())

print(f"\n{'Interpretation:':>20}")
if dissent_model.pvalues['unemployment_rate'] < 0.05:
    direction = "MORE" if dissent_model.params['unemployment_rate'] > 0 else "LESS"
    print(f"  ✅ Bank presidents from high-unemployment regions are {direction}")
    print(f"     likely to dissent (p={dissent_model.pvalues['unemployment_rate']:.4f})")
else:
    print(f"  ❌ No significant relationship (p={dissent_model.pvalues['unemployment_rate']:.4f})")

# ============================================================================
# ANALYSIS 3: DOES UNEMPLOYMENT DISCUSSION PREDICT DISSENT?
# ============================================================================

print("\n" + "="*70)
print("ANALYSIS 3: Unemployment Discussion → Dissent Tone")
print("="*70)

X3 = analysis_df[['unemployment_discussion_score']]
X3 = sm.add_constant(X3)
y3 = analysis_df['dissent_tone_score']

model3 = sm.OLS(y3, X3).fit(cov_type='HC1')

print(f"\nModel: Dissent_Tone = β₀ + β₁*Unemployment_Discussion")
print(model3.summary())

print(f"\n{'Interpretation:':>20}")
if model3.pvalues['unemployment_discussion_score'] < 0.05:
    print(f"  ✅ Speakers who discuss unemployment more have stronger")
    print(f"     dissent tone (p={model3.pvalues['unemployment_discussion_score']:.4f})")
else:
    print(f"  ❌ No significant relationship (p={model3.pvalues['unemployment_discussion_score']:.4f})")

# ============================================================================
# ANALYSIS 4: FULL MODEL WITH INTERACTION
# ============================================================================

print("\n" + "="*70)
print("ANALYSIS 4: Full Model with Interaction")
print("="*70)

# Create interaction term
analysis_df['unemp_x_discussion'] = (analysis_df['unemployment_rate'] * 
                                      analysis_df['unemployment_discussion_score'])

X4 = analysis_df[['unemployment_rate', 'unemployment_discussion_score', 'unemp_x_discussion']]
X4 = sm.add_constant(X4)
y4 = analysis_df['dissent_tone_score']

model4 = sm.OLS(y4, X4).fit(cov_type='HC1')

print(f"\nModel: Dissent_Tone = β₀ + β₁*Regional_Unemp + β₂*Unemp_Discussion + β₃*Interaction")
print(model4.summary())

print(f"\n{'Key Findings:':>20}")
for var in ['unemployment_rate', 'unemployment_discussion_score', 'unemp_x_discussion']:
    if model4.pvalues[var] < 0.05:
        print(f"  ✅ {var}: β={model4.params[var]:.4f} (p={model4.pvalues[var]:.4f})")
    else:
        print(f"  ❌ {var}: β={model4.params[var]:.4f} (p={model4.pvalues[var]:.4f})")

# ============================================================================
# VISUALIZATION
# ============================================================================

print("\n[4] Creating visualizations...")

import matplotlib.pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Regional Unemployment vs Discussion Score
ax1 = axes[0, 0]
ax1.scatter(analysis_df['unemployment_rate'], 
           analysis_df['unemployment_discussion_score'],
           alpha=0.5)
ax1.set_xlabel('Regional Unemployment Rate (%)')
ax1.set_ylabel('Unemployment Discussion Score')
ax1.set_title('Regional Conditions vs Discussion Content')
# Add regression line
z = np.polyfit(analysis_df['unemployment_rate'].dropna(), 
               analysis_df['unemployment_discussion_score'].dropna(), 1)
p = np.poly1d(z)
ax1.plot(sorted(analysis_df['unemployment_rate']), 
         p(sorted(analysis_df['unemployment_rate'])), 
         "r--", alpha=0.8, linewidth=2)

# Plot 2: Discussion Score vs Dissent Tone
ax2 = axes[0, 1]
ax2.scatter(analysis_df['unemployment_discussion_score'],
           analysis_df['dissent_tone_score'],
           alpha=0.5)
ax2.set_xlabel('Unemployment Discussion Score')
ax2.set_ylabel('Dissent Tone Score')
ax2.set_title('Discussion Content vs Dissent Behavior')

# Plot 3: Dissent rates by unemployment quartile
ax3 = axes[1, 0]
analysis_df['unemp_quartile'] = pd.qcut(analysis_df['unemployment_rate'], 4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
dissent_by_quartile = analysis_df.groupby('unemp_quartile')['is_dissent'].mean()
ax3.bar(range(len(dissent_by_quartile)), dissent_by_quartile.values)
ax3.set_xticks(range(len(dissent_by_quartile)))
ax3.set_xticklabels(dissent_by_quartile.index)
ax3.set_xlabel('Regional Unemployment Quartile')
ax3.set_ylabel('Dissent Rate')
ax3.set_title('Dissent Rates by Regional Unemployment')

# Plot 4: Distribution of discussion scores
ax4 = axes[1, 1]
ax4.hist(analysis_df['unemployment_discussion_score'], bins=30, alpha=0.7, label='Unemployment')
ax4.hist(analysis_df['dissent_tone_score'], bins=30, alpha=0.7, label='Dissent')
ax4.set_xlabel('Semantic Similarity Score')
ax4.set_ylabel('Frequency')
ax4.set_title('Distribution of Semantic Scores')
ax4.legend()

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/regional_dissent_analysis.png', dpi=300, bbox_inches='tight')
print(f"💾 Saved plot to regional_dissent_analysis.png")

# ============================================================================
# SAVE RESULTS
# ============================================================================

analysis_df.to_csv(f'{OUTPUT_DIR}/regional_dissent_data.csv', index=False)
print(f"💾 Saved analysis data to regional_dissent_data.csv")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*70)
print("SUMMARY OF FINDINGS")
print("="*70)

print("\n1. Regional Unemployment → Unemployment Discussion:")
print(f"   β = {model1.params['unemployment_rate']:.4f}, p = {model1.pvalues['unemployment_rate']:.4f}")

print("\n2. Regional Unemployment → Formal Dissent:")
print(f"   β = {dissent_model.params['unemployment_rate']:.4f}, p = {dissent_model.pvalues['unemployment_rate']:.4f}")

print("\n3. Unemployment Discussion → Dissent Tone:")
print(f"   β = {model3.params['unemployment_discussion_score']:.4f}, p = {model3.pvalues['unemployment_discussion_score']:.4f}")

print("\n" + "="*70)
print("✅ ANALYSIS COMPLETE!")
print("="*70)

print("\nInterpretation:")
print("This analysis tests whether Regional Bank Presidents from areas with")
print("high unemployment (1) discuss unemployment more in meetings and (2) are")
print("more likely to dissent from the consensus policy decision.")
print("\nSemantic embeddings measure the content of what they say, while formal")
print("dissent measures their voting behavior.")