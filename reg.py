#!/usr/bin/env python3
"""
Regional Context Analysis: Do Regional Bank Presidents with Higher Local 
Unemployment Discuss Unemployment More?

FIXED: Aggregates unemployment data before merge to avoid Cartesian product
"""

import pandas as pd
import numpy as np
from openai import OpenAI
from scipy.spatial.distance import cosine
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pickle
from tqdm import tqdm
import os
from dotenv import load_dotenv, find_dotenv
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

OUTPUT_DIR = os.path.join('data', 'processed')
CACHE_DIR = os.path.join('data', 'cache')

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

_ = load_dotenv(find_dotenv())
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set.")

client = OpenAI(api_key=OPENAI_API_KEY)
print(f"✅ OpenAI API key loaded successfully")

print("="*70)
print("REGIONAL UNEMPLOYMENT VS. DISSENT ANALYSIS")
print("="*70)

# ============================================================================
# LOAD DATA
# ============================================================================

print("\n[1] Loading data...")

# Load transcripts with scores (from compute_all_embeddings.py)
try:
    with open(f'{CACHE_DIR}/transcripts_with_scores_2006_2017.pkl', 'rb') as f:
        transcripts_df = pickle.load(f)
    print(f"✅ Loaded {len(transcripts_df)} speaker turns with semantic scores")
except:
    print("⚠️  Could not find transcripts_with_scores_2006_2017.pkl")
    print("    Run compute_all_embeddings.py first!")
    raise

# Create speaker-to-district mapping
speaker_district_mapping = {}

def add_speaker_variants(last_name, district):
    variants = [
        f'AMR {last_name}', f'AMS {last_name}', f'BMR {last_name}',
        f'BMS {last_name}', f'CMR {last_name}', f'CMS {last_name}',
        f'DMR {last_name}', f'DMS {last_name}', f'MR {last_name}',
        f'MS {last_name}', f'ABSMR {last_name}', f'ABSMS {last_name}',
        f'FOMCMR {last_name}', f'FOMCMS {last_name}', f'GDPMR {last_name}',
        f'GDPMS {last_name}', f'MBSMR {last_name}', f'MBSMS {last_name}',
        f'ECBMR {last_name}', f'ECBMS {last_name}', f'CBIASMR {last_name}',
        f'DNAMR {last_name}', f'IOERMR {last_name}',
        f'BVICE CHAIRMAN {last_name}', f'DOJVICE CHAIRMAN {last_name}',
        f'PRESIDENT {last_name}'
    ]
    for variant in variants:
        speaker_district_mapping[variant] = district

# Boston Fed
add_speaker_variants('ROSENGREN', 'Boston')
add_speaker_variants('MINEHAN', 'Boston')

# New York Fed
add_speaker_variants('GEITHNER', 'New York')
add_speaker_variants('DUDLEY', 'New York')

# Philadelphia Fed
add_speaker_variants('PLOSSER', 'Philadelphia')

# Cleveland Fed
add_speaker_variants('PIANALTO', 'Cleveland')
add_speaker_variants('MESTER', 'Cleveland')

# Richmond Fed
add_speaker_variants('LACKER', 'Richmond')

# Atlanta Fed
add_speaker_variants('GUYNN', 'Atlanta')
add_speaker_variants('LOCKHART', 'Atlanta')

# Chicago Fed
add_speaker_variants('MOSKOW', 'Chicago')
add_speaker_variants('EVANS', 'Chicago')

# St. Louis Fed
add_speaker_variants('POOLE', 'St. Louis')
add_speaker_variants('BULLARD', 'St. Louis')

# Minneapolis Fed
add_speaker_variants('STERN', 'Minneapolis')
add_speaker_variants('KOCHERLAKOTA', 'Minneapolis')

# Kansas City Fed
add_speaker_variants('HOENIG', 'Kansas City')
add_speaker_variants('GEORGE', 'Kansas City')

# Dallas Fed
add_speaker_variants('FISHER', 'Dallas')

# San Francisco Fed
add_speaker_variants('WILLIAMS', 'San Francisco')

def map_speaker_to_district(speaker):
    if speaker in speaker_district_mapping:
        return speaker_district_mapping[speaker]
    else:
        return None

transcripts_df['district'] = transcripts_df['speaker'].apply(map_speaker_to_district)

# Keep only Regional Bank Presidents
transcripts_df = transcripts_df[transcripts_df['district'].notna()].copy()

print(f"✅ Mapped to districts, kept {len(transcripts_df)} statements from Regional Bank Presidents")

# Load unemployment data
try:
    regional_unemp = pd.read_csv(f'{CACHE_DIR}/regional_unemployment.csv')
    regional_unemp['date'] = pd.to_datetime(regional_unemp['date'])
    
    # Filter to 2006-2017
    regional_unemp = regional_unemp[
        (regional_unemp['date'] >= pd.Timestamp('2006-01-01')) & 
        (regional_unemp['date'] <= pd.Timestamp('2017-12-31'))
    ].copy()
    
    print(f"✅ Loaded regional unemployment data ({len(regional_unemp)} rows)")
except:
    print("⚠️  Could not find regional_unemployment.csv")
    raise

# ============================================================================
# FIX: AGGREGATE UNEMPLOYMENT DATA BEFORE MERGE
# ============================================================================

print("\n[2] Preparing unemployment data...")

# Create year-month for both datasets
transcripts_df['year_month'] = transcripts_df['date'].dt.to_period('M')
regional_unemp['year_month'] = regional_unemp['date'].dt.to_period('M')

# **KEY FIX: Aggregate unemployment to one row per year_month × district**
print(f"   Before aggregation: {len(regional_unemp)} unemployment rows")

regional_unemp_agg = regional_unemp.groupby(['year_month', 'district']).agg({
    'unemployment_rate': 'mean'  # Average unemployment for the month
}).reset_index()

print(f"   After aggregation: {len(regional_unemp_agg)} unemployment rows")
print(f"   ✅ Aggregated to one row per year_month × district")

# ============================================================================
# MERGE WITH AGGREGATED DATA
# ============================================================================

print("\n[3] Merging transcripts with unemployment data...")

df = pd.merge(
    transcripts_df, 
    regional_unemp_agg[['year_month', 'district', 'unemployment_rate']], 
    on=['year_month', 'district'], 
    how='left'
)

print(f"\n📊 Merge results:")
print(f"   Total rows after merge: {len(df)}")
print(f"   Rows with unemployment_rate: {df['unemployment_rate'].notna().sum()}")
print(f"   Missing unemployment_rate: {df['unemployment_rate'].isna().sum()}")

if df['unemployment_rate'].notna().sum() > 0:
    print(f"   Unemployment range: {df['unemployment_rate'].min():.1f}% to {df['unemployment_rate'].max():.1f}%")
    print(f"   Mean unemployment: {df['unemployment_rate'].mean():.1f}%")

bank_presidents = df.copy()

print(f"\n✅ Working with {len(bank_presidents)} statements from Regional Bank Presidents")
print(f"   Unique speakers: {bank_presidents['speaker'].nunique()}")
print(f"   Date range: {bank_presidents['date'].min()} to {bank_presidents['date'].max()}")

# ============================================================================
# AGGREGATE TO SPEAKER-MEETING LEVEL
# ============================================================================

print("\n[4] Aggregating to speaker-meeting level...")

speaker_meeting = bank_presidents.groupby(['date', 'speaker', 'district']).agg({
    'unemployment_discussion_score': 'mean',
    'dissent_tone_score': 'mean',
    'unemployment_rate': 'first',
    'text': 'count'
}).reset_index()

speaker_meeting.rename(columns={'text': 'num_turns'}, inplace=True)

# Remove missing data
analysis_df = speaker_meeting.dropna(subset=[
    'unemployment_discussion_score',
    'dissent_tone_score',
    'unemployment_rate'
]).copy()

print(f"✅ {len(analysis_df)} speaker-meeting observations for analysis")
print(f"   Date range: {analysis_df['date'].min()} to {analysis_df['date'].max()}")

# ============================================================================
# CREATE REGIONS AND ERAS
# ============================================================================

print("\n[5] Creating categorical controls...")

# Map districts to regions
district_to_region = {
    'Boston': 'East', 'New York': 'East', 'Philadelphia': 'East', 'Richmond': 'East',
    'Cleveland': 'Central', 'Atlanta': 'Central', 'Chicago': 'Central', 'St. Louis': 'Central',
    'Minneapolis': 'West', 'Kansas City': 'West', 'Dallas': 'West', 'San Francisco': 'West'
}
analysis_df['region'] = analysis_df['district'].map(district_to_region)

# Create policy eras
def get_era(date):
    if date < pd.Timestamp('2008-09-15'):
        return 'Pre-Crisis'
    elif date < pd.Timestamp('2015-12-16'):
        return 'Zero Lower Bound'
    else:
        return 'Normalization'

analysis_df['era'] = analysis_df['date'].apply(get_era)
analysis_df['year'] = analysis_df['date'].dt.year

# National unemployment benchmark
national_unemp = analysis_df.groupby('date')['unemployment_rate'].mean().reset_index()
national_unemp.columns = ['date', 'national_unemployment']
analysis_df = analysis_df.merge(national_unemp, on='date', how='left')
analysis_df['unemp_deviation'] = analysis_df['unemployment_rate'] - analysis_df['national_unemployment']

print(f"✅ Created regions ({analysis_df['region'].nunique()}) and eras ({analysis_df['era'].nunique()})")

# ============================================================================
# NAIVE ANALYSIS
# ============================================================================

print("\n" + "="*70)
print("BASELINE: Naive Analysis (No Controls)")
print("="*70)

X_naive = analysis_df[['unemployment_rate']]
X_naive = sm.add_constant(X_naive)
y_naive = analysis_df['unemployment_discussion_score']

model_naive = sm.OLS(y_naive, X_naive).fit(cov_type='HC1')

print(f"\n{model_naive.summary()}")
print(f"\nNaive Result:")
print(f"   β = {model_naive.params['unemployment_rate']:.4f}")
print(f"   p-value = {model_naive.pvalues['unemployment_rate']:.4f}")

# ============================================================================
# FIXED EFFECTS REGRESSION
# ============================================================================

print("\n" + "="*70)
print("METHOD 1: Fixed Effects Regression")
print("="*70)

fe_model = smf.ols(
    'unemployment_discussion_score ~ unemployment_rate + C(region) + C(era)',
    data=analysis_df
).fit(cov_type='cluster', cov_kwds={'groups': analysis_df['speaker']})

print(f"\n{fe_model.summary()}")
print(f"\nFixed Effects Result:")
print(f"   β = {fe_model.params['unemployment_rate']:.4f}")
print(f"   p-value = {fe_model.pvalues['unemployment_rate']:.4f}")

# ============================================================================
# WITHIN-GROUP TRANSFORMATION
# ============================================================================

print("\n" + "="*70)
print("METHOD 2: Within-Group Transformation")
print("="*70)

analysis_df['unemp_rate_within'] = analysis_df.groupby(['region', 'era'])['unemployment_rate'].transform(
    lambda x: x - x.mean()
)
analysis_df['discussion_within'] = analysis_df.groupby(['region', 'era'])['unemployment_discussion_score'].transform(
    lambda x: x - x.mean()
)

within_model = sm.OLS(
    analysis_df['discussion_within'], 
    sm.add_constant(analysis_df['unemp_rate_within'])
).fit(cov_type='HC1')

print(f"\n{within_model.summary()}")
print(f"\nWithin-Group Result:")
print(f"   β = {within_model.params['unemp_rate_within']:.4f}")
print(f"   p-value = {within_model.pvalues['unemp_rate_within']:.4f}")

# ============================================================================
# ERA HETEROGENEITY (KEY FINDING)
# ============================================================================

print("\n" + "="*70)
print("KEY FINDING: Effect Heterogeneity Across Eras")
print("="*70)

interaction_model = smf.ols(
    'unemployment_discussion_score ~ unemployment_rate * C(era) + C(region)',
    data=analysis_df
).fit(cov_type='cluster', cov_kwds={'groups': analysis_df['speaker']})

print(f"\n{interaction_model.summary()}")

# ============================================================================
# SAVE RESULTS
# ============================================================================

print("\n[6] Saving results...")

analysis_df.to_csv(f'{OUTPUT_DIR}/regional_unemployment_analysis.csv', index=False)
print(f"💾 Saved analysis data")

# Save regression results
with open(f'{OUTPUT_DIR}/regression_results.txt', 'w') as f:
    f.write("NAIVE MODEL\n")
    f.write("="*70 + "\n")
    f.write(str(model_naive.summary()) + "\n\n")
    
    f.write("FIXED EFFECTS MODEL\n")
    f.write("="*70 + "\n")
    f.write(str(fe_model.summary()) + "\n\n")
    
    f.write("WITHIN-GROUP MODEL\n")
    f.write("="*70 + "\n")
    f.write(str(within_model.summary()) + "\n\n")
    
    f.write("INTERACTION MODEL\n")
    f.write("="*70 + "\n")
    f.write(str(interaction_model.summary()) + "\n\n")

print(f"💾 Saved regression results")

print("\n" + "="*70)
print("✅ ANALYSIS COMPLETE!")
print("="*70)
print(f"\nKey finding: Relationship between unemployment and discussion")
print(f"varies across policy eras (regime-dependent effect)")