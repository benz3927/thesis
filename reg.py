#!/usr/bin/env python3
"""
Regional Unemployment and Dissent Analysis

Research Question: Do Regional Bank Presidents from high-unemployment districts
                   express more dissent in FOMC meetings?

DV: disagreement_score (semantic measure of dissent)
IV: district_unemployment_rate (actual economic data)

Following Bobrov et al. (2024) specification with Speaker + Year FE

Author: Benjamin Zhao
Date: January 2026
"""

import pandas as pd
import numpy as np
from openai import OpenAI
import statsmodels.api as sm
import statsmodels.formula.api as smf
import pickle
import os
from dotenv import load_dotenv, find_dotenv
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'data', 'processed')
CACHE_DIR = os.path.join(SCRIPT_DIR, 'data', 'cache')

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
print("Research Question: Do bank presidents from high-unemployment")
print("                   districts express more dissent?")
print("="*70)

# ============================================================================
# LOAD DATA
# ============================================================================

print("\n[1] Loading data...")

try:
    with open(f'{CACHE_DIR}/transcripts_with_disagreement_scores_2006_2017.pkl', 'rb') as f:
        df = pickle.load(f)
    print(f"✅ Loaded {len(df)} speaker turns with disagreement scores")
    print(f"   Score range: {df['disagreement_score'].min():.4f} to {df['disagreement_score'].max():.4f}")
except:
    print("⚠️  Could not find transcripts_with_disagreement_scores_2006_2017.pkl")
    print("    Run compute_disagreement_embeddings.py first!")
    raise

# ============================================================================
# MAP SPEAKERS TO DISTRICTS
# ============================================================================

print("\n[2] Mapping speakers to districts...")

SPEAKER_TO_DISTRICT = {
    'ROSENGREN': 'Boston', 'MINEHAN': 'Boston',
    'GEITHNER': 'New York', 'DUDLEY': 'New York',
    'PLOSSER': 'Philadelphia', 'SANTOMERO': 'Philadelphia',
    'PIANALTO': 'Cleveland', 'MESTER': 'Cleveland',
    'LACKER': 'Richmond', 'BROADDUS': 'Richmond',
    'GUYNN': 'Atlanta', 'LOCKHART': 'Atlanta',
    'MOSKOW': 'Chicago', 'EVANS': 'Chicago',
    'POOLE': 'St. Louis', 'BULLARD': 'St. Louis',
    'STERN': 'Minneapolis', 'KOCHERLAKOTA': 'Minneapolis', 'KASHKARI': 'Minneapolis',
    'HOENIG': 'Kansas City', 'GEORGE': 'Kansas City',
    'FISHER': 'Dallas',
    'YELLEN': 'San Francisco', 'PARRY': 'San Francisco', 'WILLIAMS': 'San Francisco',
}

def extract_last_name(speaker):
    if pd.isna(speaker):
        return None
    speaker = str(speaker).upper().strip()
    for prefix in ['MR ', 'MS ', 'CHAIRMAN ', 'VICE CHAIRMAN ', 'PRESIDENT ', 'GOVERNOR ', 'RPIX']:
        if speaker.startswith(prefix):
            speaker = speaker[len(prefix):].strip()
    parts = speaker.split()
    return parts[0] if parts else None

df['speaker_clean'] = df['speaker'].apply(extract_last_name)
df['district'] = df['speaker_clean'].map(SPEAKER_TO_DISTRICT)

bank_presidents = df[df['district'].notna()].copy()

print(f"✅ Mapped to districts, kept {len(bank_presidents)} statements from Regional Bank Presidents")
print(f"   Unique speakers: {bank_presidents['speaker_clean'].nunique()}")
print(f"   Unique districts: {bank_presidents['district'].nunique()}")

# ============================================================================
# LOAD UNEMPLOYMENT DATA
# ============================================================================

print("\n[3] Loading unemployment data...")

try:
    regional_unemp = pd.read_csv(f'{CACHE_DIR}/regional_unemployment.csv')
    regional_unemp['date'] = pd.to_datetime(regional_unemp['date'])
    print(f"✅ Loaded regional unemployment data ({len(regional_unemp)} rows)")
except:
    print("⚠️  Could not find regional_unemployment.csv")
    raise

# ============================================================================
# MERGE DATA
# ============================================================================

print("\n[4] Merging transcripts with unemployment data...")

bank_presidents['year_month'] = bank_presidents['date'].dt.to_period('M')
regional_unemp['year_month'] = regional_unemp['date'].dt.to_period('M')

merged = pd.merge(
    bank_presidents,
    regional_unemp[['year_month', 'district', 'unemployment_rate']],
    on=['year_month', 'district'],
    how='left'
)

print(f"✅ Merge complete: {merged['unemployment_rate'].notna().sum()} rows with unemployment data")

# ============================================================================
# AGGREGATE TO SPEAKER-MEETING LEVEL
# ============================================================================

print("\n[5] Aggregating to speaker-meeting level...")

speaker_meeting = merged.groupby(['date', 'speaker_clean', 'district']).agg({
    'disagreement_score': 'mean',
    'unemployment_rate': 'first',
    'text': 'count'
}).reset_index()

speaker_meeting.rename(columns={'text': 'num_turns', 'speaker_clean': 'speaker'}, inplace=True)

# Add year variable
speaker_meeting['year'] = speaker_meeting['date'].dt.year

# Remove missing data
analysis_df = speaker_meeting.dropna(subset=['disagreement_score', 
                                              'unemployment_rate']).copy()

print(f"✅ {len(analysis_df)} speaker-meeting observations for analysis")
print(f"   Years: {analysis_df['year'].min()} to {analysis_df['year'].max()}")
print(f"   Unique speakers: {analysis_df['speaker'].nunique()}")
print(f"   Unique districts: {analysis_df['district'].nunique()}")

print(f"\n📊 Dependent variable (disagreement_score):")
print(f"   Mean: {analysis_df['disagreement_score'].mean():.4f}")
print(f"   Std: {analysis_df['disagreement_score'].std():.4f}")
print(f"   Range: {analysis_df['disagreement_score'].min():.4f} to {analysis_df['disagreement_score'].max():.4f}")

print(f"\n📊 Independent variable (unemployment_rate):")
print(f"   Mean: {analysis_df['unemployment_rate'].mean():.2f}%")
print(f"   Std: {analysis_df['unemployment_rate'].std():.2f}%")
print(f"   Range: {analysis_df['unemployment_rate'].min():.2f}% to {analysis_df['unemployment_rate'].max():.2f}%")

# ============================================================================
# REGRESSION ANALYSIS
# ============================================================================

print("\n" + "="*70)
print("REGRESSION ANALYSIS")
print("DV: disagreement_score | IV: district_unemployment_rate")
print("="*70)

# ============================================================================
# MODEL 1: NAIVE (NO CONTROLS)
# ============================================================================

print("\n[MODEL 1] Naive regression (no controls)")
print("="*70)

X_naive = sm.add_constant(analysis_df[['unemployment_rate']])
y_naive = analysis_df['disagreement_score']
model_naive = sm.OLS(y_naive, X_naive).fit(cov_type='HC1')

print(f"\n{model_naive.summary()}")
print(f"\nNaive Result:")
print(f"   β = {model_naive.params['unemployment_rate']:.6f}")
print(f"   p-value = {model_naive.pvalues['unemployment_rate']:.4f}")
print(f"   R² = {model_naive.rsquared:.3f}")

# ============================================================================
# MODEL 2: SPEAKER FIXED EFFECTS ONLY
# ============================================================================

print("\n" + "="*70)
print("[MODEL 2] Speaker Fixed Effects Only")
print("="*70)

speaker_fe_only = smf.ols(
    'disagreement_score ~ unemployment_rate + C(speaker)',
    data=analysis_df
).fit(cov_type='cluster', cov_kwds={'groups': analysis_df['speaker']})

print(f"\n{speaker_fe_only.summary()}")

print(f"\nSpeaker FE Only Result:")
print(f"   β = {speaker_fe_only.params['unemployment_rate']:.6f}")
print(f"   p-value = {speaker_fe_only.pvalues['unemployment_rate']:.4f}")
print(f"   R² = {speaker_fe_only.rsquared:.3f}")

# ============================================================================
# MODEL 3: SPEAKER + YEAR FIXED EFFECTS ⭐ PREFERRED
# ============================================================================

print("\n" + "="*70)
print("[MODEL 3] Speaker + Year Fixed Effects ⭐ PREFERRED")
print("Following Bobrov et al. (2024) - adapted for feasibility")
print("="*70)

print("\nThis specification asks:")
print("  'Do bank presidents express MORE dissent when their district")
print("   unemployment is HIGH, controlling for individual tendencies")
print("   and common time shocks?'")

speaker_year_fe = smf.ols(
    'disagreement_score ~ unemployment_rate + C(speaker) + C(year)',
    data=analysis_df
).fit(cov_type='cluster', cov_kwds={'groups': analysis_df['speaker']})

print(f"\n{speaker_year_fe.summary()}")

print(f"\nSpeaker + Year FE Result:")
print(f"   β = {speaker_year_fe.params['unemployment_rate']:.6f}")
print(f"   p-value = {speaker_year_fe.pvalues['unemployment_rate']:.4f}")
print(f"   R² = {speaker_year_fe.rsquared:.3f}")

if speaker_year_fe.pvalues['unemployment_rate'] < 0.05:
    print(f"\n   ✅ SIGNIFICANT at 5% level!")
    if speaker_year_fe.params['unemployment_rate'] > 0:
        print(f"   ✅ POSITIVE coefficient: Bank presidents express MORE dissent")
        print(f"      when their district unemployment is higher.")
    else:
        print(f"   ⚠️  NEGATIVE coefficient: Bank presidents express LESS dissent")
        print(f"      when their district unemployment is higher (counterintuitive)")
elif speaker_year_fe.pvalues['unemployment_rate'] < 0.10:
    print(f"\n   ⚡ MARGINALLY SIGNIFICANT at 10% level")
    print(f"      Suggestive evidence of effect")
else:
    print(f"\n   ❌ Not significant at conventional levels")

# ============================================================================
# COMPARISON TABLE
# ============================================================================

print("\n" + "="*70)
print("COMPARISON: All Specifications")
print("="*70)

comparison = pd.DataFrame({
    'Model': [
        'Naive (no controls)',
        'Speaker FE only',
        'Speaker + Year FE ⭐',
    ],
    'Coefficient': [
        model_naive.params['unemployment_rate'],
        speaker_fe_only.params['unemployment_rate'],
        speaker_year_fe.params['unemployment_rate'],
    ],
    'Std Error': [
        model_naive.bse['unemployment_rate'],
        speaker_fe_only.bse['unemployment_rate'],
        speaker_year_fe.bse['unemployment_rate'],
    ],
    'P-value': [
        model_naive.pvalues['unemployment_rate'],
        speaker_fe_only.pvalues['unemployment_rate'],
        speaker_year_fe.pvalues['unemployment_rate'],
    ],
    'R-squared': [
        model_naive.rsquared,
        speaker_fe_only.rsquared,
        speaker_year_fe.rsquared,
    ]
})

def add_stars(p):
    if p < 0.01:
        return '***'
    elif p < 0.05:
        return '**'
    elif p < 0.10:
        return '*'
    else:
        return ''

comparison['Sig'] = comparison['P-value'].apply(add_stars)
print(f"\n{comparison.to_string(index=False)}")
print("\nSignificance: *** p<0.01, ** p<0.05, * p<0.10")

# ============================================================================
# INTERPRETATION
# ============================================================================

print("\n" + "="*70)
print("INTERPRETATION")
print("="*70)

print("\n📊 MAIN RESULT:")
print(f"   Coefficient: {speaker_year_fe.params['unemployment_rate']:.6f}")
print(f"   P-value: {speaker_year_fe.pvalues['unemployment_rate']:.4f}")

if speaker_year_fe.pvalues['unemployment_rate'] < 0.10:
    print("\n✅ EVIDENCE FOUND:")
    print("   Within the same speaker, controlling for year effects,")
    print("   a 1 percentage point increase in district unemployment")
    print(f"   is associated with a {abs(speaker_year_fe.params['unemployment_rate']):.6f} change")
    print("   in disagreement/dissent score.")
    
    # Calculate economic significance
    score_std = analysis_df['disagreement_score'].std()
    effect_size = speaker_year_fe.params['unemployment_rate'] / score_std
    print(f"\n   Economic significance:")
    print(f"   This is {abs(effect_size):.3f} standard deviations of disagreement score")
    
    # Example calculation
    unemp_iqr = analysis_df['unemployment_rate'].quantile(0.75) - analysis_df['unemployment_rate'].quantile(0.25)
    score_change = speaker_year_fe.params['unemployment_rate'] * unemp_iqr
    print(f"\n   Example: Moving from 25th to 75th percentile of unemployment")
    print(f"   (a {unemp_iqr:.1f} percentage point increase)")
    print(f"   changes disagreement score by {abs(score_change):.6f}")
    print(f"   ({abs(score_change/score_std):.2f} standard deviations)")
else:
    print("\n⚠️  NO SIGNIFICANT EVIDENCE:")
    print("   Cannot reject null hypothesis that district unemployment")
    print("   does not affect dissent expression.")

# ============================================================================
# ROBUSTNESS: Effect by era
# ============================================================================

print("\n" + "="*70)
print("ROBUSTNESS: Effect by Era")
print("="*70)

def assign_era(year):
    if year <= 2007:
        return 'Pre-Crisis'
    elif year <= 2013:
        return 'Crisis/ZLB'
    else:
        return 'Recovery'

analysis_df['era'] = analysis_df['year'].apply(assign_era)

print("\nSplit sample by era:")
for era in ['Pre-Crisis', 'Crisis/ZLB', 'Recovery']:
    era_df = analysis_df[analysis_df['era'] == era]
    if len(era_df) > 50:
        try:
            era_model = smf.ols(
                'disagreement_score ~ unemployment_rate + C(speaker)',
                data=era_df
            ).fit(cov_type='cluster', cov_kwds={'groups': era_df['speaker']})
            
            sig = add_stars(era_model.pvalues['unemployment_rate'])
            print(f"\n{era} ({era_df['year'].min()}-{era_df['year'].max()}):")
            print(f"   N = {len(era_df)}")
            print(f"   β = {era_model.params['unemployment_rate']:.6f}")
            print(f"   p = {era_model.pvalues['unemployment_rate']:.4f} {sig}")
        except:
            print(f"\n{era}: Insufficient variation for estimation")

# ============================================================================
# SAVE RESULTS
# ============================================================================

print("\n[6] Saving results...")

analysis_df.to_csv(f'{OUTPUT_DIR}/dissent_analysis_data.csv', index=False)
print(f"💾 Saved: {OUTPUT_DIR}/dissent_analysis_data.csv")

with open(f'{OUTPUT_DIR}/dissent_regression_results.txt', 'w') as f:
    f.write("="*70 + "\n")
    f.write("REGIONAL UNEMPLOYMENT AND DISSENT ANALYSIS\n")
    f.write("DV: disagreement_score | IV: district_unemployment_rate\n")
    f.write("="*70 + "\n\n")
    
    f.write("MODEL 1: NAIVE\n")
    f.write(str(model_naive.summary()))
    f.write("\n\n")
    
    f.write("MODEL 2: SPEAKER FE ONLY\n")
    f.write(str(speaker_fe_only.summary()))
    f.write("\n\n")
    
    f.write("MODEL 3: SPEAKER + YEAR FE (PREFERRED)\n")
    f.write(str(speaker_year_fe.summary()))
    f.write("\n\n")
    
    f.write("COMPARISON TABLE:\n")
    f.write(comparison.to_string(index=False))

print(f"💾 Saved: {OUTPUT_DIR}/dissent_regression_results.txt")

comparison.to_csv(f'{OUTPUT_DIR}/dissent_comparison_table.csv', index=False)
print(f"💾 Saved: {OUTPUT_DIR}/dissent_comparison_table.csv")

print("\n" + "="*70)
print("✅ ANALYSIS COMPLETE!")
print("="*70)

print(f"\n📁 All results saved to: {OUTPUT_DIR}/")
print(f"\n🎓 Research Question: Do bank presidents from high-unemployment")
print(f"   districts express more dissent in FOMC meetings?")
print(f"\n   Main Result: β = {speaker_year_fe.params['unemployment_rate']:.6f}, p = {speaker_year_fe.pvalues['unemployment_rate']:.4f}")