#!/usr/bin/env python3
"""
Exploratory Discovery: What Economic Concepts Predict NTFS Changes?

Search through ALL possible economic/policy concepts to find what moves markets.
Then test if dissent moderates the relationship.

Author: Ben Zhao
Date: November 2025
"""

import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
from sentence_transformers import SentenceTransformer
import statsmodels.api as sm
import pickle
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================

OUTPUT_DIR = '/Users/CS/Documents/GitHub/fomc-transcript-2026/data/processed/'
CACHE_DIR = '/Users/CS/Documents/GitHub/fomc-transcript-2026/data/cache/'

# ============================================================================
# COMPREHENSIVE LIST OF ECONOMIC CONCEPTS TO TEST
# ============================================================================

CONCEPT_LIBRARY = {
    # Inflation concepts
    'high_inflation': 'Inflation is running above target and remains elevated',
    'low_inflation': 'Inflation is running below target and remains subdued',
    'inflation_expectations': 'Longer-term inflation expectations remain well anchored',
    'inflation_risk': 'Upside risks to inflation have increased',
    'disinflation': 'Inflation pressures are moderating and declining',
    
    # Growth/employment concepts
    'strong_growth': 'Economic activity is expanding at a solid pace',
    'weak_growth': 'Economic activity has slowed and growth is moderate',
    'labor_market_tight': 'Labor market conditions are strong and unemployment is low',
    'labor_market_weak': 'Labor market conditions have softened and unemployment has risen',
    'recession_risk': 'Downside risks to economic activity have increased',
    
    # Forward guidance concepts
    'rates_low_extended': 'The Committee expects to maintain low rates for an extended period',
    'rates_higher_soon': 'Economic conditions may soon warrant higher interest rates',
    'data_dependent': 'The Committee will assess incoming data in determining the pace of adjustments',
    'gradual_tightening': 'The Committee expects gradual increases in the federal funds rate',
    'patience': 'The Committee will be patient in determining future adjustments to policy',
    
    # Financial conditions
    'financial_stress': 'Financial conditions have tightened and market volatility has increased',
    'financial_accommodative': 'Financial conditions remain accommodative',
    'credit_conditions': 'Credit conditions have eased and lending has increased',
    
    # Global/external
    'global_weakness': 'Global economic developments pose risks to the outlook',
    'trade_tensions': 'Trade policy uncertainty is weighing on business investment',
    'oil_prices': 'Energy prices have declined significantly',
    
    # Policy stance
    'accommodative_stance': 'The stance of monetary policy remains accommodative',
    'neutral_stance': 'The federal funds rate is at or near its neutral level',
    'restrictive_stance': 'Monetary policy is restrictive and weighing on demand',
    'balance_sheet': 'The Committee is continuing to reduce its holdings of securities',
    
    # Uncertainty
    'high_uncertainty': 'Uncertainty about the economic outlook has increased',
    'low_uncertainty': 'The economic outlook remains positive',
    
    # QE/unconventional policy
    'asset_purchases': 'The Committee will continue its asset purchase program',
    'taper_hints': 'The Committee will assess the appropriate pace of purchases',
    
    # Sectoral
    'housing_weak': 'Housing sector activity has softened',
    'housing_strong': 'Residential investment has increased',
    'business_investment': 'Business fixed investment has slowed',
    'consumer_spending': 'Consumer spending has been growing solidly',
    
    # Balance sheet/inflation
    'supply_constraints': 'Supply chain bottlenecks continue to constrain production',
    'wage_pressures': 'Wage pressures have increased',
    
    # Communication style
    'cautious_tone': 'The Committee will proceed carefully in adjusting policy',
    'confident_tone': 'The Committee is confident in its assessment',
    'dovish_overall': 'The Committee will support the economy with accommodative policy for as long as needed',
    'hawkish_overall': 'The Committee is prepared to adjust policy to address inflation risks',
}

print(f"📚 Testing {len(CONCEPT_LIBRARY)} economic concepts")

# ============================================================================
# LOAD DATA
# ============================================================================

print("\n[1] Loading data...")

# Load event study results
df_events = pd.read_csv(f'{OUTPUT_DIR}/ntfs_event_study.csv')
df_events['date'] = pd.to_datetime(df_events['date'])

# Load statements
with open(f'{CACHE_DIR}/fomc_statements_2006_2017.pkl', 'rb') as f:
    statements_df = pickle.load(f)

statements_df['date'] = pd.to_datetime(statements_df['date'])

# Merge
df = pd.merge(df_events, statements_df[['date', 'statement']], on='date', how='left')

print(f"✅ Loaded {len(df)} FOMC meetings with statements and NTFS changes")

# ============================================================================
# LOAD EMBEDDING MODEL
# ============================================================================

print("\n[2] Loading embedding model...")
embedding_model = SentenceTransformer('all-mpnet-base-v2')

# ============================================================================
# COMPUTE CONCEPT SCORES FOR EACH STATEMENT
# ============================================================================

print("\n[3] Computing semantic similarity to all concepts...")

# Encode all concept anchors
concept_embeddings = {}
for concept_name, concept_text in tqdm(CONCEPT_LIBRARY.items(), desc="Encoding concepts"):
    concept_embeddings[concept_name] = embedding_model.encode(concept_text, normalize_embeddings=True)

# For each statement, compute similarity to each concept
concept_scores = {concept: [] for concept in CONCEPT_LIBRARY.keys()}

for idx, row in tqdm(df.iterrows(), total=len(df), desc="Scoring statements"):
    if pd.isna(row['statement']):
        for concept in CONCEPT_LIBRARY.keys():
            concept_scores[concept].append(np.nan)
        continue
    
    # Encode statement
    statement_emb = embedding_model.encode(row['statement'], normalize_embeddings=True)
    
    # Compute similarity to each concept
    for concept_name, concept_emb in concept_embeddings.items():
        # Use 1 - cosine distance as similarity score
        similarity = 1 - cosine(statement_emb, concept_emb)
        concept_scores[concept_name].append(similarity)

# Add concept scores to dataframe
for concept_name, scores in concept_scores.items():
    df[f'concept_{concept_name}'] = scores

print(f"✅ Computed {len(CONCEPT_LIBRARY)} concept scores for each statement")

# ============================================================================
# UNIVARIATE REGRESSIONS: WHICH CONCEPTS PREDICT NTFS?
# ============================================================================

print("\n[4] Running univariate regressions...")
print("\nTesting: NTFS_change = β₀ + β₁*concept_similarity + ε")
print("="*70)

results = []

for concept_name in tqdm(CONCEPT_LIBRARY.keys(), desc="Running regressions"):
    try:
        # Skip if missing data
        concept_col = f'concept_{concept_name}'
        analysis_df = df[[concept_col, 'ntfs_change']].dropna()
        
        if len(analysis_df) < 30:
            continue
        
        # Run regression
        X = analysis_df[[concept_col]]
        X = sm.add_constant(X)
        y = analysis_df['ntfs_change']
        
        model = sm.OLS(y, X).fit(cov_type='HC1')
        
        # Store results
        results.append({
            'concept': concept_name,
            'concept_text': CONCEPT_LIBRARY[concept_name],
            'beta': model.params[concept_col],
            'se': model.bse[concept_col],
            't_stat': model.tvalues[concept_col],
            'p_value': model.pvalues[concept_col],
            'r_squared': model.rsquared,
            'n_obs': int(model.nobs)
        })
    except Exception as e:
        print(f"Error with {concept_name}: {e}")
        continue

results_df = pd.DataFrame(results)
results_df = results_df.sort_values('p_value')

# ============================================================================
# DISPLAY TOP RESULTS
# ============================================================================

print("\n" + "="*70)
print("TOP 10 CONCEPTS PREDICTING NTFS CHANGES")
print("="*70)

top_10 = results_df.head(10)

for idx, row in top_10.iterrows():
    print(f"\n{idx+1}. {row['concept'].upper()}")
    print(f"   Text: \"{row['concept_text']}\"")
    print(f"   β = {row['beta']:.4f} (SE = {row['se']:.4f})")
    print(f"   t = {row['t_stat']:.2f}, p = {row['p_value']:.4f}")
    print(f"   R² = {row['r_squared']:.3f}, N = {row['n_obs']}")

# Save results
results_df.to_csv(f'{OUTPUT_DIR}/concept_discovery_results.csv', index=False)
print(f"\n💾 Saved all results to concept_discovery_results.csv")

# ============================================================================
# TEST DISSENT INTERACTION FOR TOP CONCEPTS
# ============================================================================

print("\n" + "="*70)
print("TESTING DISSENT MODERATION FOR TOP 5 CONCEPTS")
print("="*70)

top_5_concepts = results_df.head(5)['concept'].values

for concept_name in top_5_concepts:
    print(f"\n{'='*70}")
    print(f"CONCEPT: {concept_name}")
    print(f"{'='*70}")
    
    try:
        concept_col = f'concept_{concept_name}'
        analysis_df = df[[concept_col, 'ntfs_change', 'regional_dissent_share']].dropna()
        
        # Create interaction
        analysis_df['concept_x_dissent'] = analysis_df[concept_col] * analysis_df['regional_dissent_share']
        
        # Run regression with interaction
        X = analysis_df[[concept_col, 'regional_dissent_share', 'concept_x_dissent']]
        X = sm.add_constant(X)
        y = analysis_df['ntfs_change']
        
        model = sm.OLS(y, X).fit(cov_type='HC1')
        
        print(f"\nModel: NTFS = β₀ + β₁*{concept_name} + β₂*dissent + β₃*({concept_name} × dissent)")
        print(f"\nβ₁ (concept effect): {model.params[concept_col]:.4f} (p={model.pvalues[concept_col]:.4f})")
        print(f"β₂ (dissent effect): {model.params['regional_dissent_share']:.4f} (p={model.pvalues['regional_dissent_share']:.4f})")
        print(f"β₃ (interaction): {model.params['concept_x_dissent']:.4f} (p={model.pvalues['concept_x_dissent']:.4f})")
        print(f"R² = {model.rsquared:.3f}")
        
        if model.pvalues['concept_x_dissent'] < 0.10:
            if model.params['concept_x_dissent'] < 0:
                print("\n✅ SIGNIFICANT FINDING: Regional dissent REDUCES this effect!")
            else:
                print("\n✅ SIGNIFICANT FINDING: Regional dissent AMPLIFIES this effect!")
    
    except Exception as e:
        print(f"Error: {e}")

# ============================================================================
# CREATE VISUALIZATION
# ============================================================================

print("\n[5] Creating visualization...")

import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# Plot 1: Coefficient plot
ax1 = axes[0]
top_15 = results_df.head(15).sort_values('beta', ascending=True)
colors = ['red' if x < 0 else 'green' for x in top_15['beta']]
ax1.barh(range(len(top_15)), top_15['beta'], color=colors, alpha=0.7)
ax1.set_yticks(range(len(top_15)))
ax1.set_yticklabels(top_15['concept'], fontsize=9)
ax1.axvline(0, color='black', linestyle='--', linewidth=0.5)
ax1.set_xlabel('Effect on NTFS Change (basis points per unit similarity)', fontsize=10)
ax1.set_title('Top 15 Concepts Predicting NTFS Changes', fontsize=12, fontweight='bold')
ax1.grid(axis='x', alpha=0.3)

# Plot 2: P-values
ax2 = axes[1]
results_sorted = results_df.sort_values('p_value').head(20)
ax2.scatter(range(len(results_sorted)), results_sorted['p_value'], s=100, alpha=0.6)
ax2.axhline(0.05, color='red', linestyle='--', label='p=0.05', linewidth=2)
ax2.axhline(0.10, color='orange', linestyle='--', label='p=0.10', linewidth=2)
ax2.set_xticks(range(len(results_sorted)))
ax2.set_xticklabels(results_sorted['concept'], rotation=45, ha='right', fontsize=8)
ax2.set_ylabel('P-value', fontsize=10)
ax2.set_title('Statistical Significance of Top 20 Concepts', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/concept_discovery_plot.png', dpi=300, bbox_inches='tight')
print(f"💾 Saved plot to concept_discovery_plot.png")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*70)
print("SUMMARY")
print("="*70)

sig_concepts = results_df[results_df['p_value'] < 0.05]
print(f"\nFound {len(sig_concepts)} concepts significantly predicting NTFS (p < 0.05)")

if len(sig_concepts) > 0:
    print(f"\nMost powerful predictor:")
    top = results_df.iloc[0]
    print(f"  {top['concept']}: β={top['beta']:.3f}, p={top['p_value']:.4f}, R²={top['r_squared']:.3f}")
    
    print(f"\nConcept text:")
    print(f"  \"{top['concept_text']}\"")
else:
    print("\nNo concepts significantly predict NTFS at p < 0.05")
    print("Possible reasons:")
    print("  - NTFS is too noisy in this sample")
    print("  - Forward guidance effects are heterogeneous")
    print("  - Need different event window or dependent variable")

print("\n" + "="*70)
print("✅ EXPLORATORY ANALYSIS COMPLETE!")
print("="*70)

print("\nNext steps:")
print("1. Review top concepts - do they make economic sense?")
print("2. If interactions are significant, write up THAT finding")
print("3. Consider subsample analysis (ZLB vs non-ZLB, crisis vs normal)")
print("4. Try different dependent variables (2-10 spread, stock returns)")