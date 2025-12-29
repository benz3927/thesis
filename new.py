#!/usr/bin/env python3
"""
Forward Guidance, Regional Dissent, and Yield Curve Dynamics

Research Question: Does regional dissent reduce the effectiveness of forward
guidance by introducing uncertainty about the policy path?

Strategy:
1. Decompose yield curve changes into level/slope/curvature factors (PCA)
2. Measure forward guidance strength using OpenAI semantic embeddings
3. Test whether dissent moderates the FG → yield curve relationship
4. Examine regime heterogeneity (ZLB vs normal times)

Comprehensive FG Effectiveness Measurement:
✅ Multiple event windows (1-day, 2-day, 5-day, 0-to-1 day)
✅ Individual maturity analysis (2Y, 5Y, 10Y, 30Y yields separately)
✅ FG concept heterogeneity (which FG language works best)
✅ Event window robustness checks (sensitivity to window choice)
✅ Baseline PCA factor analysis (level/slope/curvature)
✅ Dissent moderation effects (FG × regional_dissent interaction)
✅ Regime-specific analysis (ZLB vs liftoff periods)

Output Files:
- yield_curve_pca_dataset.csv (full dataset with all variables)
- baseline_fg_results.csv (FG effects on PCA factors)
- maturity_specific_results.csv (FG effects by yield maturity)
- fg_concept_effectiveness.csv (which FG language works best)
- dissent_moderation_results.csv (dissent interaction effects)
- regime_specific_results.csv (regime heterogeneity)
- window_robustness_results.csv (robustness across event windows)
- yield_curve_fg_dissent_analysis.png (comprehensive visualization)

Author: Benjamin Zhao
Date: November 2025
Enhanced: November 2025
"""

import pandas as pd
import numpy as np
from scipy.spatial.distance import cosine
from openai import OpenAI
import statsmodels.api as sm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from dotenv import load_dotenv, find_dotenv

# ============================================================================
# CONFIGURATION
# ============================================================================

OUTPUT_DIR = './data/processed/'
CACHE_DIR = './data/cache/'

# Load OpenAI API key
_ = load_dotenv(find_dotenv())
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY not found. Please set it in .env file")

client = OpenAI(api_key=OPENAI_API_KEY)
print(f"✅ OpenAI API initialized")

# ============================================================================
# HELPER FUNCTION: GET OPENAI EMBEDDINGS
# ============================================================================

def get_embedding(text, model="text-embedding-3-large"):
    """Get embedding from OpenAI API (using larger model for better quality)"""
    text = str(text).replace("\n", " ").strip()
    if not text:
        return None
    try:
        response = client.embeddings.create(input=[text], model=model)
        return np.array(response.data[0].embedding)
    except Exception as e:
        print(f"Error getting embedding: {e}")
        return None

# ============================================================================
# THEORETICALLY-GROUNDED FORWARD GUIDANCE CONCEPTS
# Based on interest rate models paper + Fed literature
# ============================================================================

FG_CONCEPTS = {
    # Pure forward guidance (policy path signals)
    'extended_period': 'The Committee expects to maintain the current target range for the federal funds rate for an extended period',
    'considerable_time': 'The Committee anticipates that it will be appropriate to maintain rates for a considerable time',
    'patient': 'The Committee can be patient in beginning to normalize monetary policy',
    'gradual_increases': 'The Committee expects that gradual increases in the federal funds rate will be appropriate',
    
    # Data-dependent language (state-contingent FG)
    'data_dependent': 'The Committee will assess realized and expected economic conditions in determining the timing and pace of adjustments',
    'conditions_warrant': 'Economic conditions may warrant policy firming soon',
    
    # Threshold-based FG (Odyssean commitments)
    'unemployment_threshold': 'The Committee will maintain accommodation until unemployment falls below 6.5 percent',
    'inflation_threshold': 'The Committee will keep rates low until inflation reaches 2 percent',
    
    # Risk balance (uncertainty signals)
    'balanced_risks': 'The Committee judges that the risks to the outlook for economic activity and the labor market are nearly balanced',
    'downside_risks': 'Downside risks to the economic outlook have increased',
}

print("="*80)
print("YIELD CURVE DECOMPOSITION + FORWARD GUIDANCE ANALYSIS")
print("="*80)
print(f"\nTesting {len(FG_CONCEPTS)} forward guidance concepts")

# ============================================================================
# STEP 1: LOAD AND PREPARE DATA
# ============================================================================

print("\n[1] Loading FOMC event study data...")

# Check if required files exist
required_files = {
    'event_study': f'{OUTPUT_DIR}/ntfs_event_study.csv',
    'statements': f'{CACHE_DIR}/fomc_statements_2006_2017.pkl'
}

missing_files = []
for name, filepath in required_files.items():
    if not os.path.exists(filepath):
        missing_files.append(f"  - {name}: {filepath}")

if missing_files:
    print("❌ ERROR: Required data files not found:")
    print("\n".join(missing_files))
    print("\nPlease ensure you have run the data preparation scripts first.")
    raise FileNotFoundError("Missing required data files")

# Load your existing event study results
df_events = pd.read_csv(f'{OUTPUT_DIR}/ntfs_event_study.csv')
df_events['date'] = pd.to_datetime(df_events['date'])

# Load FOMC statements
with open(f'{CACHE_DIR}/fomc_statements_2006_2017.pkl', 'rb') as f:
    statements_df = pickle.load(f)
statements_df['date'] = pd.to_datetime(statements_df['date'])

# Merge
df = pd.merge(df_events, statements_df[['date', 'statement']], on='date', how='left')

print(f"✅ Loaded {len(df)} FOMC meetings")
print(f"   Date range: {df['date'].min()} to {df['date'].max()}")

# ============================================================================
# STEP 2: DOWNLOAD YIELD CURVE DATA FROM FRED
# ============================================================================

print("\n[2] Downloading Treasury yield curve data from FRED...")

try:
    from fredapi import Fred
    fred = Fred(api_key=os.getenv('FRED_API_KEY'))
    
    # Treasury yields at key maturities
    yield_series = {
        '2Y': 'DGS2',
        '5Y': 'DGS5', 
        '10Y': 'DGS10',
        '30Y': 'DGS30'
    }
    
    yields_data = {}
    for maturity, fred_code in yield_series.items():
        print(f"   Downloading {maturity} yield ({fred_code})...")
        yields_data[maturity] = fred.get_series(fred_code, 
                                                 observation_start='2006-01-01',
                                                 observation_end='2017-12-31')
    
    # Create yields dataframe
    yields_df = pd.DataFrame(yields_data)
    yields_df.index.name = 'date'
    yields_df = yields_df.reset_index()
    
    print(f"✅ Downloaded yield curve data: {len(yields_df)} daily observations")

except ImportError:
    print("⚠️  fredapi not installed. Install with: pip install fredapi")
    print("   Then set FRED_API_KEY in your .env file")
    print("   Get free API key at: https://fred.stlouisfed.org/docs/api/api_key.html")
    raise

except Exception as e:
    print(f"❌ Error downloading FRED data: {e}")
    print("   Make sure FRED_API_KEY is set in your .env file")
    raise

# ============================================================================
# STEP 3: COMPUTE YIELD CURVE CHANGES AROUND FOMC EVENTS (MULTIPLE WINDOWS)
# ============================================================================

print("\n[3] Computing yield curve changes around FOMC announcements...")
print("   Testing multiple event windows for robustness...")

# Define multiple event windows
WINDOWS = {
    '1day': (-1, 1),   # Standard: day before to day after
    '2day': (-2, 2),   # Wider: 2 days before to 2 days after
    '5day': (-5, 5),   # Even wider: 1 week window
    '0to1': (0, 1),    # Same day to next day (if announced during day)
}

# For each FOMC date and each window, compute yield changes
event_yields_all = []

for idx, row in df.iterrows():
    event_date = row['date']
    event_data = {'date': event_date}

    for window_name, (days_before, days_after) in WINDOWS.items():
        # Get day before and after (handling weekends)
        day_before = event_date - pd.Timedelta(days=days_before)
        day_after = event_date + pd.Timedelta(days=days_after)

        # Find closest available dates
        before_yields = yields_df[yields_df['date'] <= day_before].tail(1)
        after_yields = yields_df[yields_df['date'] >= day_after].head(1)

        if len(before_yields) > 0 and len(after_yields) > 0:
            for maturity in ['2Y', '5Y', '10Y', '30Y']:
                col_name = f'delta_{maturity}_{window_name}'
                event_data[col_name] = (after_yields[maturity].iloc[0] -
                                       before_yields[maturity].iloc[0])

    # Also add standard window without suffix (for backward compatibility)
    day_before = event_date - pd.Timedelta(days=1)
    day_after = event_date + pd.Timedelta(days=1)
    before_yields = yields_df[yields_df['date'] <= day_before].tail(1)
    after_yields = yields_df[yields_df['date'] >= day_after].head(1)

    if len(before_yields) > 0 and len(after_yields) > 0:
        event_data['delta_2Y'] = after_yields['2Y'].iloc[0] - before_yields['2Y'].iloc[0]
        event_data['delta_5Y'] = after_yields['5Y'].iloc[0] - before_yields['5Y'].iloc[0]
        event_data['delta_10Y'] = after_yields['10Y'].iloc[0] - before_yields['10Y'].iloc[0]
        event_data['delta_30Y'] = after_yields['30Y'].iloc[0] - before_yields['30Y'].iloc[0]

    event_yields_all.append(event_data)

event_yields_df = pd.DataFrame(event_yields_all)

# Merge with main dataframe
df = pd.merge(df, event_yields_df, on='date', how='left')

print(f"✅ Computed yield changes for {len(event_yields_df)} events across {len(WINDOWS)} windows")
print(f"   Windows: {list(WINDOWS.keys())}")

# ============================================================================
# STEP 4: PRINCIPAL COMPONENTS ANALYSIS OF YIELD CURVE
# ============================================================================

print("\n[4] Decomposing yield curve changes via PCA...")

# Prepare data for PCA (drop NaNs)
yield_changes = df[['delta_2Y', 'delta_5Y', 'delta_10Y', 'delta_30Y']].dropna()

# Standardize
scaler = StandardScaler()
yield_changes_scaled = scaler.fit_transform(yield_changes)

# Perform PCA
pca = PCA(n_components=3)
pca_factors = pca.fit_transform(yield_changes_scaled)

# Add to dataframe
df_pca = df.dropna(subset=['delta_2Y', 'delta_5Y', 'delta_10Y', 'delta_30Y']).copy()
df_pca['pc1_level'] = pca_factors[:, 0]
df_pca['pc2_slope'] = pca_factors[:, 1]
df_pca['pc3_curvature'] = pca_factors[:, 2]

# Print factor loadings
print(f"\n✅ PCA Results:")
print(f"   Explained variance: {pca.explained_variance_ratio_}")
print(f"   PC1 (Level):     {pca.explained_variance_ratio_[0]:.1%}")
print(f"   PC2 (Slope):     {pca.explained_variance_ratio_[1]:.1%}")
print(f"   PC3 (Curvature): {pca.explained_variance_ratio_[2]:.1%}")

print(f"\n   Factor Loadings:")
loadings_df = pd.DataFrame(
    pca.components_.T,
    columns=['PC1_Level', 'PC2_Slope', 'PC3_Curvature'],
    index=['2Y', '5Y', '10Y', '30Y']
)
print(loadings_df.round(3))

# ============================================================================
# STEP 5: COMPUTE FORWARD GUIDANCE SCORES USING OPENAI
# ============================================================================

print("\n[5] Computing forward guidance scores with OpenAI embeddings...")

# Encode FG concept anchors
print("   Encoding forward guidance concepts...")
fg_embeddings = {}
for concept_name, concept_text in tqdm(FG_CONCEPTS.items(), desc="   FG concepts"):
    fg_embeddings[concept_name] = get_embedding(concept_text)

# For each statement, compute similarity to each FG concept
fg_scores = {concept: [] for concept in FG_CONCEPTS.keys()}

print("   Scoring FOMC statements...")
for idx, row in tqdm(df_pca.iterrows(), total=len(df_pca), desc="   Statements"):
    if pd.isna(row['statement']) or len(row['statement']) < 50:
        for concept in FG_CONCEPTS.keys():
            fg_scores[concept].append(np.nan)
        continue
    
    # Get statement embedding
    statement_emb = get_embedding(row['statement'])
    
    if statement_emb is None:
        for concept in FG_CONCEPTS.keys():
            fg_scores[concept].append(np.nan)
        continue
    
    # Compute similarity to each FG concept
    for concept_name, concept_emb in fg_embeddings.items():
        similarity = 1 - cosine(statement_emb, concept_emb)
        fg_scores[concept_name].append(similarity)

# Add to dataframe
for concept_name, scores in fg_scores.items():
    df_pca[f'fg_{concept_name}'] = scores

# Create composite FG score (average across all FG concepts)
fg_cols = [f'fg_{c}' for c in FG_CONCEPTS.keys()]
df_pca['fg_composite'] = df_pca[fg_cols].mean(axis=1)

print(f"✅ Computed forward guidance scores")
print(f"   Average FG composite score: {df_pca['fg_composite'].mean():.3f}")
print(f"   Std dev: {df_pca['fg_composite'].std():.3f}")

# ============================================================================
# STEP 6: DEFINE MONETARY POLICY REGIMES
# ============================================================================

print("\n[6] Defining monetary policy regimes...")

df_pca['regime'] = 'normal'
df_pca.loc[df_pca['date'] < '2008-12-16', 'regime'] = 'pre_crisis'
df_pca.loc[(df_pca['date'] >= '2008-12-16') & (df_pca['date'] < '2015-12-16'), 'regime'] = 'ZLB'
df_pca.loc[df_pca['date'] >= '2015-12-16', 'regime'] = 'liftoff'

regime_counts = df_pca['regime'].value_counts()
print(f"\n✅ Regime breakdown:")
for regime, count in regime_counts.items():
    print(f"   {regime:12s}: {count:3d} meetings")

# ============================================================================
# STEP 7: BASELINE REGRESSIONS - DOES FG MOVE YIELD FACTORS?
# ============================================================================

print("\n" + "="*80)
print("BASELINE ANALYSIS: Forward Guidance → Yield Curve Factors")
print("="*80)

# Test each principal component
factors = ['pc1_level', 'pc2_slope', 'pc3_curvature']

baseline_results = []

for factor in factors:
    print(f"\n{'─'*80}")
    print(f"Dependent Variable: {factor.upper()}")
    print(f"{'─'*80}")
    
    # Prepare data
    reg_df = df_pca[['fg_composite', factor]].dropna()
    
    X = reg_df[['fg_composite']]
    X = sm.add_constant(X)
    y = reg_df[factor]
    
    # Run regression
    model = sm.OLS(y, X).fit(cov_type='HC1')
    
    print(f"\nModel: {factor} = β₀ + β₁*FG_composite")
    print(f"\nβ₁ (FG effect): {model.params['fg_composite']:7.4f}")
    print(f"   Std Error:   {model.bse['fg_composite']:7.4f}")
    print(f"   t-statistic: {model.tvalues['fg_composite']:7.2f}")
    print(f"   p-value:     {model.pvalues['fg_composite']:7.4f}")
    print(f"   R²:          {model.rsquared:7.3f}")
    print(f"   N:           {int(model.nobs):7d}")
    
    # Store results
    baseline_results.append({
        'factor': factor,
        'beta': model.params['fg_composite'],
        'se': model.bse['fg_composite'],
        't_stat': model.tvalues['fg_composite'],
        'p_value': model.pvalues['fg_composite'],
        'r_squared': model.rsquared,
        'n_obs': int(model.nobs)
    })
    
    # Interpretation
    if model.pvalues['fg_composite'] < 0.05:
        direction = "increases" if model.params['fg_composite'] > 0 else "decreases"
        print(f"\n✅ SIGNIFICANT: Stronger FG {direction} {factor}")
    else:
        print(f"\n❌ NOT SIGNIFICANT")

baseline_results_df = pd.DataFrame(baseline_results)

# ============================================================================
# STEP 7B: MATURITY-SPECIFIC ANALYSIS - WHICH YIELDS RESPOND MOST?
# ============================================================================

print("\n" + "="*80)
print("MATURITY-SPECIFIC ANALYSIS: FG Impact by Yield Maturity")
print("="*80)
print("\nTesting which maturities are most sensitive to forward guidance")

maturity_results = []

for maturity in ['delta_2Y', 'delta_5Y', 'delta_10Y', 'delta_30Y']:
    print(f"\n{'─'*80}")
    print(f"Dependent Variable: {maturity.upper()}")
    print(f"{'─'*80}")

    # Prepare data
    reg_df = df_pca[['fg_composite', maturity]].dropna()

    if len(reg_df) < 10:
        print(f"⚠️  Insufficient data (N={len(reg_df)})")
        continue

    X = reg_df[['fg_composite']]
    X = sm.add_constant(X)
    y = reg_df[maturity]

    # Run regression
    model = sm.OLS(y, X).fit(cov_type='HC1')

    print(f"\nModel: {maturity} = β₀ + β₁*FG_composite")
    print(f"\nβ₁ (FG effect): {model.params['fg_composite']:7.4f}")
    print(f"   Std Error:   {model.bse['fg_composite']:7.4f}")
    print(f"   t-statistic: {model.tvalues['fg_composite']:7.2f}")
    print(f"   p-value:     {model.pvalues['fg_composite']:7.4f}")
    print(f"   R²:          {model.rsquared:7.3f}")
    print(f"   N:           {int(model.nobs):7d}")

    # Store results
    maturity_results.append({
        'maturity': maturity.replace('delta_', ''),
        'beta': model.params['fg_composite'],
        'se': model.bse['fg_composite'],
        't_stat': model.tvalues['fg_composite'],
        'p_value': model.pvalues['fg_composite'],
        'r_squared': model.rsquared,
        'n_obs': int(model.nobs)
    })

    # Interpretation
    if model.pvalues['fg_composite'] < 0.05:
        direction = "increases" if model.params['fg_composite'] > 0 else "decreases"
        magnitude = abs(model.params['fg_composite'])
        print(f"\n✅ SIGNIFICANT: Stronger FG {direction} {maturity} by {magnitude:.2f} bps")
    else:
        print(f"\n❌ NOT SIGNIFICANT")

maturity_results_df = pd.DataFrame(maturity_results)

print(f"\n{'='*80}")
print("MATURITY COMPARISON:")
print(f"{'='*80}")
print("\nRanking by FG sensitivity (absolute beta):")
maturity_sorted = maturity_results_df.sort_values('beta', key=abs, ascending=False)
for idx, row in maturity_sorted.iterrows():
    sig = "***" if row['p_value'] < 0.01 else "**" if row['p_value'] < 0.05 else "*" if row['p_value'] < 0.10 else ""
    print(f"  {row['maturity']:4s}: β={row['beta']:7.4f} (p={row['p_value']:.4f}) {sig}")

print(f"\n💡 INTERPRETATION:")
print(f"   If FG works primarily through expectations channel:")
print(f"   • Short-term rates (2Y) should respond less (tied to current policy)")
print(f"   • Medium-term rates (5Y, 10Y) should respond most (expectations channel)")
print(f"   • Long-term rates (30Y) may respond less (distant future uncertainty)")

# ============================================================================
# STEP 7C: FG CONCEPT HETEROGENEITY - WHICH FG LANGUAGE WORKS BEST?
# ============================================================================

print("\n" + "="*80)
print("FG CONCEPT HETEROGENEITY: Which Forward Guidance Types Are Most Effective?")
print("="*80)

concept_effectiveness = []

for concept in FG_CONCEPTS.keys():
    fg_col = f'fg_{concept}'

    # Test on PC1 (level) - the main yield factor
    reg_df = df_pca[[fg_col, 'pc1_level']].dropna()

    if len(reg_df) < 10:
        continue

    X = sm.add_constant(reg_df[[fg_col]])
    y = reg_df['pc1_level']

    try:
        model = sm.OLS(y, X).fit(cov_type='HC1')

        concept_effectiveness.append({
            'concept': concept,
            'beta': model.params[fg_col],
            'se': model.bse[fg_col],
            'p_value': model.pvalues[fg_col],
            'r_squared': model.rsquared,
            'n_obs': int(model.nobs)
        })
    except Exception as e:
        print(f"⚠️  Error with {concept}: {e}")
        continue

concept_effectiveness_df = pd.DataFrame(concept_effectiveness).sort_values('p_value')

print(f"\nTesting {len(concept_effectiveness_df)} FG concepts on PC1 (yield level)")
print(f"\n{'─'*80}")
print(f"Most Effective Forward Guidance Types:")
print(f"{'─'*80}")

for idx, row in concept_effectiveness_df.head(10).iterrows():
    sig = "***" if row['p_value'] < 0.01 else "**" if row['p_value'] < 0.05 else "*" if row['p_value'] < 0.10 else ""
    print(f"  {row['concept']:25s}: β={row['beta']:7.4f}, p={row['p_value']:.4f} {sig}, R²={row['r_squared']:.3f}")

print(f"\n💡 KEY INSIGHTS:")
sig_concepts = concept_effectiveness_df[concept_effectiveness_df['p_value'] < 0.10]
if len(sig_concepts) > 0:
    print(f"   • {len(sig_concepts)} out of {len(FG_CONCEPTS)} FG concepts significantly affect yields")
    most_effective = concept_effectiveness_df.iloc[0]
    print(f"   • Most effective: '{most_effective['concept']}' (p={most_effective['p_value']:.4f})")
    print(f"   • This suggests the market responds most to specific FG language patterns")
else:
    print(f"   • No individual FG concepts reach significance")
    print(f"   • Composite score may be capturing general FG presence better")

# ============================================================================
# STEP 8: DISSENT MODERATION - DOES DISSENT WEAKEN FG EFFECTS?
# ============================================================================

print("\n" + "="*80)
print("DISSENT MODERATION ANALYSIS")
print("="*80)
print("\nResearch Question: Does regional dissent reduce FG effectiveness?")
print("Model: yield_factor = β₀ + β₁*FG + β₂*dissent + β₃*(FG × dissent)")

dissent_results = []

for factor in factors:
    print(f"\n{'─'*80}")
    print(f"Dependent Variable: {factor.upper()}")
    print(f"{'─'*80}")
    
    # Prepare data with interaction
    reg_df = df_pca[['fg_composite', 'regional_dissent_share', factor]].dropna()
    
    # Create interaction term
    reg_df['fg_x_dissent'] = reg_df['fg_composite'] * reg_df['regional_dissent_share']
    
    X = reg_df[['fg_composite', 'regional_dissent_share', 'fg_x_dissent']]
    X = sm.add_constant(X)
    y = reg_df[factor]
    
    # Run regression
    model = sm.OLS(y, X).fit(cov_type='HC1')
    
    print(f"\nCoefficients:")
    print(f"  β₁ (FG):          {model.params['fg_composite']:7.4f} (p={model.pvalues['fg_composite']:.4f})")
    print(f"  β₂ (dissent):     {model.params['regional_dissent_share']:7.4f} (p={model.pvalues['regional_dissent_share']:.4f})")
    print(f"  β₃ (interaction): {model.params['fg_x_dissent']:7.4f} (p={model.pvalues['fg_x_dissent']:.4f})")
    print(f"  R²:               {model.rsquared:7.3f}")
    print(f"  N:                {int(model.nobs):7d}")
    
    # Store results
    dissent_results.append({
        'factor': factor,
        'beta_fg': model.params['fg_composite'],
        'beta_dissent': model.params['regional_dissent_share'],
        'beta_interaction': model.params['fg_x_dissent'],
        'p_fg': model.pvalues['fg_composite'],
        'p_dissent': model.pvalues['regional_dissent_share'],
        'p_interaction': model.pvalues['fg_x_dissent'],
        'r_squared': model.rsquared,
        'n_obs': int(model.nobs)
    })
    
    # Interpretation
    if model.pvalues['fg_x_dissent'] < 0.10:
        if model.params['fg_x_dissent'] < 0:
            print(f"\n✅ KEY FINDING: Dissent WEAKENS forward guidance effect on {factor}!")
            print(f"   → Higher dissent reduces how much FG moves this factor")
        else:
            print(f"\n✅ KEY FINDING: Dissent AMPLIFIES forward guidance effect on {factor}!")
            print(f"   → Higher dissent increases how much FG moves this factor")
    else:
        print(f"\n❌ No significant interaction effect")

dissent_results_df = pd.DataFrame(dissent_results)

# ============================================================================
# STEP 9: REGIME-SPECIFIC ANALYSIS - ZLB VS NORMAL TIMES
# ============================================================================

print("\n" + "="*80)
print("REGIME-SPECIFIC ANALYSIS: Does FG matter more at ZLB?")
print("="*80)

regime_results = []

for regime in ['ZLB', 'liftoff']:
    print(f"\n{'='*80}")
    print(f"REGIME: {regime.upper()}")
    print(f"{'='*80}")
    
    regime_df = df_pca[df_pca['regime'] == regime].copy()
    
    for factor in factors:
        print(f"\n{factor.upper()}:")
        
        # Prepare data
        reg_df = regime_df[['fg_composite', 'regional_dissent_share', factor]].dropna()
        
        if len(reg_df) < 10:
            print(f"  ⚠️  Insufficient data (N={len(reg_df)})")
            continue
        
        # Create interaction
        reg_df['fg_x_dissent'] = reg_df['fg_composite'] * reg_df['regional_dissent_share']
        
        X = reg_df[['fg_composite', 'regional_dissent_share', 'fg_x_dissent']]
        X = sm.add_constant(X)
        y = reg_df[factor]
        
        # Run regression
        try:
            model = sm.OLS(y, X).fit(cov_type='HC1')
            
            print(f"  β₁ (FG):          {model.params['fg_composite']:7.4f} (p={model.pvalues['fg_composite']:.4f})")
            print(f"  β₃ (interaction): {model.params['fg_x_dissent']:7.4f} (p={model.pvalues['fg_x_dissent']:.4f})")
            print(f"  R²: {model.rsquared:.3f}, N={int(model.nobs)}")
            
            regime_results.append({
                'regime': regime,
                'factor': factor,
                'beta_fg': model.params['fg_composite'],
                'beta_interaction': model.params['fg_x_dissent'],
                'p_fg': model.pvalues['fg_composite'],
                'p_interaction': model.pvalues['fg_x_dissent'],
                'r_squared': model.rsquared,
                'n_obs': int(model.nobs)
            })
            
        except Exception as e:
            print(f"  ⚠️  Regression failed: {e}")

regime_results_df = pd.DataFrame(regime_results)

# ============================================================================
# STEP 9B: EVENT WINDOW ROBUSTNESS - TEST ALTERNATIVE WINDOWS
# ============================================================================

print("\n" + "="*80)
print("EVENT WINDOW ROBUSTNESS: Testing Alternative Event Windows")
print("="*80)
print("\nDoes FG effectiveness depend on the event window choice?")

window_robustness = []

for window_name in WINDOWS.keys():
    print(f"\n{'─'*80}")
    print(f"Window: {window_name} ({WINDOWS[window_name][0]} to +{WINDOWS[window_name][1]} days)")
    print(f"{'─'*80}")

    # Use 10Y yield as the benchmark (most liquid, mid-maturity)
    delta_col = f'delta_10Y_{window_name}'

    if delta_col not in df_pca.columns:
        print(f"⚠️  Column {delta_col} not found")
        continue

    reg_df = df_pca[['fg_composite', delta_col]].dropna()

    if len(reg_df) < 10:
        print(f"⚠️  Insufficient data (N={len(reg_df)})")
        continue

    X = sm.add_constant(reg_df[['fg_composite']])
    y = reg_df[delta_col]

    try:
        model = sm.OLS(y, X).fit(cov_type='HC1')

        print(f"  β (FG → 10Y yield): {model.params['fg_composite']:7.4f} (p={model.pvalues['fg_composite']:.4f})")
        print(f"  R²: {model.rsquared:.3f}, N={int(model.nobs)}")

        window_robustness.append({
            'window': window_name,
            'days_before': WINDOWS[window_name][0],
            'days_after': WINDOWS[window_name][1],
            'beta': model.params['fg_composite'],
            'se': model.bse['fg_composite'],
            'p_value': model.pvalues['fg_composite'],
            'r_squared': model.rsquared,
            'n_obs': int(model.nobs)
        })

        if model.pvalues['fg_composite'] < 0.05:
            print(f"  ✅ SIGNIFICANT in this window")
        else:
            print(f"  ❌ Not significant")

    except Exception as e:
        print(f"  ⚠️  Regression failed: {e}")

window_robustness_df = pd.DataFrame(window_robustness)

if len(window_robustness_df) > 0:
    print(f"\n{'='*80}")
    print("ROBUSTNESS SUMMARY:")
    print(f"{'='*80}")
    print("\nFG effect on 10Y yields across different event windows:")
    for idx, row in window_robustness_df.iterrows():
        sig = "***" if row['p_value'] < 0.01 else "**" if row['p_value'] < 0.05 else "*" if row['p_value'] < 0.10 else ""
        print(f"  {row['window']:6s}: β={row['beta']:7.4f} (p={row['p_value']:.4f}) {sig}")

    print(f"\n💡 INTERPRETATION:")
    sig_windows = window_robustness_df[window_robustness_df['p_value'] < 0.05]
    if len(sig_windows) >= len(window_robustness_df) * 0.75:
        print(f"   ✅ FG effect is ROBUST across {len(sig_windows)}/{len(window_robustness_df)} windows")
        print(f"   • Results are not sensitive to event window choice")
    elif len(sig_windows) > 0:
        print(f"   ⚠️  FG effect is MIXED: significant in {len(sig_windows)}/{len(window_robustness_df)} windows")
        print(f"   • Results may be sensitive to event window choice")
    else:
        print(f"   ❌ FG effect is NOT robust across alternative windows")

# ============================================================================
# STEP 10: VISUALIZATION
# ============================================================================

print("\n[10] Creating visualizations...")

fig = plt.figure(figsize=(20, 16))
gs = fig.add_gridspec(4, 4, hspace=0.35, wspace=0.35)

# Plot 1: PCA loadings
ax1 = fig.add_subplot(gs[0, 0])
loadings_df.plot(kind='bar', ax=ax1, color=['#2E86AB', '#A23B72', '#F18F01'])
ax1.set_title('PCA Factor Loadings', fontsize=12, fontweight='bold')
ax1.set_xlabel('Maturity')
ax1.set_ylabel('Loading')
ax1.legend(title='Factor', loc='best')
ax1.grid(alpha=0.3)

# Plot 2: Baseline FG effects
ax2 = fig.add_subplot(gs[0, 1])
colors = ['green' if p < 0.05 else 'gray' for p in baseline_results_df['p_value']]
ax2.barh(baseline_results_df['factor'], baseline_results_df['beta'], color=colors, alpha=0.7)
ax2.axvline(0, color='black', linestyle='--', linewidth=0.5)
ax2.set_xlabel('β coefficient')
ax2.set_title('Baseline: FG → Yield Factors', fontsize=12, fontweight='bold')
ax2.grid(axis='x', alpha=0.3)

# Plot 3: Interaction effects
ax3 = fig.add_subplot(gs[0, 2])
colors = ['red' if p < 0.10 else 'gray' for p in dissent_results_df['p_interaction']]
ax3.barh(dissent_results_df['factor'], dissent_results_df['beta_interaction'], 
         color=colors, alpha=0.7)
ax3.axvline(0, color='black', linestyle='--', linewidth=0.5)
ax3.set_xlabel('β₃ (interaction)')
ax3.set_title('Dissent Moderation Effect', fontsize=12, fontweight='bold')
ax3.grid(axis='x', alpha=0.3)

# Plot 4-6: Scatter plots for each factor
for i, factor in enumerate(factors):
    ax = fig.add_subplot(gs[1, i])
    
    plot_df = df_pca[[factor, 'fg_composite']].dropna()
    
    ax.scatter(plot_df['fg_composite'], plot_df[factor], alpha=0.5, s=50)
    
    # Add regression line
    z = np.polyfit(plot_df['fg_composite'], plot_df[factor], 1)
    p = np.poly1d(z)
    x_line = np.linspace(plot_df['fg_composite'].min(), plot_df['fg_composite'].max(), 100)
    ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)
    
    ax.set_xlabel('FG Composite Score')
    ax.set_ylabel(factor)
    ax.set_title(f'FG vs {factor.upper()}', fontsize=11, fontweight='bold')
    ax.grid(alpha=0.3)

# Plot 7-9: Regime comparison
if len(regime_results_df) > 0:
    for i, factor in enumerate(factors):
        ax = fig.add_subplot(gs[2, i])

        factor_data = regime_results_df[regime_results_df['factor'] == factor]

        if len(factor_data) > 0:
            x = np.arange(len(factor_data))
            width = 0.35

            ax.bar(x - width/2, factor_data['beta_fg'], width, label='FG Effect',
                   alpha=0.7, color='#2E86AB')
            ax.bar(x + width/2, factor_data['beta_interaction'], width,
                   label='FG×Dissent', alpha=0.7, color='#A23B72')

            ax.set_xticks(x)
            ax.set_xticklabels(factor_data['regime'])
            ax.set_ylabel('β coefficient')
            ax.set_title(f'{factor.upper()} by Regime', fontsize=11, fontweight='bold')
            ax.legend()
            ax.axhline(0, color='black', linestyle='--', linewidth=0.5)
            ax.grid(axis='y', alpha=0.3)

# Plot 10: Maturity-specific effects
ax10 = fig.add_subplot(gs[2, 3])
if len(maturity_results_df) > 0:
    colors = ['green' if p < 0.05 else 'orange' if p < 0.10 else 'gray'
              for p in maturity_results_df['p_value']]
    ax10.barh(maturity_results_df['maturity'], maturity_results_df['beta'],
              color=colors, alpha=0.7)
    ax10.axvline(0, color='black', linestyle='--', linewidth=0.5)
    ax10.set_xlabel('β coefficient')
    ax10.set_title('FG Impact by Maturity', fontsize=11, fontweight='bold')
    ax10.grid(axis='x', alpha=0.3)

# Plot 11: FG concept effectiveness (top 8)
ax11 = fig.add_subplot(gs[3, 0:2])
if len(concept_effectiveness_df) > 0:
    top_concepts = concept_effectiveness_df.head(8).copy()
    colors = ['green' if p < 0.05 else 'orange' if p < 0.10 else 'gray'
              for p in top_concepts['p_value']]

    # Shorten concept names for display
    top_concepts['concept_short'] = top_concepts['concept'].str.replace('_', ' ').str[:20]

    ax11.barh(top_concepts['concept_short'], top_concepts['beta'],
              color=colors, alpha=0.7)
    ax11.axvline(0, color='black', linestyle='--', linewidth=0.5)
    ax11.set_xlabel('β coefficient (effect on PC1)')
    ax11.set_title('Most Effective FG Concepts', fontsize=11, fontweight='bold')
    ax11.grid(axis='x', alpha=0.3)
    ax11.invert_yaxis()

# Plot 12: Event window robustness
ax12 = fig.add_subplot(gs[3, 2:4])
if len(window_robustness_df) > 0:
    colors = ['green' if p < 0.05 else 'orange' if p < 0.10 else 'gray'
              for p in window_robustness_df['p_value']]

    ax12.bar(window_robustness_df['window'], window_robustness_df['beta'],
             color=colors, alpha=0.7)
    ax12.axhline(0, color='black', linestyle='--', linewidth=0.5)
    ax12.set_ylabel('β coefficient')
    ax12.set_xlabel('Event Window')
    ax12.set_title('Robustness: FG → 10Y Yield Across Windows', fontsize=11, fontweight='bold')
    ax12.grid(axis='y', alpha=0.3)
    ax12.tick_params(axis='x', rotation=45)

plt.savefig(f'{OUTPUT_DIR}/yield_curve_fg_dissent_analysis.png', 
            dpi=300, bbox_inches='tight')
print(f"💾 Saved visualization to yield_curve_fg_dissent_analysis.png")

# ============================================================================
# STEP 11: SAVE RESULTS
# ============================================================================

print("\n[11] Saving results...")

# Save main dataset
df_pca.to_csv(f'{OUTPUT_DIR}/yield_curve_pca_dataset.csv', index=False)
print(f"💾 Saved dataset: yield_curve_pca_dataset.csv")

# Save all regression results
baseline_results_df.to_csv(f'{OUTPUT_DIR}/baseline_fg_results.csv', index=False)
maturity_results_df.to_csv(f'{OUTPUT_DIR}/maturity_specific_results.csv', index=False)
concept_effectiveness_df.to_csv(f'{OUTPUT_DIR}/fg_concept_effectiveness.csv', index=False)
dissent_results_df.to_csv(f'{OUTPUT_DIR}/dissent_moderation_results.csv', index=False)
regime_results_df.to_csv(f'{OUTPUT_DIR}/regime_specific_results.csv', index=False)
window_robustness_df.to_csv(f'{OUTPUT_DIR}/window_robustness_results.csv', index=False)
print(f"💾 Saved all regression results (6 result files)")

# ============================================================================
# STEP 12: EXECUTIVE SUMMARY
# ============================================================================

print("\n" + "="*80)
print("EXECUTIVE SUMMARY")
print("="*80)

print(f"\n📊 SAMPLE:")
print(f"   Total FOMC meetings analyzed: {len(df_pca)}")
print(f"   Date range: {df_pca['date'].min().strftime('%Y-%m-%d')} to {df_pca['date'].max().strftime('%Y-%m-%d')}")
print(f"   ZLB period: {len(df_pca[df_pca['regime']=='ZLB'])} meetings")
print(f"   Liftoff period: {len(df_pca[df_pca['regime']=='liftoff'])} meetings")

print(f"\n📈 PCA DECOMPOSITION:")
print(f"   PC1 (Level) explains {pca.explained_variance_ratio_[0]:.1%} of yield curve variation")
print(f"   PC2 (Slope) explains {pca.explained_variance_ratio_[1]:.1%} of yield curve variation")
print(f"   PC3 (Curvature) explains {pca.explained_variance_ratio_[2]:.1%} of yield curve variation")

print(f"\n🎯 KEY FINDINGS:")

# Finding 1: Baseline FG effects
sig_baseline = baseline_results_df[baseline_results_df['p_value'] < 0.05]
if len(sig_baseline) > 0:
    print(f"\n   1. Forward Guidance Effects:")
    for _, row in sig_baseline.iterrows():
        direction = "increases" if row['beta'] > 0 else "decreases"
        print(f"      • FG {direction} {row['factor']} (β={row['beta']:.3f}, p={row['p_value']:.3f})")
else:
    print(f"\n   1. No significant baseline FG effects detected")

# Finding 2: Dissent moderation
sig_interaction = dissent_results_df[dissent_results_df['p_interaction'] < 0.10]
if len(sig_interaction) > 0:
    print(f"\n   2. ⭐ DISSENT MODERATION EFFECTS:")
    for _, row in sig_interaction.iterrows():
        if row['beta_interaction'] < 0:
            print(f"      • Regional dissent WEAKENS FG effect on {row['factor']}")
            print(f"        (β₃={row['beta_interaction']:.3f}, p={row['p_interaction']:.3f})")
            print(f"        → Policy uncertainty reduces forward guidance credibility!")
        else:
            print(f"      • Regional dissent AMPLIFIES FG effect on {row['factor']}")
            print(f"        (β₃={row['beta_interaction']:.3f}, p={row['p_interaction']:.3f})")
else:
    print(f"\n   2. No significant dissent moderation effects")

# Finding 3: Regime heterogeneity
if len(regime_results_df) > 0:
    zlb_sig = regime_results_df[(regime_results_df['regime']=='ZLB') & 
                                (regime_results_df['p_interaction']<0.10)]
    if len(zlb_sig) > 0:
        print(f"\n   3. ⭐ REGIME EFFECTS (ZLB Period):")
        for _, row in zlb_sig.iterrows():
            print(f"      • During ZLB, dissent moderation stronger for {row['factor']}")
            print(f"        (β₃={row['beta_interaction']:.3f}, p={row['p_interaction']:.3f})")

print(f"\n📝 INTERPRETATION:")
print(f"   This analysis tests whether forward guidance effectiveness depends on")
print(f"   regional dissent. If dissent creates uncertainty about future policy,")
print(f"   it should weaken the transmission of FG to longer-term yields.")
print(f"   ")
print(f"   Expected pattern (if hypothesis holds):")
print(f"   • FG should primarily move PC1 (level) - shifts entire yield curve")
print(f"   • Dissent should reduce this effect (negative β₃ on PC1)")
print(f"   • Effect should be stronger during ZLB (when FG is main tool)")

# Finding 3: Maturity structure
print(f"\n   3. Maturity-Specific Effects:")
if len(maturity_results_df) > 0:
    sig_maturities = maturity_results_df[maturity_results_df['p_value'] < 0.05]
    if len(sig_maturities) > 0:
        print(f"      FG significantly affects {len(sig_maturities)}/4 yield maturities:")
        for _, row in sig_maturities.iterrows():
            print(f"      • {row['maturity']}: β={row['beta']:.3f} (p={row['p_value']:.3f})")

        most_responsive = maturity_results_df.loc[maturity_results_df['beta'].abs().idxmax()]
        print(f"      ⭐ Most responsive: {most_responsive['maturity']} yields")
    else:
        print(f"      No significant effects across maturities")

# Finding 4: FG concept heterogeneity
print(f"\n   4. FG Language Effectiveness:")
if len(concept_effectiveness_df) > 0:
    sig_concepts = concept_effectiveness_df[concept_effectiveness_df['p_value'] < 0.05]
    if len(sig_concepts) > 0:
        print(f"      {len(sig_concepts)}/{len(FG_CONCEPTS)} FG concepts significantly affect yields:")
        for _, row in sig_concepts.head(3).iterrows():
            print(f"      • '{row['concept']}': β={row['beta']:.3f} (p={row['p_value']:.3f})")
        print(f"      → Specific FG language matters!")
    else:
        print(f"      No individual concepts are significant")
        print(f"      → Composite FG measure captures general FG presence better")

# Finding 5: Robustness
print(f"\n   5. ⭐ ROBUSTNESS CHECKS:")
if len(window_robustness_df) > 0:
    sig_windows = window_robustness_df[window_robustness_df['p_value'] < 0.05]
    print(f"      Event window robustness: {len(sig_windows)}/{len(window_robustness_df)} windows significant")
    if len(sig_windows) >= len(window_robustness_df) * 0.75:
        print(f"      ✅ Results are ROBUST to event window choice!")
    elif len(sig_windows) > 0:
        print(f"      ⚠️  Results show MIXED robustness")
        print(f"      → May need to justify baseline window choice")
    else:
        print(f"      ❌ Results NOT robust to window choice")
        print(f"      → Findings may be sensitive to specification")

print(f"\n💡 NEXT STEPS FOR THESIS:")
print(f"   1. If you find significant dissent moderation → THIS IS YOUR RESULT!")
print(f"   2. Write up mechanism: dissent creates policy path uncertainty")
print(f"   3. Show regime heterogeneity supports the mechanism")
print(f"   4. ✅ DONE: Robustness checks across event windows & FG measures")
print(f"   5. Compare to literature: how does this relate to info aggregation?")
print(f"   6. Use maturity-specific results to validate transmission channel")
print(f"   7. Leverage FG concept analysis to refine your FG measure")

print("\n" + "="*80)
print("✅ ANALYSIS COMPLETE!")
print("="*80)

print(f"\n📁 Output files:")
print(f"\n   Dataset:")
print(f"   • {OUTPUT_DIR}/yield_curve_pca_dataset.csv")
print(f"\n   Regression Results:")
print(f"   • {OUTPUT_DIR}/baseline_fg_results.csv              (PCA factors)")
print(f"   • {OUTPUT_DIR}/maturity_specific_results.csv        (individual maturities)")
print(f"   • {OUTPUT_DIR}/fg_concept_effectiveness.csv         (FG language types)")
print(f"   • {OUTPUT_DIR}/dissent_moderation_results.csv       (dissent interactions)")
print(f"   • {OUTPUT_DIR}/regime_specific_results.csv          (ZLB vs liftoff)")
print(f"   • {OUTPUT_DIR}/window_robustness_results.csv        (event window tests)")
print(f"\n   Visualization:")
print(f"   • {OUTPUT_DIR}/yield_curve_fg_dissent_analysis.png  (comprehensive dashboard)")