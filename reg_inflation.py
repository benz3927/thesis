#!/usr/bin/env python3
"""
Table A5: Inflation robustness check
Adds inflation_gap as control to main Bobrov spec.
Note: St. Louis and Minneapolis may be missing from regional CPI data.
Sample restricted to districts with CPI coverage.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import os
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("TABLE A5: INFLATION ROBUSTNESS")
print("="*70)

# ============================================================================
# LOAD DATA
# ============================================================================

print("\n[1] Loading data...")

votes = pd.read_excel("data/FOMC_Dissents_Data.xlsx", skiprows=3)
votes["date"] = pd.to_datetime(votes["FOMC Meeting"])

dissent_records = []
for _, row in votes.iterrows():
    for col, direction in [("Dissenters Tighter", -1),
                            ("Dissenters Easier", +1),
                            ("Dissenters Other/Indeterminate", 0)]:
        if pd.notna(row.get(col)):
            for name in str(row[col]).split(", "):
                dissent_records.append({
                    "date": row["date"],
                    "name": name.strip().upper(),
                    "vote_direction": direction
                })
dissent_df = pd.DataFrame(dissent_records)

scores_v8 = pd.read_csv("data/cache/gpt_dissent_scores_v8.csv")
scores_v8['date'] = pd.to_datetime(scores_v8['date'])
scores_v8 = scores_v8.rename(columns={'gpt_dissent_direction': 'speech_v8'})

has_claude = os.path.exists("data/cache/claude_dissent_scores_v8.csv")
if has_claude:
    scores_claude = pd.read_csv("data/cache/claude_dissent_scores_v8.csv")
    scores_claude['date'] = pd.to_datetime(scores_claude['date'])
    scores_claude = scores_claude.rename(columns={'claude_dissent_direction': 'speech_claude'})
    scores_v8 = scores_v8.merge(
        scores_claude[['speaker', 'date', 'speech_claude']], on=['speaker', 'date'], how='left'
    )
    print("    v Claude scores loaded")

scores_banks = scores_v8[
    scores_v8['district'].notna() & (scores_v8['district'] != 'New York')
].copy()

def get_vote_direction(row, dissent_df):
    speaker_upper = row['speaker'].upper()
    for _, d in dissent_df[dissent_df['date'] == row['date']].iterrows():
        if d['name'] in speaker_upper:
            return d['vote_direction']
    return 0

scores_banks['vote_direction'] = scores_banks.apply(
    lambda r: get_vote_direction(r, dissent_df), axis=1
)

# ============================================================================
# MERGE UNEMPLOYMENT
# ============================================================================

print("\n[2] Merging unemployment data...")

unemp = pd.read_csv("data/cache/regional_unemployment_all.csv")
unemp['date'] = pd.to_datetime(unemp['date'])
unemp['year_month'] = unemp['date'].dt.to_period('M')

nat_unemp = unemp.groupby('year_month')['unemployment_rate'].mean().rename('nat_unemp')
unemp = unemp.merge(nat_unemp, on='year_month')
unemp['unemployment_gap'] = unemp['unemployment_rate'] - unemp['nat_unemp']

scores_banks['year_month'] = scores_banks['date'].dt.to_period('M')

merged = scores_banks.merge(
    unemp[['year_month', 'district', 'unemployment_rate', 'unemployment_gap']],
    on=['year_month', 'district'],
    how='inner'
)

# ============================================================================
# MERGE INFLATION
# ============================================================================

print("\n[3] Merging inflation data...")

infl = pd.read_csv("data/cache/regional_inflation.csv")
infl['year_month'] = infl['year_month'].apply(lambda x: pd.Period(x, freq='M'))
infl = infl[['year_month', 'district', 'inflation_gap']].copy()

merged_infl = merged.merge(infl, on=['year_month', 'district'], how='inner')
merged_infl['meeting_id'] = merged_infl['date'].astype(str)

covered  = sorted(merged_infl['district'].unique())
missing  = [d for d in sorted(merged['district'].unique()) if d not in covered]

print(f"    Full sample N:      {len(merged)}")
print(f"    Inflation sample N: {len(merged_infl)}")
print(f"    Districts covered:  {covered}")
if missing:
    print(f"    Districts dropped:  {missing} (no CPI data)")

# ============================================================================
# HELPER
# ============================================================================

def sig_stars(p):
    if p < 0.001: return "***"
    elif p < 0.01:  return "**"
    elif p < 0.05:  return "*"
    elif p < 0.10:  return "†"
    else:           return ""

def run_with_inflation(y_var, data):
    y = data[y_var].dropna()
    d = data.loc[y.index]
    spk_fe = pd.get_dummies(d['speaker'],    prefix='spk', drop_first=True, dtype=float)
    mtg_fe = pd.get_dummies(d['meeting_id'], prefix='mtg', drop_first=True, dtype=float)
    X = sm.add_constant(pd.concat([d[['unemployment_gap', 'inflation_gap']], spk_fe, mtg_fe], axis=1))
    m = sm.OLS(y, X).fit(cov_type='cluster', cov_kwds={'groups': d['speaker']})
    return {
        'b_unemp': m.params['unemployment_gap'], 'se_unemp': m.bse['unemployment_gap'], 'p_unemp': m.pvalues['unemployment_gap'],
        'b_infl':  m.params['inflation_gap'],    'se_infl':  m.bse['inflation_gap'],    'p_infl':  m.pvalues['inflation_gap'],
        'r2': m.rsquared, 'n': int(y.count())
    }

# ============================================================================
# RUN
# ============================================================================

print("\n[4] Running regressions...")

outcomes = [('vote_direction', 'Votes'), ('speech_v8', 'Speech (GPT v8)')]
if has_claude:
    outcomes.append(('speech_claude', 'Speech (Claude)'))

results = {label: run_with_inflation(y_var, merged_infl) for y_var, label in outcomes}

# ============================================================================
# PRINT TABLE
# ============================================================================

print("\n" + "="*80)
print("TABLE A5: ROBUSTNESS — CONTROLLING FOR REGIONAL INFLATION GAP")
print("Speaker FE + Meeting FE, Excludes New York, Clustered SE by Speaker")
if missing:
    print(f"Districts excluded from this table (no CPI): {missing}")
print("="*80)

hdr = f"{'Outcome':<22} {'b_unemp':>9} {'SE':>7} {'p':>7}   {'b_infl':>9} {'SE':>7} {'p':>7}   {'R2':>6} {'N':>5}"
print(f"\n{hdr}")
print("-"*85)

for label, r in results.items():
    print(
        f"{label:<22} "
        f"{r['b_unemp']:>+9.4f} {r['se_unemp']:>7.4f} {r['p_unemp']:>7.4f}{sig_stars(r['p_unemp']):<3} "
        f"{r['b_infl']:>+9.4f} {r['se_infl']:>7.4f} {r['p_infl']:>7.4f}{sig_stars(r['p_infl']):<3} "
        f"{r['r2']:>6.3f} {r['n']:>5}"
    )

print(f"""
Notes:
  Positive b_unemp = higher district unemployment gap -> more dovish stance.
  Positive b_infl  = higher district inflation gap   -> more hawkish stance (expected sign).
  Regional CPI from BLS metro-area series (one metro per district).
  Full sample N = {len(merged)}; inflation-sample N = {len(merged_infl)}.
""")

print("="*70)
print("Done! Copy these numbers into Table A5 in your thesis.")
print("="*70)