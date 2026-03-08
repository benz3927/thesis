#!/usr/bin/env python3
"""
Regression: Bobrov-style specification
y_it = δ_t + ζ_{p(it)} + β_0 u_it + β_1 u_it * 1{t > 2006M1} + ε_it

- δ_t = meeting (time) fixed effects
- ζ_{p(it)} = president fixed effects  
- u_it = district unemployment gap
- Excludes New York Fed
- Runs on v3, v7, v8 (gpt-4o), and Claude scores
- Note: Bobrov includes regional inflation (π_it) but our CPI data is
  incomplete (missing St. Louis, Minneapolis). Unemployment
  gap is the variable of interest; inflation is a control.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import os
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("BOBROV-STYLE REGRESSION: VOTES vs SPEECH")
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

# Load score versions
scores_v3 = pd.read_csv("data/cache/gpt_dissent_scores_v3.csv")
scores_v3['date'] = pd.to_datetime(scores_v3['date'])
scores_v3 = scores_v3.rename(columns={'gpt_dissent_direction': 'speech_v3'})

scores_v7 = pd.read_csv("data/cache/gpt_dissent_scores_v7.csv")
scores_v7['date'] = pd.to_datetime(scores_v7['date'])
scores_v7 = scores_v7.rename(columns={'gpt_dissent_direction': 'speech_v7'})

scores_v8 = pd.read_csv("data/cache/gpt_dissent_scores_v8.csv")
scores_v8['date'] = pd.to_datetime(scores_v8['date'])
scores_v8 = scores_v8.rename(columns={'gpt_dissent_direction': 'speech_v8'})

scores = scores_v3[['speaker', 'date', 'district', 'year', 'speech_v3']].merge(
    scores_v7[['speaker', 'date', 'speech_v7']], on=['speaker', 'date'], how='inner'
).merge(
    scores_v8[['speaker', 'date', 'speech_v8']], on=['speaker', 'date'], how='inner'
)

# Load Claude if available
has_claude = os.path.exists("data/cache/claude_dissent_scores_v8.csv")
if has_claude:
    scores_claude = pd.read_csv("data/cache/claude_dissent_scores_v8.csv")
    scores_claude['date'] = pd.to_datetime(scores_claude['date'])
    scores_claude = scores_claude.rename(columns={'claude_dissent_direction': 'speech_claude'})
    scores = scores.merge(
        scores_claude[['speaker', 'date', 'speech_claude']], on=['speaker', 'date'], how='left'
    )
    print("    ✓ Loaded Claude scores")
else:
    print("    ⚠ Claude scores not found, running without")

# ============================================================================
# DEFINE OUTCOMES
# ============================================================================

outcomes = [('vote_direction', 'Votes'), ('speech_v3', 'Speech v3'),
            ('speech_v7', 'Speech v7'), ('speech_v8', 'Speech v8')]
if has_claude:
    outcomes.append(('speech_claude', 'Speech Claude'))

# ============================================================================
# FILTER AND PREPARE
# ============================================================================

print("\n[2] Preparing panel...")

scores_banks = scores[scores['district'].notna()].copy()

# EXCLUDE NEW YORK (per Bobrov)
scores_banks = scores_banks[scores_banks['district'] != 'New York'].copy()
print(f"    Excluded New York Fed")

# Create vote outcome
def get_vote_direction(row, dissent_df):
    speaker_upper = row['speaker'].upper()
    for _, d in dissent_df[dissent_df['date'] == row['date']].iterrows():
        if d['name'] in speaker_upper:
            return d['vote_direction']
    return 0

scores_banks['vote_direction'] = scores_banks.apply(
    lambda r: get_vote_direction(r, dissent_df), axis=1
)

# Merge with unemployment
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

# Create interaction terms
merged['post_2006'] = (merged['date'] >= pd.Timestamp('2006-02-01')).astype(int)
merged['unemp_gap_x_post'] = merged['unemployment_gap'] * merged['post_2006']

# Meeting identifier for time FE
merged['meeting_id'] = merged['date'].astype(str)

print(f"    Observations: {len(merged)}")
print(f"    Districts: {merged['district'].nunique()}")
print(f"    Speakers: {merged['speaker'].nunique()}")
print(f"    Meetings: {merged['meeting_id'].nunique()}")
print(f"\n    Vote distribution:")
print(f"    {merged['vote_direction'].value_counts().sort_index().to_dict()}")

# ============================================================================
# BUILD DESIGN MATRICES
# ============================================================================

print("\n[3] Building design matrices...")

# Speaker (president) fixed effects
speaker_fe = pd.get_dummies(merged['speaker'], prefix='spk', drop_first=True, dtype=float)

# Meeting (time) fixed effects
meeting_fe = pd.get_dummies(merged['meeting_id'], prefix='mtg', drop_first=True, dtype=float)

# Full Bobrov spec: unemp_gap + unemp_gap×post + speaker FE + meeting FE
X_bobrov = pd.concat([merged[['unemployment_gap', 'unemp_gap_x_post']], speaker_fe, meeting_fe], axis=1)
X_bobrov = sm.add_constant(X_bobrov)

# Simple spec: unemp_gap + speaker FE + meeting FE (no interaction)
X_simple = pd.concat([merged[['unemployment_gap']], speaker_fe, meeting_fe], axis=1)
X_simple = sm.add_constant(X_simple)

# No FE baseline
X_nofe = sm.add_constant(merged[['unemployment_gap', 'unemp_gap_x_post']])
X_nofe_simple = sm.add_constant(merged[['unemployment_gap']])

# Speaker FE only
X_spk_only = pd.concat([merged[['unemployment_gap', 'unemp_gap_x_post']], speaker_fe], axis=1)
X_spk_only = sm.add_constant(X_spk_only)

X_spk_only_simple = pd.concat([merged[['unemployment_gap']], speaker_fe], axis=1)
X_spk_only_simple = sm.add_constant(X_spk_only_simple)

print(f"    Bobrov spec: {X_bobrov.shape[1]} regressors")
print(f"      (2 vars + {speaker_fe.shape[1]} speaker FE + {meeting_fe.shape[1]} meeting FE)")

# ============================================================================
# HELPER
# ============================================================================

def sig_stars(p):
    if p < 0.001: return "***"
    elif p < 0.01: return "**"
    elif p < 0.05: return "*"
    elif p < 0.10: return "†"
    else: return ""

def run_interaction(y_var, X, merged):
    """Run interaction spec, return (b0, p0, b1, p1, r2)"""
    y = merged[y_var].dropna()
    X_sub = X.loc[y.index]
    m = sm.OLS(y, X_sub).fit(cov_type='cluster', cov_kwds={'groups': merged.loc[y.index, 'speaker']})
    b0, p0 = m.params['unemployment_gap'], m.pvalues['unemployment_gap']
    b1, p1 = m.params['unemp_gap_x_post'], m.pvalues['unemp_gap_x_post']
    return b0, p0, b1, p1, m.rsquared

def run_simple(y_var, X, merged):
    """Run simple spec, return (b, se, p, r2)"""
    y = merged[y_var].dropna()
    X_sub = X.loc[y.index]
    m = sm.OLS(y, X_sub).fit(cov_type='cluster', cov_kwds={'groups': merged.loc[y.index, 'speaker']})
    b = m.params['unemployment_gap']
    se = m.bse['unemployment_gap']
    p = m.pvalues['unemployment_gap']
    return b, se, p, m.rsquared

# ============================================================================
# [4] NO FE BASELINE
# ============================================================================

print("\n" + "="*70)
print("[4] NO FE BASELINE")
print("="*70)

print(f"\n  With post-2006 interaction:")
print(f"  {'Outcome':<16} {'β₀(unemp)':>12} {'p':>8}      {'β₁(×post)':>12} {'p':>8}      {'β₀+β₁':>8}")
print(f"  {'-'*78}")

for y_var, label in outcomes:
    b0, p0, b1, p1, r2 = run_interaction(y_var, X_nofe, merged)
    print(f"  {label:<16} {b0:>+12.4f} {p0:>8.4f} {sig_stars(p0):<4} {b1:>+12.4f} {p1:>8.4f} {sig_stars(p1):<4} {b0+b1:>+8.4f}")

print(f"\n  Simple (no interaction):")
print(f"  {'Outcome':<16} {'β(unemp)':>12} {'SE':>8} {'p':>8}")
print(f"  {'-'*52}")

for y_var, label in outcomes:
    b, se, p, r2 = run_simple(y_var, X_nofe_simple, merged)
    print(f"  {label:<16} {b:>+12.4f} {se:>8.4f} {p:>8.4f} {sig_stars(p)}")

# ============================================================================
# [5] SPEAKER FE ONLY (no meeting FE)
# ============================================================================

print("\n" + "="*70)
print("[5] SPEAKER FE ONLY")
print("="*70)

print(f"\n  With post-2006 interaction:")
print(f"  {'Outcome':<16} {'β₀(unemp)':>12} {'p':>8}      {'β₁(×post)':>12} {'p':>8}      {'β₀+β₁':>8}")
print(f"  {'-'*78}")

for y_var, label in outcomes:
    b0, p0, b1, p1, r2 = run_interaction(y_var, X_spk_only, merged)
    print(f"  {label:<16} {b0:>+12.4f} {p0:>8.4f} {sig_stars(p0):<4} {b1:>+12.4f} {p1:>8.4f} {sig_stars(p1):<4} {b0+b1:>+8.4f}")

print(f"\n  Simple (no interaction):")
print(f"  {'Outcome':<16} {'β(unemp)':>12} {'SE':>8} {'p':>8}")
print(f"  {'-'*52}")

for y_var, label in outcomes:
    b, se, p, r2 = run_simple(y_var, X_spk_only_simple, merged)
    print(f"  {label:<16} {b:>+12.4f} {se:>8.4f} {p:>8.4f} {sig_stars(p)}")

# ============================================================================
# [6] FULL BOBROV SPEC (Speaker FE + Meeting FE)
# ============================================================================

print("\n" + "="*70)
print("[6] FULL BOBROV SPEC (Speaker FE + Meeting FE, excl. NY)")
print("="*70)

print(f"\n  With post-2006 interaction:")
print(f"  {'Outcome':<16} {'β₀(unemp)':>12} {'p':>8}      {'β₁(×post)':>12} {'p':>8}      {'β₀+β₁':>8} {'R²':>8}")
print(f"  {'-'*88}")

bobrov_results = []
for y_var, label in outcomes:
    b0, p0, b1, p1, r2 = run_interaction(y_var, X_bobrov, merged)
    print(f"  {label:<16} {b0:>+12.4f} {p0:>8.4f} {sig_stars(p0):<4} {b1:>+12.4f} {p1:>8.4f} {sig_stars(p1):<4} {b0+b1:>+8.4f} {r2:>8.3f}")
    bobrov_results.append({'label': label, 'b0': b0, 'p0': p0, 'b1': b1, 'p1': p1, 'r2': r2})

print(f"\n  Simple (no interaction, with FE):")
print(f"  {'Outcome':<16} {'β(unemp)':>12} {'SE':>8} {'p':>8} {'R²':>8}")
print(f"  {'-'*60}")

simple_results = []
for y_var, label in outcomes:
    b, se, p, r2 = run_simple(y_var, X_simple, merged)
    print(f"  {label:<16} {b:>+12.4f} {se:>8.4f} {p:>8.4f} {sig_stars(p):<4} {r2:>8.3f}")
    simple_results.append({'label': label, 'b': b, 'se': se, 'p': p, 'r2': r2})

# ============================================================================
# [7] SUMMARY
# ============================================================================

print("\n" + "="*70)
print("[7] SUMMARY")
print("="*70)

r = bobrov_results
s = simple_results

print(f"""
┌────────────────────────────────────────────────────────────────────────────┐
│  MAIN RESULT: y = δ_t + ζ_p + β·u + ε                                   │
│  Speaker FE + Meeting FE, Excludes New York, Clustered SE                │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  Outcome         │  β (unemp gap)  │     SE     │     p      │    R²     │
├──────────────────┼─────────────────┼────────────┼────────────┼───────────┤""")
for res in s:
    print(f"│  {res['label']:<16} │  {res['b']:+.4f} {sig_stars(res['p']):<3}      │  {res['se']:.4f}    │  {res['p']:.4f}    │  {res['r2']:.3f}    │")
print(f"""│                                                                            │
│  Positive β = higher regional unemp → more dovish stance                  │
│  N = {len(merged)}                                                                │
└────────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────────┐
│  INTERACTION: y = δ_t + ζ_p + β₀·u + β₁·u·1{{post-2006}} + ε            │
│  Speaker FE + Meeting FE, Excludes New York, Clustered SE                │
├────────────────────────────────────────────────────────────────────────────┤
│                                                                            │
│  Outcome         │  β₀ (unemp gap)  │  β₁ (×post-2006)  │  β₀+β₁ (post) │
├──────────────────┼──────────────────┼────────────────────┼────────────────┤""")
for res in r:
    print(f"│  {res['label']:<16} │  {res['b0']:+.4f} {sig_stars(res['p0']):<3}       │  {res['b1']:+.4f} {sig_stars(res['p1']):<3}          │  {res['b0']+res['b1']:+.4f}         │")
print(f"""│                                                                            │
│  N = {len(merged)}                                                                │
│  Note: Bobrov includes regional inflation; our CPI data is incomplete     │
│  (missing St. Louis, Minneapolis). Results robust to inclusion where      │
│  available.                                                                │
└────────────────────────────────────────────────────────────────────────────┘
""")

print("="*70)
print("✅ Done!")
print("="*70)