#!/usr/bin/env python3
"""
Additional Bobrov-style robustness regressions:
  1. Raw unemployment rate (no gap)
  2. Lagged unemployment gap (prior meeting)
  3. District GDP gap (from Bobrov CountyGDP.csv)
  4. Scheduled meetings only
  5. Leave-one-out by district

All use Speaker FE + Meeting FE, clustered SE by speaker, excl. New York.
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import os
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("ADDITIONAL BOBROV-STYLE ROBUSTNESS REGRESSIONS")
print("="*70)

# ============================================================================
# HELPERS
# ============================================================================

def sig_stars(p):
    if p < 0.001: return "***"
    elif p < 0.01: return "**"
    elif p < 0.05: return "*"
    elif p < 0.10: return "†"
    return ""

def run_reg(y_var, x_var, data, label=None):
    """Speaker FE + Meeting FE, clustered SE by speaker."""
    d = data[[y_var, x_var, 'speaker', 'meeting_id']].dropna()
    y = d[y_var]
    spk_fe = pd.get_dummies(d['speaker'],    prefix='spk', drop_first=True, dtype=float)
    mtg_fe = pd.get_dummies(d['meeting_id'], prefix='mtg', drop_first=True, dtype=float)
    X = sm.add_constant(pd.concat([d[[x_var]], spk_fe, mtg_fe], axis=1))
    m = sm.OLS(y, X).fit(cov_type='cluster', cov_kwds={'groups': d['speaker']})
    return {
        'label': label or x_var,
        'b': m.params[x_var], 'se': m.bse[x_var], 'p': m.pvalues[x_var],
        'r2': m.rsquared, 'n': int(y.count())
    }

def print_table(title, rows):
    print(f"\n{'='*75}")
    print(title)
    print(f"{'='*75}")
    hdr = f"{'Outcome':<25} {'β':>9} {'SE':>8} {'p':>8} {'':3} {'R²':>6} {'N':>6}"
    print(hdr)
    print("-"*65)
    for r in rows:
        stars = sig_stars(r['p'])
        print(f"{r['label']:<25} {r['b']:>+9.4f} {r['se']:>8.4f} {r['p']:>8.4f}{stars:<3} {r['r2']:>6.3f} {r['n']:>6}")

# ============================================================================
# LOAD BASE DATA
# ============================================================================

print("\n[1] Loading base data...")

votes = pd.read_excel("data/FOMC_Dissents_Data.xlsx", skiprows=3)
votes["date"] = pd.to_datetime(votes["FOMC Meeting"])

dissent_records = []
for _, row in votes.iterrows():
    for col, direction in [("Dissenters Tighter", -1),
                            ("Dissenters Easier", +1),
                            ("Dissenters Other/Indeterminate", 0)]:
        if pd.notna(row.get(col)):
            for name in str(row[col]).split(", "):
                dissent_records.append({"date": row["date"], "name": name.strip().upper(), "vote_direction": direction})
dissent_df = pd.DataFrame(dissent_records)

scores = pd.read_csv("data/cache/gpt_dissent_scores_v8.csv")
scores['date'] = pd.to_datetime(scores['date'])
scores = scores.rename(columns={'gpt_dissent_direction': 'speech_v8'})

has_claude = os.path.exists("data/cache/claude_dissent_scores_v8.csv")
if has_claude:
    sc = pd.read_csv("data/cache/claude_dissent_scores_v8.csv")
    sc['date'] = pd.to_datetime(sc['date'])
    sc = sc.rename(columns={'claude_dissent_direction': 'speech_claude'})
    scores = scores.merge(sc[['speaker','date','speech_claude']], on=['speaker','date'], how='left')

banks = scores[scores['district'].notna() & (scores['district'] != 'New York')].copy()

def get_vote(row, ddf):
    su = row['speaker'].upper()
    for _, d in ddf[ddf['date'] == row['date']].iterrows():
        if d['name'] in su:
            return d['vote_direction']
    return 0

banks['vote_direction'] = banks.apply(lambda r: get_vote(r, dissent_df), axis=1)
banks['meeting_id'] = banks['date'].astype(str)
banks['year_month'] = banks['date'].dt.to_period('M')

# ============================================================================
# LOAD UNEMPLOYMENT
# ============================================================================

print("[2] Loading unemployment...")

unemp = pd.read_csv("data/cache/regional_unemployment_all.csv")
unemp['date'] = pd.to_datetime(unemp['date'])
unemp['year_month'] = unemp['date'].dt.to_period('M')
nat_u = unemp.groupby('year_month')['unemployment_rate'].mean().rename('nat_unemp')
unemp = unemp.merge(nat_u, on='year_month')
unemp['unemployment_gap'] = unemp['unemployment_rate'] - unemp['nat_unemp']

base = banks.merge(
    unemp[['year_month','district','unemployment_rate','unemployment_gap']],
    on=['year_month','district'], how='inner'
)
print(f"    Base sample N = {len(base)}")

outcomes = ['vote_direction', 'speech_v8'] + (['speech_claude'] if has_claude else [])
labels   = ['Votes', 'Speech (GPT-4o)'] + (['Speech (Claude)'] if has_claude else [])

# ============================================================================
# SPEC 1: RAW UNEMPLOYMENT RATE (NO GAP)
# ============================================================================

print("\n[3] Spec 1: Raw unemployment rate...")

rows = [run_reg(y, 'unemployment_rate', base, f"{lbl}") for y, lbl in zip(outcomes, labels)]
print_table("SPEC 1: Raw Unemployment Rate (No Gap)", rows)
print("  Notes: Uses district unemployment rate directly, not demeaned by national.")

# ============================================================================
# SPEC 2: LAGGED UNEMPLOYMENT GAP
# ============================================================================

print("\n[4] Spec 2: Lagged unemployment gap...")

# Lag = value from the meeting immediately before this one, for same district
base_sorted = base.sort_values(['district','date'])
base_sorted['unemp_gap_lag'] = base_sorted.groupby('district')['unemployment_gap'].shift(1)
base_lag = base_sorted.dropna(subset=['unemp_gap_lag'])

rows = [run_reg(y, 'unemp_gap_lag', base_lag, f"{lbl}") for y, lbl in zip(outcomes, labels)]
print_table("SPEC 2: Lagged Unemployment Gap (Prior Meeting)", rows)
print("  Notes: Unemployment gap from the meeting immediately preceding this one.")

# ============================================================================
# SPEC 3: SCHEDULED MEETINGS ONLY
# ============================================================================

print("\n[5] Spec 3: Scheduled meetings only...")

# Unscheduled meetings tend to be crisis response (e.g. 2001, 2008, 2020)
# Identify by checking for multiple meetings in the same month
meeting_counts = base.groupby(base['date'].dt.to_period('M'))['meeting_id'].nunique()
multi_month = meeting_counts[meeting_counts > 1].index
base_sched = base[~base['date'].dt.to_period('M').isin(multi_month)].copy()

print(f"    Unscheduled meetings dropped: {len(base) - len(base_sched)} obs")
rows = [run_reg(y, 'unemployment_gap', base_sched, f"{lbl}") for y, lbl in zip(outcomes, labels)]
print_table("SPEC 3: Scheduled Meetings Only", rows)
print("  Notes: Drops meetings in months with multiple FOMC meetings (unscheduled/emergency).")

# ============================================================================
# SPEC 4: GDP GAP (from Bobrov CountyGDP.csv)
# ============================================================================

print("\n[6] Spec 4: District GDP gap...")

gdp_path = "data/CountyGDP.csv"
counties_path = "data/FedCounties.csv"

if not os.path.exists(gdp_path):
    print("    WARNING: data/CountyGDP.csv not found — skipping GDP spec.")
    print("    Download from: https://doi.org/10.3886/E210141V1")
else:
    gdp_raw = pd.read_csv(gdp_path, encoding='latin1', dtype=str)
    counties = pd.read_csv(counties_path, encoding='latin1')

    # District number -> name
    district_names = {
        1:'Boston', 2:'New York', 3:'Philadelphia', 4:'Cleveland',
        5:'Richmond', 6:'Atlanta', 7:'Chicago', 8:'St. Louis',
        9:'Minneapolis', 10:'Kansas City', 11:'Dallas', 12:'San Francisco'
    }

    # GeoFIPS is county FIPS (zero-padded 5-digit string like "00000" or "01001")
    # LineCode 1 = All industry total GDP
    gdp = gdp_raw[gdp_raw['LineCode'] == '1'].copy()
    gdp['GeoFIPS'] = gdp['GeoFIPS'].str.strip().str.replace('"', '')

    # Melt wide -> long
    year_cols = [c for c in gdp.columns if c.startswith('v')]
    gdp_long = gdp[['GeoFIPS', 'GeoName'] + year_cols].melt(
        id_vars=['GeoFIPS', 'GeoName'], var_name='year_str', value_name='gdp'
    )
    gdp_long['year'] = gdp_long['year_str'].str.replace('v', '').astype(int)
    gdp_long['gdp'] = pd.to_numeric(gdp_long['gdp'], errors='coerce')
    gdp_long = gdp_long.dropna(subset=['gdp'])

    # Map county FIPS -> district
    counties['GEOID_str'] = counties['GEOID'].astype(str).str.zfill(5)
    fips_district = counties.set_index('GEOID_str')['District'].to_dict()
    gdp_long['district_num'] = gdp_long['GeoFIPS'].map(fips_district)
    gdp_long['district'] = gdp_long['district_num'].map(district_names)
    gdp_long = gdp_long.dropna(subset=['district'])

    # Aggregate to district-year (sum GDP across counties)
    dist_gdp = (
        gdp_long.groupby(['district', 'year'])['gdp']
        .sum().reset_index()
        .rename(columns={'gdp': 'district_gdp'})
    )

    # YoY growth rate
    dist_gdp = dist_gdp.sort_values(['district', 'year'])
    dist_gdp['gdp_growth'] = dist_gdp.groupby('district')['district_gdp'].pct_change() * 100

    # National average growth per year
    nat_gdp = dist_gdp.groupby('year')['gdp_growth'].mean().rename('nat_gdp_growth')
    dist_gdp = dist_gdp.merge(nat_gdp, on='year')
    dist_gdp['gdp_gap'] = dist_gdp['gdp_growth'] - dist_gdp['nat_gdp_growth']

    # Merge with base (on year)
    base['year'] = base['date'].dt.year
    base_gdp = base.merge(dist_gdp[['district','year','gdp_gap']], on=['district','year'], how='inner')
    base_gdp = base_gdp.dropna(subset=['gdp_gap'])

    print(f"    GDP sample N = {len(base_gdp)}")
    rows = [run_reg(y, 'gdp_gap', base_gdp, f"{lbl}") for y, lbl in zip(outcomes, labels)]
    print_table("SPEC 4: District GDP Growth Gap", rows)
    print("  Notes: GDP gap = district real GDP YoY growth minus national average (BEA county GDP).")
    print("  Positive = district growing faster than national -> expected dovish sign.")

# ============================================================================
# SPEC 5: LEAVE-ONE-OUT BY DISTRICT
# ============================================================================

print("\n[7] Spec 5: Leave-one-out by district...")

districts = [d for d in base['district'].unique() if d != 'New York']

print(f"\n  Leave-one-out for Speech (GPT-4o), unemployment gap:")
print(f"  {'Dropped':15} {'β':>9} {'SE':>8} {'p':>8} {'':3} {'N':>6}")
print(f"  {'-'*55}")

for drop in sorted(districts):
    sub = base[base['district'] != drop].copy()
    r = run_reg('speech_v8', 'unemployment_gap', sub)
    stars = sig_stars(r['p'])
    print(f"  Drop {drop:<12} {r['b']:>+9.4f} {r['se']:>8.4f} {r['p']:>8.4f}{stars:<3} {r['n']:>6}")

# Also do votes
print(f"\n  Leave-one-out for Votes, unemployment gap:")
print(f"  {'Dropped':15} {'β':>9} {'SE':>8} {'p':>8} {'':3} {'N':>6}")
print(f"  {'-'*55}")
for drop in sorted(districts):
    sub = base[base['district'] != drop].copy()
    r = run_reg('vote_direction', 'unemployment_gap', sub)
    stars = sig_stars(r['p'])
    print(f"  Drop {drop:<12} {r['b']:>+9.4f} {r['se']:>8.4f} {r['p']:>8.4f}{stars:<3} {r['n']:>6}")

# ============================================================================
# DONE
# ============================================================================

print("\n" + "="*70)
print("DONE. Summary of what to look for:")
print("="*70)
print("""
  Spec 1 (raw rate):    Should be similar to gap spec — confirms result not
                        driven by the demeaning procedure.
  Spec 2 (lagged):      If significant, rules out reverse causality — FOMC 
                        members can't respond to future unemployment.
  Spec 3 (scheduled):   If result holds, not driven by emergency crisis meetings.
  Spec 4 (GDP gap):     Alternative real activity measure — if consistent with
                        unemployment, supports broader interpretation.
  Spec 5 (LOO):         If β stays significant across all drops, no single
                        district is driving the result.
""")