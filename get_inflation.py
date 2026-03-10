#!/usr/bin/env python3
"""
Build district-level inflation from Bobrov et al. replication data.
Uses StateCPI.csv (Hazell et al. 2022 nontradable CPI) and FedCounties.csv
exactly as Bobrov et al. do — aggregates state -> district, computes inflation gap.
"""

import pandas as pd
import numpy as np
import os

CACHE_DIR = 'data/cache'
os.makedirs(CACHE_DIR, exist_ok=True)

print("="*70)
print("BUILDING DISTRICT INFLATION FROM BOBROV REPLICATION DATA")
print("="*70)

# ============================================================================
# FIPS -> STATE NAME MAPPING
# ============================================================================

fips_to_state = {
    1: 'Alabama', 2: 'Alaska', 4: 'Arizona', 5: 'Arkansas', 6: 'California',
    8: 'Colorado', 9: 'Connecticut', 10: 'Delaware', 11: 'District of Columbia',
    12: 'Florida', 13: 'Georgia', 15: 'Hawaii', 16: 'Idaho', 17: 'Illinois',
    18: 'Indiana', 19: 'Iowa', 20: 'Kansas', 21: 'Kentucky', 22: 'Louisiana',
    23: 'Maine', 24: 'Maryland', 25: 'Massachusetts', 26: 'Michigan',
    27: 'Minnesota', 28: 'Mississippi', 29: 'Missouri', 30: 'Montana',
    31: 'Nebraska', 32: 'Nevada', 33: 'New Hampshire', 34: 'New Jersey',
    35: 'New Mexico', 36: 'New York', 37: 'North Carolina', 38: 'North Dakota',
    39: 'Ohio', 40: 'Oklahoma', 41: 'Oregon', 42: 'Pennsylvania',
    44: 'Rhode Island', 45: 'South Carolina', 46: 'South Dakota',
    47: 'Tennessee', 48: 'Texas', 49: 'Utah', 50: 'Vermont',
    51: 'Virginia', 53: 'Washington', 54: 'West Virginia',
    55: 'Wisconsin', 56: 'Wyoming'
}

# ============================================================================
# LOAD DATA
# ============================================================================

print("\n[1] Loading StateCPI.csv...")
cpi = pd.read_csv("data/StateCPI.csv")
print(f"    Columns: {list(cpi.columns)}")
print(f"    Shape: {cpi.shape}")
print(f"    States: {cpi['state'].nunique()}")
print(f"    Years: {cpi['year'].min()} - {cpi['year'].max()}")

print("\n[2] Loading FedCounties.csv...")
counties = pd.read_csv("data/FedCounties.csv", encoding='latin1')
print(f"    Columns: {list(counties.columns)}")
print(f"    Shape: {counties.shape}")

# ============================================================================
# BUILD STATE -> DISTRICT MAPPING
# ============================================================================

print("\n[3] Building state -> district mapping...")

# Add state name to counties using FIPS
counties['state_name'] = counties['STATEFP'].map(fips_to_state)

# For states split across districts, a state may map to multiple districts
# Use the most common district per state (simple majority)
# This matches Bobrov's approach for state-level inflation data
district_names = {
    1: 'Boston', 2: 'New York', 3: 'Philadelphia', 4: 'Cleveland',
    5: 'Richmond', 6: 'Atlanta', 7: 'Chicago', 8: 'St. Louis',
    9: 'Minneapolis', 10: 'Kansas City', 11: 'Dallas', 12: 'San Francisco'
}

state_district = (
    counties.groupby('state_name')['District']
    .agg(lambda x: x.value_counts().index[0])
    .reset_index()
    .rename(columns={'District': 'district'})
)
state_district['district'] = state_district['district'].map(district_names)

print(f"    State-district pairs: {len(state_district)}")
print(f"    Districts covered: {sorted(state_district['district'].unique())}")

# ============================================================================
# MERGE CPI WITH DISTRICT
# ============================================================================

print("\n[4] Merging CPI with district mapping...")

# Standardize state name format
cpi['state'] = cpi['state'].str.strip()
state_district['state_name'] = state_district['state_name'].str.strip()

cpi_merged = cpi.merge(
    state_district,
    left_on='state',
    right_on='state_name',
    how='left'
)

missing = cpi_merged[cpi_merged['district'].isna()]['state'].unique()
if len(missing) > 0:
    print(f"    Warning: No district for states: {missing}")

cpi_merged = cpi_merged.dropna(subset=['district'])
print(f"    Rows after merge: {len(cpi_merged)}")

# ============================================================================
# AGGREGATE TO DISTRICT LEVEL (simple average across states)
# ============================================================================

print("\n[5] Aggregating to district-quarter level...")

# Use pi_nt (nontradable) to match Bobrov exactly
district_cpi = (
    cpi_merged
    .groupby(['district', 'year', 'quarter'])
    .agg(
        pi_nt=('pi_nt', 'mean'),   # nontradable — matches Bobrov
        pi_t=('pi_t', 'mean'),     # tradable
        pi=('pi', 'mean')          # overall
    )
    .reset_index()
)

print(f"    District-quarter obs: {len(district_cpi)}")
print(f"    Districts: {sorted(district_cpi['district'].unique())}")

# ============================================================================
# BUILD DATE AND COMPUTE INFLATION GAP
# ============================================================================

print("\n[6] Building date and computing inflation gap...")

# Convert year+quarter to date (first month of quarter)
quarter_to_month = {1: 1, 2: 4, 3: 7, 4: 10}
district_cpi['month'] = district_cpi['quarter'].map(quarter_to_month)
district_cpi['date'] = pd.to_datetime(
    dict(year=district_cpi['year'],
         month=district_cpi['month'],
         day=1)
)

# National average per quarter
nat_avg = (
    district_cpi
    .groupby('date')[['pi_nt', 'pi_t', 'pi']]
    .mean()
    .rename(columns={'pi_nt': 'nat_pi_nt', 'pi_t': 'nat_pi_t', 'pi': 'nat_pi'})
    .reset_index()
)

district_cpi = district_cpi.merge(nat_avg, on='date', how='left')

# Gaps (district minus national) — matches Bobrov spec
district_cpi['inflation_gap']    = district_cpi['pi_nt'] - district_cpi['nat_pi_nt']  # main
district_cpi['inflation_gap_t']  = district_cpi['pi_t']  - district_cpi['nat_pi_t']
district_cpi['inflation_gap_all']= district_cpi['pi']    - district_cpi['nat_pi']

print(f"    Date range: {district_cpi['date'].min()} to {district_cpi['date'].max()}")
print(f"    Mean inflation gap: {district_cpi['inflation_gap'].mean():.4f}")

# ============================================================================
# ADD YEAR_MONTH FOR MERGING WITH REG DATA
# ============================================================================

# Quarterly data: assign same value to all 3 months in each quarter
# Expand to monthly frequency for merging with monthly FOMC meeting data
rows = []
for _, row in district_cpi.iterrows():
    for month_offset in range(3):
        new_date = row['date'] + pd.DateOffset(months=month_offset)
        r = row.copy()
        r['date_monthly'] = new_date
        r['year_month'] = new_date.to_period('M')
        rows.append(r)

district_cpi_monthly = pd.DataFrame(rows)

print(f"\n    Expanded to monthly: {len(district_cpi_monthly)} rows")

# ============================================================================
# SAVE
# ============================================================================

print("\n[7] Saving...")

# Save quarterly version
district_cpi.to_csv(f"{CACHE_DIR}/district_inflation_bobrov.csv", index=False)
print(f"    Saved: {CACHE_DIR}/district_inflation_bobrov.csv")

# Save monthly version (for merging with reg.py)
district_cpi_monthly[['year_month', 'district', 'inflation_gap', 
                        'inflation_gap_t', 'inflation_gap_all']].to_csv(
    f"{CACHE_DIR}/regional_inflation.csv", index=False
)
print(f"    Saved: {CACHE_DIR}/regional_inflation.csv (overwrites metro-area version)")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*70)
print("SUMMARY — Average inflation gap by district (nontradable)")
print("="*70)
summary = (
    district_cpi
    .groupby('district')['inflation_gap']
    .agg(['mean', 'std', 'count'])
    .round(4)
    .sort_values('mean')
)
print(summary)

missing_districts = [d for d in ['Boston','Philadelphia','Cleveland','Richmond',
                                   'Atlanta','Chicago','St. Louis','Minneapolis',
                                   'Kansas City','Dallas','San Francisco']
                     if d not in district_cpi['district'].unique()]
if missing_districts:
    print(f"\nWarning: Missing districts: {missing_districts}")
    print("These will be excluded from Table A5.")

print("\n✅ Done! Now run reg.py or the inflation robustness script.")