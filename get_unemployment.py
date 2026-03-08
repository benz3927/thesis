#!/usr/bin/env python3
"""
Download all available unemployment data from FRED for all states
Aggregate to Fed district level using POPULATION-WEIGHTED averages
"""

import pandas as pd
import numpy as np
from fredapi import Fred
import os
from tqdm import tqdm
import time
from dotenv import load_dotenv

load_dotenv()

CACHE_DIR = 'data/cache'
fred = Fred(api_key=os.getenv('FRED_API_KEY'))

# State to district mapping
STATE_TO_DISTRICT = {
    'CT': 'Boston', 'MA': 'Boston', 'ME': 'Boston', 'NH': 'Boston', 'RI': 'Boston', 'VT': 'Boston',
    'NY': 'New York',
    'DE': 'Philadelphia', 'NJ': 'Philadelphia', 'PA': 'Philadelphia',
    'OH': 'Cleveland', 'KY': 'Cleveland',
    'DC': 'Richmond', 'MD': 'Richmond', 'NC': 'Richmond', 'SC': 'Richmond', 'VA': 'Richmond', 'WV': 'Richmond',
    'AL': 'Atlanta', 'FL': 'Atlanta', 'GA': 'Atlanta', 'MS': 'Atlanta', 'TN': 'Atlanta',
    'IA': 'Chicago', 'IL': 'Chicago', 'IN': 'Chicago', 'MI': 'Chicago', 'WI': 'Chicago',
    'AR': 'St. Louis', 'MO': 'St. Louis',
    'MN': 'Minneapolis', 'MT': 'Minneapolis', 'ND': 'Minneapolis', 'SD': 'Minneapolis',
    'CO': 'Kansas City', 'KS': 'Kansas City', 'NE': 'Kansas City', 'NM': 'Kansas City', 'OK': 'Kansas City', 'WY': 'Kansas City',
    'TX': 'Dallas', 'LA': 'Dallas',
    'AK': 'San Francisco', 'AZ': 'San Francisco', 'CA': 'San Francisco', 'HI': 'San Francisco',
    'ID': 'San Francisco', 'NV': 'San Francisco', 'OR': 'San Francisco', 'UT': 'San Francisco', 'WA': 'San Francisco',
}

STATE_SERIES = {
    'AL': 'ALUR', 'AK': 'AKUR', 'AZ': 'AZUR', 'AR': 'ARUR', 'CA': 'CAUR', 'CO': 'COUR',
    'CT': 'CTUR', 'DE': 'DEUR', 'DC': 'DCUR', 'FL': 'FLUR', 'GA': 'GAUR', 'HI': 'HIUR',
    'ID': 'IDUR', 'IL': 'ILUR', 'IN': 'INUR', 'IA': 'IAUR', 'KS': 'KSUR', 'KY': 'KYUR',
    'LA': 'LAUR', 'ME': 'MEUR', 'MD': 'MDUR', 'MA': 'MAUR', 'MI': 'MIUR', 'MN': 'MNUR',
    'MS': 'MSUR', 'MO': 'MOUR', 'MT': 'MTUR', 'NE': 'NEUR', 'NV': 'NVUR', 'NH': 'NHUR',
    'NJ': 'NJUR', 'NM': 'NMUR', 'NY': 'NYUR', 'NC': 'NCUR', 'ND': 'NDUR', 'OH': 'OHUR',
    'OK': 'OKUR', 'OR': 'ORUR', 'PA': 'PAUR', 'RI': 'RIUR', 'SC': 'SCUR', 'SD': 'SDUR',
    'TN': 'TNUR', 'TX': 'TXUR', 'UT': 'UTUR', 'VT': 'VTUR', 'VA': 'VAUR', 'WA': 'WAUR',
    'WV': 'WVUR', 'WI': 'WIUR', 'WY': 'WYUR'
}

# FRED annual population series (thousands)
# Format: {state_abbr}POP  (annual, in thousands)
STATE_POP_SERIES = {
    'AL': 'ALPOP', 'AK': 'AKPOP', 'AZ': 'AZPOP', 'AR': 'ARPOP', 'CA': 'CAPOP', 'CO': 'COPOP',
    'CT': 'CTPOP', 'DE': 'DEPOP', 'DC': 'DCPOP', 'FL': 'FLPOP', 'GA': 'GAPOP', 'HI': 'HIPOP',
    'ID': 'IDPOP', 'IL': 'ILPOP', 'IN': 'INPOP', 'IA': 'IAPOP', 'KS': 'KSPOP', 'KY': 'KYPOP',
    'LA': 'LAPOP', 'ME': 'MEPOP', 'MD': 'MDPOP', 'MA': 'MAPOP', 'MI': 'MIPOP', 'MN': 'MNPOP',
    'MS': 'MSPOP', 'MO': 'MOPOP', 'MT': 'MTPOP', 'NE': 'NEPOP', 'NV': 'NVPOP', 'NH': 'NHPOP',
    'NJ': 'NJPOP', 'NM': 'NMPOP', 'NY': 'NYPOP', 'NC': 'NCPOP', 'ND': 'NDPOP', 'OH': 'OHPOP',
    'OK': 'OKPOP', 'OR': 'ORPOP', 'PA': 'PAPOP', 'RI': 'RIPOP', 'SC': 'SCPOP', 'SD': 'SDPOP',
    'TN': 'TNPOP', 'TX': 'TXPOP', 'UT': 'UTPOP', 'VT': 'VTPOP', 'VA': 'VAPOP', 'WA': 'WAPOP',
    'WV': 'WVPOP', 'WI': 'WIPOP', 'WY': 'WYPOP'
}

# ============================================================================
# 1. Download unemployment rates
# ============================================================================
print("=" * 60)
print("STEP 1: Downloading state unemployment rates from FRED...")
print("=" * 60)

state_data = []
for state, series_id in tqdm(STATE_SERIES.items(), desc="Unemployment"):
    try:
        data = fred.get_series(series_id)
        for date, value in data.items():
            if pd.notna(value):
                state_data.append({
                    'date': date,
                    'state': state,
                    'unemployment_rate': value
                })
        time.sleep(0.1)
    except Exception as e:
        print(f"  ⚠ Error for {state} ({series_id}): {e}")

df = pd.DataFrame(state_data)
df['date'] = pd.to_datetime(df['date'])
df['district'] = df['state'].map(STATE_TO_DISTRICT)
print(f"  ✓ {len(df)} state-month observations")

# ============================================================================
# 2. Download population data (annual, in thousands)
# ============================================================================
print("\n" + "=" * 60)
print("STEP 2: Downloading state population from FRED...")
print("=" * 60)

pop_data = []
for state, series_id in tqdm(STATE_POP_SERIES.items(), desc="Population"):
    try:
        data = fred.get_series(series_id)
        for date, value in data.items():
            if pd.notna(value):
                pop_data.append({
                    'date': date,
                    'state': state,
                    'population': value * 1000  # convert from thousands
                })
        time.sleep(0.1)
    except Exception as e:
        print(f"  ⚠ Error for {state} ({series_id}): {e}")

pop_df = pd.DataFrame(pop_data)
pop_df['date'] = pd.to_datetime(pop_df['date'])
pop_df['year'] = pop_df['date'].dt.year
print(f"  ✓ {len(pop_df)} state-year population observations")

# ============================================================================
# 3. Merge population onto monthly unemployment
# ============================================================================
print("\n" + "=" * 60)
print("STEP 3: Merging population weights...")
print("=" * 60)

# Population is annual — merge on state + year
df['year'] = df['date'].dt.year
pop_annual = pop_df.groupby(['state', 'year'])['population'].first().reset_index()

df = df.merge(pop_annual, on=['state', 'year'], how='left')

# Check for missing population
missing_pop = df['population'].isna().sum()
if missing_pop > 0:
    print(f"  ⚠ {missing_pop} rows missing population — filling with nearest year")
    # Forward/backward fill within each state
    df = df.sort_values(['state', 'date'])
    df['population'] = df.groupby('state')['population'].transform(
        lambda x: x.fillna(method='ffill').fillna(method='bfill')
    )
    still_missing = df['population'].isna().sum()
    if still_missing > 0:
        print(f"  ⚠ Still {still_missing} missing after fill — dropping")
        df = df.dropna(subset=['population'])

print(f"  ✓ Merged. {len(df)} observations with population weights")

# ============================================================================
# 4. Aggregate to district level with population weights
# ============================================================================
print("\n" + "=" * 60)
print("STEP 4: Computing population-weighted district unemployment...")
print("=" * 60)

def weighted_avg(group):
    weights = group['population']
    values = group['unemployment_rate']
    if weights.sum() == 0:
        return values.mean()
    return np.average(values, weights=weights)

district_df = df.groupby(['date', 'district']).apply(
    weighted_avg, include_groups=False
).reset_index()
district_df.columns = ['date', 'district', 'unemployment_rate']

# Also save the unweighted version for comparison
district_unweighted = df.groupby(['date', 'district']).agg({
    'unemployment_rate': 'mean'
}).reset_index()
district_unweighted.columns = ['date', 'district', 'unemployment_rate_unweighted']

# Merge for comparison
compare = district_df.merge(district_unweighted, on=['date', 'district'])

print(f"\n  ✓ {len(district_df)} district-month observations")
print(f"  Date range: {district_df['date'].min().date()} to {district_df['date'].max().date()}")
print(f"  Districts: {sorted(district_df['district'].unique())}")

# Show difference between weighted and unweighted
print("\n  Weighted vs Unweighted comparison (avg absolute difference):")
for dist in sorted(compare['district'].unique()):
    sub = compare[compare['district'] == dist]
    diff = (sub['unemployment_rate'] - sub['unemployment_rate_unweighted']).abs().mean()
    print(f"    {dist:<16} {diff:.3f} pp")

# ============================================================================
# 5. Save
# ============================================================================
print("\n" + "=" * 60)
print("STEP 5: Saving...")
print("=" * 60)

# Save weighted (this is the one your regression uses)
output_file = f'{CACHE_DIR}/regional_unemployment_all.csv'
district_df.to_csv(output_file, index=False)
print(f"  💾 Weighted:   {output_file}")

# Save unweighted for robustness check
output_unw = f'{CACHE_DIR}/regional_unemployment_unweighted.csv'
district_unweighted.to_csv(output_unw, index=False)
print(f"  💾 Unweighted: {output_unw}")

# Also save state-level for reference
state_output = f'{CACHE_DIR}/state_unemployment_all.csv'
df[['date', 'state', 'district', 'unemployment_rate', 'population']].to_csv(state_output, index=False)
print(f"  💾 State-level: {state_output}")

print(f"\n✅ Done! Population-weighted district unemployment ready.")
print(f"   Re-run reg.py — it reads from {output_file} so no code changes needed.")