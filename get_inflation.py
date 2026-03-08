#!/usr/bin/env python3
"""
Download regional CPI data from FRED and test inflation gaps
"""

import pandas as pd
import numpy as np
from fredapi import Fred
from datetime import datetime
import time
import os
from dotenv import load_dotenv

# ============================================================================
# SETUP
# ============================================================================

# Load environment variables
load_dotenv()
FRED_API_KEY = os.getenv('FRED_API_KEY')

if not FRED_API_KEY:
    raise ValueError("FRED_API_KEY not found in .env file!")

fred = Fred(api_key=FRED_API_KEY)

CACHE_DIR = 'data/cache'
os.makedirs(CACHE_DIR, exist_ok=True)

print("="*80)
print("DOWNLOADING REGIONAL CPI DATA FROM FRED")
print("="*80)
print(f"✅ Loaded FRED API key from .env")

# ============================================================================
# FRED SERIES FOR REGIONAL CPI (by metro area)
# ============================================================================

# FIXED: Better series IDs for Minneapolis and St. Louis
metro_cpi_series = {
    # Boston Fed
    'Boston': 'CUURA103SA0',  # Boston-Cambridge-Newton
    
    # New York Fed  
    'New York': 'CUURA101SA0',  # New York-Newark-Jersey City
    
    # Philadelphia Fed
    'Philadelphia': 'CUURA102SA0',  # Philadelphia-Camden-Wilmington
    
    # Cleveland Fed
    'Cleveland': 'CUURA210SA0',  # Cleveland-Akron
    
    # Richmond Fed
    'Richmond': 'CUURA311SA0',  # Washington-Arlington-Alexandria
    
    # Atlanta Fed
    'Atlanta': 'CUURA319SA0',  # Atlanta-Sandy Springs-Roswell
    
    # Chicago Fed
    'Chicago': 'CUURA207SA0',  # Chicago-Naperville-Elgin
    
    # St. Louis Fed - FIXED
    'St. Louis': 'CUUSS49ASA0',  # St. Louis (different format)
    
    # Minneapolis Fed - FIXED  
    'Minneapolis': 'CUURA211SA0',  # Minneapolis-St. Paul
    
    # Kansas City Fed
    'Kansas City': 'CUURA208SA0',  # Kansas City
    
    # Dallas Fed
    'Dallas': 'CUURA316SA0',  # Dallas-Fort Worth-Arlington
    
    # San Francisco Fed
    'San Francisco': 'CUURA422SA0',  # San Francisco-Oakland-Hayward
}

print(f"\n📍 Will download CPI for {len(metro_cpi_series)} metro areas")

# ============================================================================
# DOWNLOAD DATA
# ============================================================================

print("\n[1] Downloading CPI data from FRED...")

all_data = []

for district, series_id in metro_cpi_series.items():
    try:
        print(f"   Downloading {district}... ", end='')
        
        # Get CPI data (2005-2018 to match your sample)
        cpi_data = fred.get_series(
            series_id, 
            observation_start='2005-01-01',
            observation_end='2018-12-31'
        )
        
        if len(cpi_data) == 0:
            print(f"⚠️  No data (series may not exist)")
            continue
        
        # Convert to DataFrame
        df = pd.DataFrame({
            'date': cpi_data.index,
            'cpi': cpi_data.values,
            'district': district
        })
        
        # FIXED: Ensure date is datetime
        df['date'] = pd.to_datetime(df['date'])
        
        all_data.append(df)
        print(f"✅ ({len(df)} observations)")
        
        # Be nice to FRED API
        time.sleep(0.5)
        
    except Exception as e:
        print(f"❌ Error: {e}")

if len(all_data) == 0:
    raise ValueError("No data downloaded!")

# Combine all data
cpi_df = pd.concat(all_data, ignore_index=True)

print(f"\n✅ Downloaded {len(cpi_df):,} CPI observations")
print(f"   Districts: {cpi_df['district'].nunique()}")
print(f"   Date range: {cpi_df['date'].min()} to {cpi_df['date'].max()}")

# ============================================================================
# COMPUTE INFLATION RATES
# ============================================================================

print("\n[2] Computing year-over-year inflation rates...")

# Sort by district and date
cpi_df = cpi_df.sort_values(['district', 'date'])

# Compute YoY inflation (12-month percent change)
cpi_df['inflation_rate'] = cpi_df.groupby('district')['cpi'].pct_change(12, fill_method=None) * 100

# Drop first 12 months (no YoY comparison)
cpi_df = cpi_df.dropna(subset=['inflation_rate'])

print(f"✅ Computed inflation rates")
print(f"   Mean inflation: {cpi_df['inflation_rate'].mean():.2f}%")
print(f"   Range: [{cpi_df['inflation_rate'].min():.2f}%, {cpi_df['inflation_rate'].max():.2f}%]")

# ============================================================================
# COMPUTE INFLATION GAP
# ============================================================================

print("\n[3] Computing inflation gaps (district - national)...")

# FIXED: Ensure date is datetime before merge
cpi_df['date'] = pd.to_datetime(cpi_df['date'])

# Compute national average inflation for each date
national_inflation = cpi_df.groupby('date')['inflation_rate'].mean().reset_index()
national_inflation.columns = ['date', 'national_inflation']
national_inflation['date'] = pd.to_datetime(national_inflation['date'])

# Merge back
cpi_df = pd.merge(cpi_df, national_inflation, on='date', how='left')

# Compute gap
cpi_df['inflation_gap'] = cpi_df['inflation_rate'] - cpi_df['national_inflation']

print(f"✅ Computed inflation gaps")
print(f"   Mean gap: {cpi_df['inflation_gap'].mean():.2f}pp")
print(f"   Range: [{cpi_df['inflation_gap'].min():.2f}pp, {cpi_df['inflation_gap'].max():.2f}pp]")

# Show average gaps by district
print(f"\n   Average inflation gap by district:")
district_gaps = cpi_df.groupby('district')['inflation_gap'].mean().sort_values()
for district, gap in district_gaps.items():
    direction = "ABOVE" if gap > 0 else "BELOW"
    print(f"      {district:15s}: {gap:+.2f}pp ({direction} national)")

# ============================================================================
# SAVE
# ============================================================================

print("\n[4] Saving data...")

output_file = f'{CACHE_DIR}/regional_inflation.csv'
cpi_df.to_csv(output_file, index=False)

print(f"✅ Saved: {output_file}")
print(f"   Rows: {len(cpi_df):,}")
print(f"   Columns: {list(cpi_df.columns)}")

# ============================================================================
# SUMMARY STATS
# ============================================================================

print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

print(f"\nBy district:")
summary = cpi_df.groupby('district').agg({
    'inflation_rate': ['mean', 'std', 'min', 'max'],
    'inflation_gap': ['mean', 'std'],
    'date': 'count'
}).round(2)
summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
print(summary)

# Check for missing districts
print(f"\n   Districts with data: {sorted(cpi_df['district'].unique())}")

print("\n" + "="*80)
print("✅ DOWNLOAD COMPLETE!")
print("="*80)
print("\nNext step: Run inflation analysis")
print("  python test_inflation_dissent.py")