#!/usr/bin/env python3
"""
Quick diagnostic: Check if unemployment data makes sense
"""

import pandas as pd
import pickle

CACHE_DIR = 'data/cache'

print("="*70)
print("UNEMPLOYMENT DATA DIAGNOSTICS")
print("="*70)

# ============================================================================
# CHECK 1: Load and inspect unemployment CSV
# ============================================================================

print("\n[CHECK 1] Inspecting regional_unemployment.csv")
print("-"*70)

unemp = pd.read_csv(f'{CACHE_DIR}/regional_unemployment.csv')
unemp['date'] = pd.to_datetime(unemp['date'])

print(f"\nShape: {unemp.shape}")
print(f"Columns: {list(unemp.columns)}")
print(f"\nFirst 20 rows:")
print(unemp.head(20))

print(f"\n\nUnique districts: {sorted(unemp['district'].unique())}")
print(f"Date range: {unemp['date'].min()} to {unemp['date'].max()}")
print(f"Unemployment range: {unemp['unemployment_rate'].min():.2f}% to {unemp['unemployment_rate'].max():.2f}%")

# Check variation across districts at a single point in time
print("\n" + "="*70)
print("VARIATION CHECK: Same date, different districts?")
print("="*70)

sample_dates = unemp['date'].unique()[:3]
for sample_date in sample_dates:
    sample = unemp[unemp['date'] == sample_date].sort_values('unemployment_rate')
    print(f"\nDate: {sample_date}")
    print(f"Districts with unemployment:")
    for _, row in sample.iterrows():
        print(f"  {row['district']:20s}: {row['unemployment_rate']:.2f}%")
    print(f"  Range: {sample['unemployment_rate'].min():.2f}% - {sample['unemployment_rate'].max():.2f}%")
    print(f"  Std Dev: {sample['unemployment_rate'].std():.2f}%")
    
    if sample['unemployment_rate'].std() < 0.1:
        print(f"  ⚠️  WARNING: Very low variation! All districts nearly identical.")

# Check variation within a district over time
print("\n" + "="*70)
print("TIME VARIATION CHECK: Same district, different times?")
print("="*70)

test_districts = ['Boston', 'San Francisco', 'Dallas']
for district in test_districts:
    district_data = unemp[unemp['district'] == district].sort_values('date')
    if len(district_data) > 0:
        print(f"\n{district}:")
        print(f"  Date range: {district_data['date'].min()} to {district_data['date'].max()}")
        print(f"  Unemployment range: {district_data['unemployment_rate'].min():.2f}% - {district_data['unemployment_rate'].max():.2f}%")
        print(f"  Mean: {district_data['unemployment_rate'].mean():.2f}%")
        print(f"  Std Dev: {district_data['unemployment_rate'].std():.2f}%")
        
        print(f"\n  Sample time series:")
        sample_times = district_data.iloc[::max(1, len(district_data)//5)]  # 5 evenly spaced points
        for _, row in sample_times.iterrows():
            print(f"    {row['date'].strftime('%Y-%m')}: {row['unemployment_rate']:.2f}%")

# ============================================================================
# CHECK 2: Compare to known historical unemployment
# ============================================================================

print("\n" + "="*70)
print("SANITY CHECK: Does this match known history?")
print("="*70)

print("\nExpected patterns (2006-2017):")
print("  - 2006-2007: ~4-5% (pre-crisis)")
print("  - 2009-2010: ~9-10% (peak of Great Recession)")
print("  - 2017: ~4-5% (recovery)")

for year in [2006, 2009, 2010, 2017]:
    year_data = unemp[unemp['date'].dt.year == year]
    if len(year_data) > 0:
        avg_unemp = year_data['unemployment_rate'].mean()
        min_unemp = year_data['unemployment_rate'].min()
        max_unemp = year_data['unemployment_rate'].max()
        print(f"\n{year}: avg={avg_unemp:.2f}%, range=[{min_unemp:.2f}%, {max_unemp:.2f}%]")
        
        if year in [2009, 2010] and avg_unemp < 7.0:
            print(f"  ⚠️  WARNING: Too low for recession period!")
        if year in [2006, 2017] and avg_unemp > 6.0:
            print(f"  ⚠️  WARNING: Too high for this period!")

# ============================================================================
# CHECK 3: Merge check
# ============================================================================

print("\n" + "="*70)
print("MERGE CHECK: Will transcripts match unemployment?")
print("="*70)

# Load transcripts
with open(f'{CACHE_DIR}/transcripts_with_scores_2006_2017.pkl', 'rb') as f:
    transcripts = pickle.load(f)

# Extract speaker names
def extract_last_name(speaker):
    if pd.isna(speaker):
        return None
    speaker = str(speaker).upper().strip()
    for prefix in ['MR ', 'MS ', 'CHAIRMAN ', 'VICE CHAIRMAN ', 'PRESIDENT ', 'GOVERNOR ', 'RPIX']:
        if speaker.startswith(prefix):
            speaker = speaker[len(prefix):].strip()
    parts = speaker.split()
    return parts[0] if parts else None

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

transcripts['speaker_clean'] = transcripts['speaker'].apply(extract_last_name)
transcripts['district'] = transcripts['speaker_clean'].map(SPEAKER_TO_DISTRICT)
bank_presidents = transcripts[transcripts['district'].notna()].copy()

print(f"\nTranscript date range: {bank_presidents['date'].min()} to {bank_presidents['date'].max()}")
print(f"Unemployment date range: {unemp['date'].min()} to {unemp['date'].max()}")

print(f"\nTranscript districts: {sorted(bank_presidents['district'].unique())}")
print(f"Unemployment districts: {sorted(unemp['district'].unique())}")

# Test a sample merge
bank_presidents['year_month'] = bank_presidents['date'].dt.to_period('M')
unemp['year_month'] = unemp['date'].dt.to_period('M')

test_merge = bank_presidents.merge(
    unemp[['year_month', 'district', 'unemployment_rate']],
    on=['year_month', 'district'],
    how='left'
)

print(f"\n\nMerge test:")
print(f"  Transcripts: {len(bank_presidents)}")
print(f"  After merge: {len(test_merge)}")
print(f"  With unemployment: {test_merge['unemployment_rate'].notna().sum()}")
print(f"  Missing unemployment: {test_merge['unemployment_rate'].isna().sum()}")

if test_merge['unemployment_rate'].notna().sum() > 0:
    print(f"\n  Merged unemployment range: {test_merge['unemployment_rate'].min():.2f}% - {test_merge['unemployment_rate'].max():.2f}%")
    
    # Show a few examples
    print(f"\n  Sample merged records:")
    sample = test_merge[test_merge['unemployment_rate'].notna()].sample(min(10, len(test_merge)))
    for _, row in sample.iterrows():
        print(f"    {row['date'].strftime('%Y-%m')}: {row['speaker_clean']:15s} ({row['district']:15s}) → {row['unemployment_rate']:.2f}%")

print("\n" + "="*70)
print("DIAGNOSIS COMPLETE")
print("="*70)

print("\n🔍 Key Questions:")
print("  1. Do districts have DIFFERENT unemployment at the same time?")
print("  2. Does each district's unemployment CHANGE over time?")
print("  3. Do the values match known historical patterns?")
print("\nIf NO to any of these, your unemployment data needs to be fixed!")