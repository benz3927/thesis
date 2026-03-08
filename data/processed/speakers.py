#!/usr/bin/env python3
"""
Check when specific Bank Presidents served
"""

import pandas as pd
import glob

print("="*80)
print("CHECKING BANK PRESIDENT TENURE")
print("="*80)

# Load all transcripts
files = glob.glob('data/processed/Transcripts/*_t.csv')
print(f"\nLoading {len(files)} files...")

all_data = []
for f in files:
    df = pd.read_csv(f)
    
    # Rename speaker column if needed
    if 'Speaker' in df.columns:
        df = df.rename(columns={'Speaker': 'speaker'})
    
    # Extract date from filename (format: YYYYMMDD_t.csv)
    date_str = f.split('/')[-1].split('_')[0]
    date = pd.to_datetime(date_str, format='%Y%m%d')
    df['date'] = date
    df['year'] = date.year
    
    all_data.append(df[['speaker', 'date', 'year']])

all_df = pd.concat(all_data, ignore_index=True)

print(f"✅ Loaded {len(all_df):,} total statements")
print(f"   Year range: {all_df['year'].min()} - {all_df['year'].max()}")

# Check specific speakers
speakers_to_check = [
    'JORDAN', 'BROADDUS', 'PARRY', 'MCTEER',
    'BOEHNE', 'MELZER', 'SANTOMERO', 'FORRESTAL',
    'SYRON', 'KEEHN', 'MCDONOUGH'
]

print("\n" + "="*80)
print("SPEAKER TENURE ANALYSIS")
print("="*80)

for name in speakers_to_check:
    matches = all_df[all_df['speaker'].str.contains(name, case=False, na=False)]
    
    if len(matches) > 0:
        in_2006_2017 = matches[matches['year'].between(2006, 2017)]
        
        print(f"\n{name}:")
        print(f"  Total statements: {len(matches):,}")
        print(f"  Year range: {matches['year'].min()} - {matches['year'].max()}")
        print(f"  In 2006-2017: {len(in_2006_2017):,} statements")
        
        if len(in_2006_2017) > 0:
            print(f"  ✅ ACTIVE in analysis period")
        else:
            print(f"  ❌ NOT in analysis period (retired before 2006)")
    else:
        print(f"\n{name}: NOT FOUND")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)

# Count how many are in 2006-2017
in_period = 0
out_of_period = 0

for name in speakers_to_check:
    matches = all_df[all_df['speaker'].str.contains(name, case=False, na=False)]
    if len(matches) > 0:
        in_2006_2017 = matches[matches['year'].between(2006, 2017)]
        if len(in_2006_2017) > 0:
            in_period += 1
        else:
            out_of_period += 1

print(f"\nOf the {len(speakers_to_check)} checked speakers:")
print(f"  ✅ Active in 2006-2017: {in_period}")
print(f"  ❌ Retired before 2006: {out_of_period}")

print("\n" + "="*80)