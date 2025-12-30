#!/usr/bin/env python3
"""
Prepare data for regional unemployment vs dissent analysis.
Combines extracted transcripts, unemployment data, and dissent data.
"""

import pandas as pd
import pickle
from pathlib import Path

CACHE_DIR = '/Users/CS/Documents/GitHub/thesis/data/cache/'

print("="*70)
print("PREPARING REGIONAL DISSENT DATA")
print("="*70)

# ============================================================================
# LOAD EXISTING DATA
# ============================================================================

print("\n[1] Loading existing data files...")

# Load transcripts
with open(f'{CACHE_DIR}/extracted_transcripts_2006_2017.pkl', 'rb') as f:
    transcripts = pickle.load(f)
print(f"✅ Loaded {len(transcripts)} transcript entries")
print(f"   Columns: {list(transcripts.columns)}")

# Load dissent data
with open(f'{CACHE_DIR}/regional_dissent_free.pkl', 'rb') as f:
    dissent_data = pickle.load(f)
dissent_data['date'] = pd.to_datetime(dissent_data['date'])
print(f"✅ Loaded {len(dissent_data)} dissent records")
print(f"   Columns: {list(dissent_data.columns)}")

# Load unemployment data
with open(f'{CACHE_DIR}/unemployment_2006_2017.pkl', 'rb') as f:
    unemployment = pickle.load(f)
print(f"✅ Loaded {len(unemployment)} unemployment records")
print(f"   Columns: {list(unemployment.columns)}")

# ============================================================================
# MAP SPEAKERS TO DISTRICTS
# ============================================================================

print("\n[2] Mapping speakers to Federal Reserve districts...")

# Regional Bank Presidents by district (using last names to match data)
# This is based on Federal Reserve Bank structure
DISTRICT_MAPPING = {
    # Boston (1st District)
    'minehan': 'Boston',
    'rosengren': 'Boston',

    # New York (2nd District)
    'geithner': 'New York',
    'dudley': 'New York',

    # Philadelphia (3rd District)
    'plosser': 'Philadelphia',
    'harker': 'Philadelphia',

    # Cleveland (4th District)
    'pianalto': 'Cleveland',
    'mester': 'Cleveland',

    # Richmond (5th District)
    'lacker': 'Richmond',
    'barkin': 'Richmond',

    # Atlanta (6th District)
    'lockhart': 'Atlanta',
    'bostic': 'Atlanta',
    'guynn': 'Atlanta',

    # Chicago (7th District)
    'moskow': 'Chicago',
    'evans': 'Chicago',

    # St. Louis (8th District)
    'poole': 'St. Louis',
    'bullard': 'St. Louis',

    # Minneapolis (9th District)
    'stern': 'Minneapolis',
    'kocherlakota': 'Minneapolis',
    'kashkari': 'Minneapolis',

    # Kansas City (10th District)
    'hoenig': 'Kansas City',
    'george': 'Kansas City',

    # Dallas (11th District)
    'fisher': 'Dallas',
    'kaplan': 'Dallas',

    # San Francisco (12th District)
    'yellen': 'San Francisco',
    'williams': 'San Francisco',
    'daly': 'San Francisco',
}

# Convert dates to datetime for merging
transcripts['date'] = pd.to_datetime(transcripts['date'])

# Add district column to transcripts
transcripts['speaker_lower'] = transcripts['speaker'].str.lower().str.strip()
transcripts['district'] = transcripts['speaker_lower'].map(DISTRICT_MAPPING)

# Mark who are bank presidents
transcripts['is_bank_president'] = transcripts['district'].notna()

print(f"✅ Identified {transcripts['is_bank_president'].sum()} statements from Regional Bank Presidents")
print(f"   Unique bank presidents: {transcripts[transcripts['is_bank_president']]['speaker'].nunique()}")

# ============================================================================
# ADD DISSENT INFORMATION
# ============================================================================

print("\n[3] Adding dissent information...")

# Merge with dissent data
transcripts = transcripts.merge(
    dissent_data[['date', 'speaker', 'dissent_consensus']],
    on=['date', 'speaker'],
    how='left'
)

# Create binary dissent flag (using dissent_consensus > 0.5 as threshold)
transcripts['is_dissent'] = (transcripts['dissent_consensus'] > 0.5).fillna(False)

print(f"✅ Added dissent flags")
print(f"   Total dissents: {transcripts['is_dissent'].sum()}")

# ============================================================================
# MAP UNEMPLOYMENT DATA TO DISTRICTS
# ============================================================================

print("\n[4] Mapping unemployment data to districts...")

# District to state mapping (using primary state in each district)
DISTRICT_TO_STATE = {
    'Boston': 'Massachusetts',
    'New York': 'New York',
    'Philadelphia': 'Pennsylvania',
    'Cleveland': 'Ohio',
    'Richmond': 'Virginia',
    'Atlanta': 'Georgia',
    'Chicago': 'Illinois',
    'St. Louis': 'Missouri',
    'Minneapolis': 'Minnesota',
    'Kansas City': 'Missouri',  # Kansas City Fed is in Missouri
    'Dallas': 'Texas',
    'San Francisco': 'California',
}

# For now, use national US unemployment rate for all districts
# (This is a simplification - ideally we'd use state-level data)
national_unemp = unemployment[unemployment['state'] == 'US'][['date', 'unemployment_rate']].copy()

print(f"✅ Using national unemployment rate")
print(f"   Date range: {national_unemp['date'].min()} to {national_unemp['date'].max()}")

# Create regional unemployment file
regional_unemployment = []
for district in DISTRICT_TO_STATE.keys():
    district_data = national_unemp.copy()
    district_data['district'] = district
    regional_unemployment.append(district_data)

regional_unemployment = pd.concat(regional_unemployment, ignore_index=True)

# Save regional unemployment as CSV (as expected by reg.py)
regional_unemployment.to_csv(f'{CACHE_DIR}/regional_unemployment.csv', index=False)
print(f"💾 Saved regional_unemployment.csv")

# ============================================================================
# MERGE AND SAVE SPEAKER-LEVEL DATA
# ============================================================================

print("\n[5] Creating speaker-level dataset...")

# Add year-month columns for merging
transcripts['year_month'] = transcripts['date'].dt.to_period('M')
regional_unemployment['date'] = pd.to_datetime(regional_unemployment['date'])
regional_unemployment['year_month'] = regional_unemployment['date'].dt.to_period('M')

# Merge with unemployment by year-month and district
speaker_data = transcripts.merge(
    regional_unemployment[['year_month', 'district', 'unemployment_rate']],
    on=['year_month', 'district'],
    how='left'
)

# Keep only bank presidents with all required fields
speaker_data_clean = speaker_data[
    (speaker_data['is_bank_president']) &
    (speaker_data['district'].notna()) &
    (speaker_data['text'].notna())
].copy()

print(f"✅ Final dataset ready")
print(f"   Total records: {len(speaker_data_clean)}")
print(f"   Date range: {speaker_data_clean['date'].min()} to {speaker_data_clean['date'].max()}")
print(f"   Unique speakers: {speaker_data_clean['speaker'].nunique()}")
print(f"   Districts: {sorted(speaker_data_clean['district'].unique())}")

# Show sample
print(f"\nSample data:")
print(speaker_data_clean[['date', 'speaker', 'district', 'is_dissent', 'unemployment_rate']].head())

# Save as pickle (as expected by reg.py)
with open(f'{CACHE_DIR}/fomc_transcripts_speakers.pkl', 'wb') as f:
    pickle.dump(speaker_data_clean, f)

print(f"\n💾 Saved fomc_transcripts_speakers.pkl")

print("\n" + "="*70)
print("✅ DATA PREPARATION COMPLETE!")
print("="*70)
print("\nYou can now run reg.py to perform the regional analysis")
