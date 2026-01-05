#!/usr/bin/env python3
"""
Quick script to examine what's in the 3 input pickle files
"""
import pickle
import pandas as pd

CACHE_DIR = 'data/cache'

print("="*70)
print("EXAMINING PICKLE FILES")
print("="*70)

# 1. Extracted transcripts 2006-2017
print("\n1. EXTRACTED TRANSCRIPTS 2006-2017")
print("-" * 70)
with open(f'{CACHE_DIR}/extracted_transcripts_2006_2017.pkl', 'rb') as f:
    transcripts = pickle.load(f)
print(f"Type: {type(transcripts)}")
print(f"Shape: {transcripts.shape}")
print(f"Columns: {list(transcripts.columns)}")
print(f"\nFirst few rows:")
print(transcripts.head())
print(f"\nDate range: {transcripts['date'].min()} to {transcripts['date'].max()}")
print(f"Unique speakers: {transcripts['speaker'].nunique()}")

# 2. Regional dissent data
print("\n\n2. REGIONAL DISSENT DATA")
print("-" * 70)
with open(f'{CACHE_DIR}/regional_dissent_free.pkl', 'rb') as f:
    dissent = pickle.load(f)
print(f"Type: {type(dissent)}")
print(f"Shape: {dissent.shape}")
print(f"Columns: {list(dissent.columns)}")
print(f"\nFirst few rows:")
print(dissent.head())

# 3. Unemployment 2006-2017
print("\n\n3. UNEMPLOYMENT 2006-2017")
print("-" * 70)
with open(f'{CACHE_DIR}/unemployment_2006_2017.pkl', 'rb') as f:
    unemployment = pickle.load(f)
print(f"Type: {type(unemployment)}")
print(f"Shape: {unemployment.shape}")
print(f"Columns: {list(unemployment.columns)}")
print(f"\nFirst few rows:")
print(unemployment.head())
print(f"\nDate range: {unemployment['date'].min()} to {unemployment['date'].max()}")

print("\n" + "="*70)
print("CONCLUSION:")
print("="*70)
print("These are NOT embeddings - they are raw DataFrames containing:")
print("  1. Transcript text + metadata (speaker, date, etc.)")
print("  2. Dissent/voting records")
print("  3. Unemployment rates by date")
print("\nEmbeddings are computed LATER in reg.py for semantic similarity.")
print("="*70)
