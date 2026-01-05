# Data Pipeline Explanation

## Overview

You have a multi-stage data pipeline for your FOMC thesis. Here's how it works:

## Stage 1: Raw Data → 2006-2017 Filtered Data

The 3 input pickle files ARE created using embeddings:

### 1. `extracted_transcripts_2006_2017.pkl`
- **Source**: `extracted_transcripts.pkl` (full 1976-2018 transcripts)
- **Process**: Simple date filter to 2006-2017
- **Contains**: Raw transcript text + metadata (speaker, date, word_count)
- **Size**: 21,316 rows (speaker-statement pairs)
- **Columns**: `['date', 'year', 'month', 'speaker', 'text', 'word_count']`

### 2. `regional_dissent_free.pkl` ⭐ **USES EMBEDDINGS**
- **What it is**: Semantic dissent scores computed via OpenAI embeddings
- **Process**:
  1. For each FOMC meeting, get all speakers' statements
  2. Compute OpenAI embeddings for each statement
  3. Compute consensus embedding (average of all speakers)
  4. Find Fed Chair's embedding
  5. Calculate cosine distance from each speaker to:
     - Consensus → `dissent_consensus`
     - Fed Chair → `dissent_chair`
- **Contains**: Pre-computed dissent scores (NOT raw embeddings)
- **Size**: 2,440 rows (speaker-meeting pairs)
- **Columns**: `['date', 'speaker', 'dissent_consensus', 'dissent_chair', 'num_speakers']`
- **Interpretation**:
  - Higher score = more dissent (more different from consensus/chair)
  - Range: 0.0 to 1.0 (cosine distance)
  - Mean dissent_consensus: 0.26
  - Mean dissent_chair: 0.42

### 3. `unemployment_2006_2017.pkl`
- **Source**: `regional_unemployment.csv` (or external data)
- **Process**: Filter to 2006-2017, use national unemployment rate
- **Contains**: Monthly unemployment rates
- **Size**: 1,728 rows (monthly observations)
- **Columns**: `['date', 'unemployment_rate', 'state']`

## Stage 2: Regional Analysis

**Script**: `prepare_regional_data.py`

**Input**: The 3 files above
**Output**: `fomc_transcripts_speakers.pkl`

**Process**:
1. Load transcripts (21,316 rows)
2. Filter to only Regional Bank Presidents (9,109 statements from 24 speakers)
3. Map speakers to their Fed districts (Atlanta, Boston, Chicago, etc.)
4. Add dissent information from `regional_dissent_free.pkl`
5. Add unemployment rates
6. Create final speaker-level dataset

## Stage 3: Regression Analysis

**Script**: `reg.py`

**Input**: `fomc_transcripts_speakers.pkl`
**Output**: Regression results analyzing regional unemployment effects

**Process**: ⭐ **ALSO USES EMBEDDINGS**
1. Load speaker data
2. Compute semantic similarity between:
   - Each speaker's text
   - Reference phrases about unemployment
3. Run regressions testing if regional unemployment affects dissent

## Key Points

### Where Embeddings Are Used:
1. ✅ **Creating `regional_dissent_free.pkl`** - Computes semantic distance between speakers
2. ✅ **In `reg.py`** - Computes semantic similarity to unemployment topics

### Where Embeddings Are NOT Used:
- `extracted_transcripts_2006_2017.pkl` - Just raw text filtering
- `unemployment_2006_2017.pkl` - Just unemployment data
- `prepare_regional_data.py` - Just merges data, no embeddings

## How to Recreate the Files

If you need to recreate the 3 input files (e.g., on a new computer):

```bash
# Option 1: Run the automated script
python create_2006_2017_data.py

# Option 2: Copy from your current computer
# They're already in: /Users/CS/Documents/GitHub/thesis/data/cache/
```

**⚠️ Warning**: `create_2006_2017_data.py` will make ~2,000+ OpenAI API calls to recreate `regional_dissent_free.pkl` (costs ~$1-2)

## Data Flow Diagram

```
Raw HTMLs (1976-2018)
    ↓ (extraction scripts in new-programs/)
extracted_transcripts.pkl (92,748 rows)
    ↓ (filter to 2006-2017)
extracted_transcripts_2006_2017.pkl (21,316 rows)
    ↓
    ├→ + regional_dissent_free.pkl (embeddings! 2,440 rows)
    ├→ + unemployment_2006_2017.pkl (1,728 rows)
    ↓
[prepare_regional_data.py]
    ↓
fomc_transcripts_speakers.pkl (9,109 rows - regional presidents only)
    ↓
[reg.py] (more embeddings!)
    ↓
Regression results
```

## Quick Check

To verify your files are correct:
```bash
python examine_pickles.py
```

This shows the structure, size, and sample data from each file.
