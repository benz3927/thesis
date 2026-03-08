#!/usr/bin/env python3
"""
KEYWORD DISSENT SCORING v7 - FINAL ATTEMPT
With negation handling and stricter phrase matching
"""

import pandas as pd
import numpy as np
import os
import glob
import re
from tqdm import tqdm
from sklearn.metrics import roc_auc_score

CACHE_DIR = 'data/cache'
TRANSCRIPTS_DIR = 'data/processed/Transcripts'
OUTPUT_FILE = f'{CACHE_DIR}/keyword_dissent_scores_v7.csv'

print("="*70)
print("KEYWORD-BASED DISSENT SCORING v7 - FINAL ATTEMPT")
print("="*70)

# ============================================================================
# DICTIONARIES - FOCUS ON FIRST-PERSON POLICY STATEMENTS
# ============================================================================

print("\n[1] Loading dictionaries...")

# AGREEMENT - First person explicit support
AGREEMENT_PHRASES = [
    r'\bi support\b', r'\bi agree\b', r'\bi favor\b', r'\bi concur\b',
    r'\bi endorse\b', r'\bi can support\b', r'\bi will support\b',
    r'\bi am comfortable\b', r'\bi\'m comfortable\b',
    r'\bcount me in\b', r'\bsign me up\b',
    r'\bsupport (?:alternative |option )?[abc]\b',
    r'\bsupport the (?:chairman|chair|proposal|recommendation)\b'
]

# DOVISH - First person wants EASIER policy
DOVISH_PHRASES = [
    # Direct policy preference
    r'\bi (?:would )?prefer (?:to )?wait\b',
    r'\bi (?:would )?prefer (?:we )?hold\b', 
    r'\bi (?:would )?prefer (?:a )?(?:lower|smaller|less)\b',
    r'\bi think we should wait\b',
    r'\bi favor (?:waiting|holding|patience)\b',
    r'\bwe(?:\'re| are) moving too (?:fast|quickly|aggressively)\b',
    r'\bthis (?:is|seems) (?:too )?(?:premature|aggressive)\b',
    r'\btoo (?:soon|early|fast|aggressive|restrictive|tight)\b',
    r'\bshould (?:wait|hold|pause)\b',
    r'\bnot (?:yet|now|ready)\b',
    r'\bmore (?:accommodation|stimulus|easing)\b',
    r'\bcut (?:rates?|the rate)\b',
    r'\blower (?:rates?|the rate)\b',
    # Concerns suggesting easier policy
    r'\bgrowth is (?:weakening|slowing|declining)\b',
    r'\bworried about (?:growth|employment|jobs|recession)\b',
    r'\bconcerned about (?:growth|employment|jobs|recession)\b',
    r'\bdownside risks? (?:are|remain|concern)\b',
    r'\brisk of recession\b',
    r'\bunemployment (?:is |remains )?(?:too )?high\b',
    r'\blabor market (?:is )?weak\b'
]

# HAWKISH - First person wants TIGHTER policy
HAWKISH_PHRASES = [
    # Direct policy preference  
    r'\bi (?:would )?prefer (?:to )?(?:act|move|raise|tighten)\b',
    r'\bi (?:would )?prefer (?:a )?(?:higher|larger|more)\b',
    r'\bi think we should (?:act|move|raise|tighten)\b',
    r'\bi favor (?:acting|moving|raising|tightening)\b',
    r'\bwe(?:\'re| are) (?:falling |getting )?behind(?: the curve)?\b',
    r'\bthis (?:is|seems) (?:too )?(?:cautious|timid|slow)\b',
    r'\btoo (?:slow|cautious|easy|loose|accommodative|late)\b',
    r'\bshould (?:act|move|raise|tighten)\b',
    r'\bnot (?:enough|sufficient)\b',
    r'\bmore (?:tightening|aggressive)\b',
    r'\braise (?:rates?|the rate)\b',
    r'\bhigher (?:rates?|the rate)\b',
    # Concerns suggesting tighter policy
    r'\binflation is (?:rising|elevated|too high|a concern)\b',
    r'\bworried about inflation\b',
    r'\bconcerned about inflation\b',
    r'\bupside risks? to inflation\b',
    r'\boverheating\b',
    r'\blabor market (?:is )?(?:too )?tight\b',
    r'\bwage (?:pressure|growth)\b'
]

# NEGATION PATTERNS - flip the meaning
NEGATION_WINDOW = 4  # words

print(f"    Agreement phrases: {len(AGREEMENT_PHRASES)}")
print(f"    Dovish phrases: {len(DOVISH_PHRASES)}")
print(f"    Hawkish phrases: {len(HAWKISH_PHRASES)}")

# ============================================================================
# SCORING FUNCTION WITH NEGATION HANDLING
# ============================================================================

def has_negation_before(text, match_start, window=4):
    """Check if there's a negation word within N words before the match"""
    negations = ['not', "n't", 'no', 'never', 'neither', 'without', "don't", "doesn't", "didn't", "won't", "wouldn't", "shouldn't", "couldn't"]
    
    # Get text before match
    before_text = text[:match_start].lower()
    words_before = before_text.split()[-window:]
    
    for word in words_before:
        for neg in negations:
            if neg in word:
                return True
    return False

def score_with_direction(text, speaker):
    """Score dissent with direction using regex and negation handling"""
    
    if "CHAIR" in speaker.upper():
        return 0.0, 0, 0, 0, [], []
    
    text_lower = text.lower()
    word_count = len(text_lower.split())
    
    if word_count == 0:
        return 0.0, 0, 0, 0, [], []
    
    # Find matches with negation handling
    agreement_found = []
    dovish_found = []
    hawkish_found = []
    
    for pattern in AGREEMENT_PHRASES:
        for match in re.finditer(pattern, text_lower):
            if not has_negation_before(text_lower, match.start()):
                agreement_found.append(match.group())
    
    for pattern in DOVISH_PHRASES:
        for match in re.finditer(pattern, text_lower):
            if has_negation_before(text_lower, match.start()):
                # Negated dovish = hawkish
                hawkish_found.append(f"NOT:{match.group()}")
            else:
                dovish_found.append(match.group())
    
    for pattern in HAWKISH_PHRASES:
        for match in re.finditer(pattern, text_lower):
            if has_negation_before(text_lower, match.start()):
                # Negated hawkish = dovish
                dovish_found.append(f"NOT:{match.group()}")
            else:
                hawkish_found.append(match.group())
    
    agreement_count = len(agreement_found)
    dovish_count = len(dovish_found)
    hawkish_count = len(hawkish_found)
    
    # Normalize by text length (per 1000 words)
    norm = 1000 / word_count
    
    dovish_rate = dovish_count * norm
    hawkish_rate = hawkish_count * norm
    
    # Score: positive = dovish, negative = hawkish
    score = 0.0
    score += dovish_rate * 2.0
    score -= hawkish_rate * 2.0
    
    score = round(max(-10.0, min(10.0, score)), 1)
    
    return score, agreement_count, dovish_count, hawkish_count, dovish_found, hawkish_found

# ============================================================================
# DISTRICT MAPPING
# ============================================================================

DISTRICT_MAPPING = {
    'syron': 'Boston', 'minehan': 'Boston', 'rosengren': 'Boston',
    'mcdonough': 'New York', 'geithner': 'New York', 'dudley': 'New York',
    'boehne': 'Philadelphia', 'santomero': 'Philadelphia', 'plosser': 'Philadelphia', 'harker': 'Philadelphia',
    'jordan': 'Cleveland', 'pianalto': 'Cleveland', 'mester': 'Cleveland',
    'broaddus': 'Richmond', 'lacker': 'Richmond', 'barkin': 'Richmond',
    'forrestal': 'Atlanta', 'guynn': 'Atlanta', 'lockhart': 'Atlanta', 'bostic': 'Atlanta',
    'keehn': 'Chicago', 'moskow': 'Chicago', 'evans': 'Chicago',
    'melzer': 'St. Louis', 'poole': 'St. Louis', 'bullard': 'St. Louis',
    'stern': 'Minneapolis', 'kocherlakota': 'Minneapolis', 'kashkari': 'Minneapolis',
    'hoenig': 'Kansas City', 'george': 'Kansas City',
    'mcteer': 'Dallas', 'fisher': 'Dallas', 'kaplan': 'Dallas',
    'parry': 'San Francisco', 'yellen': 'San Francisco', 'williams': 'San Francisco', 'daly': 'San Francisco',
}

def get_district(speaker):
    if pd.isna(speaker):
        return None
    speaker_lower = str(speaker).lower()
    for name, district in DISTRICT_MAPPING.items():
        if name in speaker_lower:
            return district
    return None

# ============================================================================
# LOAD TRANSCRIPTS
# ============================================================================

print("\n[2] Loading transcripts...")

transcript_files = glob.glob(f'{TRANSCRIPTS_DIR}/*_t.csv')
all_data = []
for file_path in tqdm(transcript_files, desc="    Loading"):
    try:
        df = pd.read_csv(file_path)
        filename = os.path.basename(file_path)
        date_str = filename.split('_')[0]
        date = pd.to_datetime(date_str, format='%Y%m%d')
        
        text_col = 'clean_transcript_text' if 'clean_transcript_text' in df.columns else 'transcript_text'
        speaker_col = 'Speaker' if 'Speaker' in df.columns else 'speaker'
        
        if text_col not in df.columns or speaker_col not in df.columns:
            continue
        
        df = df[[speaker_col, text_col]].rename(columns={speaker_col: 'speaker', text_col: 'text'})
        df['date'] = date
        df = df.dropna(subset=['text', 'speaker'])
        all_data.append(df)
    except:
        continue

transcripts = pd.concat(all_data, ignore_index=True)
transcripts['district'] = transcripts['speaker'].apply(get_district)
print(f"    Loaded {len(transcripts):,} statements")

# ============================================================================
# AGGREGATE
# ============================================================================

print("\n[3] Aggregating to speaker-meeting level...")

grouped = (
    transcripts
    .groupby(['speaker', 'date', 'district'])
    .agg({'text': lambda x: ' '.join(x)})
    .reset_index()
)
grouped['year'] = grouped['date'].dt.year
print(f"    Speaker-meetings: {len(grouped):,}")

# ============================================================================
# SCORE ALL
# ============================================================================

print("\n[4] Scoring with direction...")

results = grouped.apply(
    lambda row: score_with_direction(row['text'], row['speaker']), axis=1
)

grouped['keyword_dissent_direction'] = [r[0] for r in results]
grouped['agreement_count'] = [r[1] for r in results]
grouped['dovish_count'] = [r[2] for r in results]
grouped['hawkish_count'] = [r[3] for r in results]
grouped['dovish_phrases'] = [', '.join(r[4][:5]) for r in results]  # top 5
grouped['hawkish_phrases'] = [', '.join(r[5][:5]) for r in results]  # top 5
grouped['word_count'] = grouped['text'].apply(lambda x: len(x.split()))

# ============================================================================
# VALIDATION
# ============================================================================

print("\n" + "="*70)
print("[5] VALIDATION")
print("="*70)

votes = pd.read_excel("data/FOMC_Dissents_Data.xlsx", skiprows=3)
votes["date"] = pd.to_datetime(votes["FOMC Meeting"])

dissent_records = []
for _, row in votes.iterrows():
    for col, direction in [("Dissenters Tighter", "tighter"), 
                            ("Dissenters Easier", "easier"),
                            ("Dissenters Other/Indeterminate", "other")]:
        if pd.notna(row.get(col)):
            for name in str(row[col]).split(", "):
                dissent_records.append({
                    "date": row["date"], 
                    "name": name.strip().upper(),
                    "direction": direction
                })

dissent_df = pd.DataFrame(dissent_records)
dissent_df = dissent_df[(dissent_df['date'].dt.year >= 1994) & (dissent_df['date'].dt.year <= 2020)]

# Check direction accuracy
print("\nDirection validation on known dissenters:\n")
correct_direction = 0
total_direction = 0
results_list = []

for _, dissent in dissent_df.iterrows():
    if dissent['direction'] == 'other':
        continue
    match = grouped[
        (grouped['date'] == dissent['date']) & 
        (grouped['speaker'].str.upper().str.contains(dissent['name']))
    ]
    if len(match) > 0:
        row = match.iloc[0]
        score = row['keyword_dissent_direction']
        
        if dissent['direction'] == 'tighter':
            correct_dir = score < 0
        else:  # easier
            correct_dir = score > 0
        
        total_direction += 1
        if correct_dir:
            correct_direction += 1
            check = "✓"
        else:
            check = "✗"
        
        results_list.append({
            'name': dissent['name'],
            'date': dissent['date'],
            'voted': dissent['direction'],
            'score': score,
            'correct': correct_dir
        })
        
        print(f"  {dissent['name']:<12} {dissent['date'].strftime('%Y-%m-%d')} | Voted: {dissent['direction']:<7} | Score: {score:+5.1f} {check}")

print(f"\nDirection accuracy: {correct_direction}/{total_direction} = {correct_direction/total_direction:.1%}")

# ============================================================================
# SAVE
# ============================================================================

print("\n[6] Saving results...")

output = grouped[['speaker', 'date', 'district', 'year', 'keyword_dissent_direction',
                  'agreement_count', 'dovish_count', 'hawkish_count', 
                  'dovish_phrases', 'hawkish_phrases', 'word_count']].copy()
output['date'] = output['date'].dt.strftime('%Y-%m-%d')
output.to_csv(OUTPUT_FILE, index=False)

print(f"    Saved: {OUTPUT_FILE}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*70)
print("[7] SUMMARY")
print("="*70)

print(f"\nScore distribution:")
print(grouped['keyword_dissent_direction'].describe())

print(f"\nDirection counts:")
grouped['direction_cat'] = pd.cut(grouped['keyword_dissent_direction'], 
                                  bins=[-11, -0.5, 0.5, 11], 
                                  labels=['Hawkish', 'Neutral', 'Dovish'])
print(grouped['direction_cat'].value_counts())

print(f"\n✅ Done! CSV saved to: {OUTPUT_FILE}")