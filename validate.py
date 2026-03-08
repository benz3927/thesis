#!/usr/bin/env python3
"""
VALIDATION: Compare GPT v2 vs v3 vs v5 vs v6 vs v6-placebo vs v7 vs v8 vs Claude vs Keywords
"""

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import glob
import os

print("="*70)
print("VALIDATION: GPT v2/v3/v5/v6/v6p/v7/v8 vs Claude vs Keywords")
print("="*70)

# ============================================================================
# LOAD SCORES
# ============================================================================

print("\n[1] Loading scores...")

gpt_v2 = pd.read_csv("data/cache/gpt_dissent_scores_v2.csv")
gpt_v2['date'] = pd.to_datetime(gpt_v2['date'])
gpt_v2 = gpt_v2.rename(columns={'gpt_dissent_score': 'gpt_v2_score'})

gpt_v3 = pd.read_csv("data/cache/gpt_dissent_scores_v3.csv")
gpt_v3['date'] = pd.to_datetime(gpt_v3['date'])
gpt_v3 = gpt_v3.rename(columns={'gpt_dissent_direction': 'gpt_v3_score'})

gpt_v5 = pd.read_csv("data/cache/gpt_dissent_scores_v5.csv")
gpt_v5['date'] = pd.to_datetime(gpt_v5['date'])
gpt_v5 = gpt_v5.rename(columns={'gpt_dissent_direction': 'gpt_v5_score'})

kw_v7 = pd.read_csv("data/cache/keyword_dissent_scores_v7.csv")
kw_v7['date'] = pd.to_datetime(kw_v7['date'])
kw_v7 = kw_v7.rename(columns={'keyword_dissent_direction': 'kw_v7_score'})

# Optional versions
has_v6 = os.path.exists("data/cache/gpt_dissent_scores_v6.csv")
has_v6p = os.path.exists("data/cache/gpt_dissent_scores_v6_placebo.csv")
has_v7 = os.path.exists("data/cache/gpt_dissent_scores_v7.csv")
has_v8 = os.path.exists("data/cache/gpt_dissent_scores_v8.csv")
has_claude = os.path.exists("data/cache/claude_dissent_scores_v8.csv")

# Merge base scores
scores = gpt_v2[['speaker', 'date', 'gpt_v2_score']].merge(
    gpt_v3[['speaker', 'date', 'gpt_v3_score']], on=['speaker', 'date'], how='outer'
).merge(
    gpt_v5[['speaker', 'date', 'gpt_v5_score']], on=['speaker', 'date'], how='outer'
).merge(
    kw_v7[['speaker', 'date', 'kw_v7_score']], on=['speaker', 'date'], how='outer'
)

if has_v6:
    gpt_v6 = pd.read_csv("data/cache/gpt_dissent_scores_v6.csv")
    gpt_v6['date'] = pd.to_datetime(gpt_v6['date'])
    gpt_v6 = gpt_v6.rename(columns={'gpt_dissent_direction': 'gpt_v6_score'})
    scores = scores.merge(gpt_v6[['speaker', 'date', 'gpt_v6_score']], on=['speaker', 'date'], how='outer')
    print("    ✓ Loaded GPT v6")

if has_v6p:
    gpt_v6p = pd.read_csv("data/cache/gpt_dissent_scores_v6_placebo.csv")
    gpt_v6p['date'] = pd.to_datetime(gpt_v6p['date'])
    gpt_v6p = gpt_v6p.rename(columns={'gpt_dissent_direction': 'gpt_v6p_score'})
    scores = scores.merge(gpt_v6p[['speaker', 'date', 'gpt_v6p_score']], on=['speaker', 'date'], how='outer')
    print("    ✓ Loaded GPT v6-placebo")

if has_v7:
    gpt_v7 = pd.read_csv("data/cache/gpt_dissent_scores_v7.csv")
    gpt_v7['date'] = pd.to_datetime(gpt_v7['date'])
    gpt_v7 = gpt_v7.rename(columns={'gpt_dissent_direction': 'gpt_v7_score'})
    scores = scores.merge(gpt_v7[['speaker', 'date', 'gpt_v7_score']], on=['speaker', 'date'], how='outer')
    print("    ✓ Loaded GPT v7")

if has_v8:
    gpt_v8 = pd.read_csv("data/cache/gpt_dissent_scores_v8.csv")
    gpt_v8['date'] = pd.to_datetime(gpt_v8['date'])
    gpt_v8 = gpt_v8.rename(columns={'gpt_dissent_direction': 'gpt_v8_score'})
    scores = scores.merge(gpt_v8[['speaker', 'date', 'gpt_v8_score']], on=['speaker', 'date'], how='outer')
    print("    ✓ Loaded GPT v8 (gpt-4o)")

if has_claude:
    claude_scores = pd.read_csv("data/cache/claude_dissent_scores_v8.csv")
    claude_scores['date'] = pd.to_datetime(claude_scores['date'])
    claude_scores = claude_scores.rename(columns={'claude_dissent_direction': 'claude_score'})
    scores = scores.merge(claude_scores[['speaker', 'date', 'claude_score']], on=['speaker', 'date'], how='outer')
    print("    ✓ Loaded Claude (sonnet, v8 prompt)")

print(f"    Total observations: {len(scores)}")

# ============================================================================
# LOAD DISSENT DATA
# ============================================================================

print("\n[2] Loading dissent data...")

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

print(f"    Dissent records: {len(dissent_df)}")
print(f"    Tighter: {(dissent_df['direction'] == 'tighter').sum()}")
print(f"    Easier: {(dissent_df['direction'] == 'easier').sum()}")
print(f"    Other: {(dissent_df['direction'] == 'other').sum()}")

# ============================================================================
# MATCH DISSENTS TO SCORES
# ============================================================================

print("\n[3] Matching dissents to scores...")

def match_dissent(row, dissent_df):
    speaker_upper = row['speaker'].upper()
    for _, d in dissent_df[dissent_df['date'] == row['date']].iterrows():
        if d['name'] in speaker_upper:
            return d['direction']
    return 'none'

scores['dissent_direction'] = scores.apply(lambda r: match_dissent(r, dissent_df), axis=1)
scores['dissented'] = (scores['dissent_direction'] != 'none').astype(int)

print(f"    Matched dissents: {scores['dissented'].sum()}")

# ============================================================================
# VALIDATION 1: INTENSITY (can we detect ANY dissent?)
# ============================================================================

print("\n" + "="*70)
print("[4] VALIDATION 1: Detecting ANY dissent (intensity)")
print("="*70)

dissenters_names = dissent_df['name'].unique()
def is_potential_dissenter(speaker):
    speaker_upper = speaker.upper()
    for name in dissenters_names:
        if name in speaker_upper:
            return True
    return False

filtered = scores[scores['speaker'].apply(is_potential_dissenter)].copy()
filtered = filtered.dropna(subset=['gpt_v2_score'])

print(f"\n    Filtered to potential dissenters: {len(filtered)} obs")

auc_gpt_v2 = roc_auc_score(filtered['dissented'], filtered['gpt_v2_score'])
print(f"\n    GPT v2 (intensity 0-10):  ROC-AUC = {auc_gpt_v2:.4f}")

filtered_v3 = filtered.dropna(subset=['gpt_v3_score'])
auc_gpt_v3 = roc_auc_score(filtered_v3['dissented'], filtered_v3['gpt_v3_score'].abs())
print(f"    GPT v3 (|direction|):     ROC-AUC = {auc_gpt_v3:.4f}")

filtered_v5 = filtered.dropna(subset=['gpt_v5_score'])
auc_gpt_v5 = roc_auc_score(filtered_v5['dissented'], filtered_v5['gpt_v5_score'].abs())
print(f"    GPT v5 (|direction|):     ROC-AUC = {auc_gpt_v5:.4f}")

auc_v6 = None
auc_v6p = None
auc_v7 = None
auc_v8 = None
auc_claude = None

if has_v6:
    filtered_v6 = filtered.dropna(subset=['gpt_v6_score'])
    auc_v6 = roc_auc_score(filtered_v6['dissented'], filtered_v6['gpt_v6_score'].abs())
    print(f"    GPT v6 (|direction|):     ROC-AUC = {auc_v6:.4f}")

if has_v6p:
    filtered_v6p = filtered.dropna(subset=['gpt_v6p_score'])
    auc_v6p = roc_auc_score(filtered_v6p['dissented'], filtered_v6p['gpt_v6p_score'].abs())
    print(f"    GPT v6-placebo (|dir|):   ROC-AUC = {auc_v6p:.4f}")

if has_v7:
    filtered_v7 = filtered.dropna(subset=['gpt_v7_score'])
    auc_v7 = roc_auc_score(filtered_v7['dissented'], filtered_v7['gpt_v7_score'].abs())
    print(f"    GPT v7 (|direction|):     ROC-AUC = {auc_v7:.4f}")

if has_v8:
    filtered_v8 = filtered.dropna(subset=['gpt_v8_score'])
    auc_v8 = roc_auc_score(filtered_v8['dissented'], filtered_v8['gpt_v8_score'].abs())
    print(f"    GPT v8/4o (|direction|):  ROC-AUC = {auc_v8:.4f}")

if has_claude:
    filtered_claude = filtered.dropna(subset=['claude_score'])
    auc_claude = roc_auc_score(filtered_claude['dissented'], filtered_claude['claude_score'].abs())
    print(f"    Claude (|direction|):     ROC-AUC = {auc_claude:.4f}")

filtered_kw = filtered.dropna(subset=['kw_v7_score'])
auc_kw = roc_auc_score(filtered_kw['dissented'], filtered_kw['kw_v7_score'].abs())
print(f"    Keywords v7 (|direction|): ROC-AUC = {auc_kw:.4f}")

# ============================================================================
# VALIDATION 2: DIRECTION
# ============================================================================

print("\n" + "="*70)
print("[5] VALIDATION 2: Detecting correct DIRECTION")
print("="*70)

directional = filtered[filtered['dissent_direction'].isin(['tighter', 'easier'])].copy()
directional = directional.dropna(subset=['gpt_v3_score', 'gpt_v5_score', 'kw_v7_score'])

print(f"\n    Directional dissents: {len(directional)} obs")

def direction_accuracy(col, df):
    correct = (
        ((df['dissent_direction'] == 'tighter') & (df[col] < 0)) |
        ((df['dissent_direction'] == 'easier') & (df[col] > 0))
    )
    return correct.sum(), len(df), correct.mean()

n3, t3, acc3 = direction_accuracy('gpt_v3_score', directional)
print(f"\n    GPT v3 direction accuracy:     {n3}/{t3} = {acc3:.1%}")

n5, t5, acc5 = direction_accuracy('gpt_v5_score', directional)
print(f"    GPT v5 direction accuracy:     {n5}/{t5} = {acc5:.1%}")

acc6 = None
acc6p = None
acc7 = None
acc8 = None
acc_claude = None

if has_v6:
    dir_v6 = directional.dropna(subset=['gpt_v6_score'])
    n6, t6, acc6 = direction_accuracy('gpt_v6_score', dir_v6)
    print(f"    GPT v6 direction accuracy:     {n6}/{t6} = {acc6:.1%}")

if has_v6p:
    dir_v6p = directional.dropna(subset=['gpt_v6p_score'])
    n6p, t6p, acc6p = direction_accuracy('gpt_v6p_score', dir_v6p)
    print(f"    GPT v6-placebo direction:      {n6p}/{t6p} = {acc6p:.1%}")

if has_v7:
    dir_v7 = directional.dropna(subset=['gpt_v7_score'])
    n7, t7, acc7 = direction_accuracy('gpt_v7_score', dir_v7)
    print(f"    GPT v7 direction accuracy:     {n7}/{t7} = {acc7:.1%}")

if has_v8:
    dir_v8 = directional.dropna(subset=['gpt_v8_score'])
    n8, t8, acc8 = direction_accuracy('gpt_v8_score', dir_v8)
    print(f"    GPT v8/4o direction accuracy:  {n8}/{t8} = {acc8:.1%}")

if has_claude:
    dir_claude = directional.dropna(subset=['claude_score'])
    nc, tc, acc_claude = direction_accuracy('claude_score', dir_claude)
    print(f"    Claude direction accuracy:     {nc}/{tc} = {acc_claude:.1%}")

nk, tk, acck = direction_accuracy('kw_v7_score', directional)
print(f"    Keywords v7 direction accuracy: {nk}/{tk} = {acck:.1%}")

# ============================================================================
# VALIDATION 3: WITHIN-PERSON
# ============================================================================

print("\n" + "="*70)
print("[6] VALIDATION 3: Within-person comparison")
print("="*70)

dissenter_counts = dissent_df['name'].value_counts()
top_dissenters = dissenter_counts[dissenter_counts >= 3].index

header = f"{'Name':<15} {'GPT_v3':>10} {'GPT_v5':>10}"
subheader = f"{'':15} {'D vs N':>10} {'D vs N':>10}"
if has_v7:
    header += f" {'GPT_v7':>10}"
    subheader += f" {'D vs N':>10}"
if has_v8:
    header += f" {'v8/4o':>10}"
    subheader += f" {'D vs N':>10}"
if has_claude:
    header += f" {'Claude':>10}"
    subheader += f" {'D vs N':>10}"
header += f" {'KW_v7':>10}"
subheader += f" {'D vs N':>10}"

print(f"\n{header}")
print(subheader)
print("-"*len(header))

wp = {'v3': 0, 'v5': 0, 'v7': 0, 'v8': 0, 'claude': 0, 'kw': 0}
total = 0
failed = {'v3': [], 'v5': [], 'v7': [], 'v8': [], 'claude': []}

for name in top_dissenters[:15]:
    matches = scores[scores['speaker'].str.upper().str.contains(name)]
    if len(matches) < 3:
        continue

    dissent_dates = set(dissent_df[dissent_df['name'] == name]['date'])
    d_mtgs = matches[matches['date'].isin(dissent_dates)]
    n_mtgs = matches[~matches['date'].isin(dissent_dates)]

    if len(d_mtgs) > 0 and len(n_mtgs) > 0:
        total += 1

        v3_diff = d_mtgs['gpt_v3_score'].abs().mean() - n_mtgs['gpt_v3_score'].abs().mean()
        v5_diff = d_mtgs['gpt_v5_score'].abs().mean() - n_mtgs['gpt_v5_score'].abs().mean()

        row_str = f"{name:<15} {v3_diff:+.2f} {'✓' if v3_diff > 0 else '✗':>3} {v5_diff:+.2f} {'✓' if v5_diff > 0 else '✗':>3}"

        if v3_diff > 0: wp['v3'] += 1
        else: failed['v3'].append({'name': name, 'diff': v3_diff})
        if v5_diff > 0: wp['v5'] += 1
        else: failed['v5'].append({'name': name, 'diff': v5_diff})

        if has_v7:
            v7_diff = d_mtgs['gpt_v7_score'].abs().mean() - n_mtgs['gpt_v7_score'].abs().mean()
            row_str += f" {v7_diff:+.2f} {'✓' if v7_diff > 0 else '✗':>3}"
            if v7_diff > 0: wp['v7'] += 1
            else: failed['v7'].append({'name': name, 'diff': v7_diff})

        if has_v8:
            v8_diff = d_mtgs['gpt_v8_score'].abs().mean() - n_mtgs['gpt_v8_score'].abs().mean()
            row_str += f" {v8_diff:+.2f} {'✓' if v8_diff > 0 else '✗':>3}"
            if v8_diff > 0: wp['v8'] += 1
            else: failed['v8'].append({'name': name, 'diff': v8_diff})

        if has_claude:
            cl_d = d_mtgs['claude_score'].dropna()
            cl_n = n_mtgs['claude_score'].dropna()
            if len(cl_d) > 0 and len(cl_n) > 0:
                cl_diff = cl_d.abs().mean() - cl_n.abs().mean()
                row_str += f" {cl_diff:+.2f} {'✓' if cl_diff > 0 else '✗':>3}"
                if cl_diff > 0: wp['claude'] += 1
                else: failed['claude'].append({'name': name, 'diff': cl_diff})
            else:
                row_str += f" {'N/A':>10}"

        kw_diff = d_mtgs['kw_v7_score'].abs().mean() - n_mtgs['kw_v7_score'].abs().mean()
        row_str += f" {kw_diff:+.2f} {'✓' if kw_diff > 0 else '✗':>3}"
        if kw_diff > 0: wp['kw'] += 1

        print(row_str)

print("-"*len(header))
wp_str = f"{'Within-person':<15} {wp['v3']}/{total}     {wp['v5']}/{total}    "
if has_v7: wp_str += f" {wp['v7']}/{total}    "
if has_v8: wp_str += f" {wp['v8']}/{total}    "
if has_claude: wp_str += f" {wp['claude']}/{total}    "
wp_str += f" {wp['kw']}/{total}"
print(wp_str)

for version in ['v3', 'v5', 'v7', 'v8', 'claude']:
    if failed[version]:
        print(f"\n  {version} failures: {', '.join([f['name'] for f in failed[version]])}")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*70)
print("[7] SUMMARY")
print("="*70)

print(f"\n  {'Metric':<25} {'v3':>8} {'v5':>8}", end="")
if has_v7: print(f" {'v7':>8}", end="")
if has_v8: print(f" {'v8/4o':>8}", end="")
if has_claude: print(f" {'Claude':>8}", end="")
print(f" {'KW v7':>8}")

print(f"  {'-'*25} {'-'*8} {'-'*8}", end="")
if has_v7: print(f" {'-'*8}", end="")
if has_v8: print(f" {'-'*8}", end="")
if has_claude: print(f" {'-'*8}", end="")
print(f" {'-'*8}")

print(f"  {'ROC-AUC':<25} {auc_gpt_v3:.3f}    {auc_gpt_v5:.3f}   ", end="")
if has_v7: print(f" {auc_v7:.3f}   ", end="")
if has_v8: print(f" {auc_v8:.3f}   ", end="")
if has_claude: print(f" {auc_claude:.3f}   ", end="")
print(f" {auc_kw:.3f}")

print(f"  {'Direction accuracy':<25} {acc3:.1%}    {acc5:.1%}   ", end="")
if has_v7: print(f" {acc7:.1%}   ", end="")
if has_v8: print(f" {acc8:.1%}   ", end="")
if has_claude: print(f" {acc_claude:.1%}   ", end="")
print(f" {acck:.1%}")

print(f"  {'Within-person':<25} {wp['v3']}/{total}      {wp['v5']}/{total}     ", end="")
if has_v7: print(f" {wp['v7']}/{total}     ", end="")
if has_v8: print(f" {wp['v8']}/{total}     ", end="")
if has_claude: print(f" {wp['claude']}/{total}     ", end="")
print(f" {wp['kw']}/{total}")

# ============================================================================
# PLOT ROC CURVES (CLEAN - for thesis)
# ============================================================================

print("\n[8] Plotting clean ROC (thesis version)...")

fig, ax = plt.subplots(figsize=(8, 6))

fpr, tpr, _ = roc_curve(filtered_v3['dissented'], filtered_v3['gpt_v3_score'].abs())
ax.plot(fpr, tpr, label=f'GPT v3 – baseline (AUC={auc_gpt_v3:.3f})',
        linewidth=2, color='#F59E0B')

if has_v7:
    fpr, tpr, _ = roc_curve(filtered_v7['dissented'], filtered_v7['gpt_v7_score'].abs())
    ax.plot(fpr, tpr, label=f'GPT v7 – few-shot (AUC={auc_v7:.3f})',
            linewidth=2, color='#3B82F6')

if has_v8:
    fpr, tpr, _ = roc_curve(filtered_v8['dissented'], filtered_v8['gpt_v8_score'].abs())
    ax.plot(fpr, tpr, label=f'GPT v8 – GPT-4o (AUC={auc_v8:.3f})',
            linewidth=2.5, color='#10B981')

if has_claude:
    fpr, tpr, _ = roc_curve(filtered_claude['dissented'], filtered_claude['claude_score'].abs())
    ax.plot(fpr, tpr, label=f'Claude Sonnet (AUC={auc_claude:.3f})',
            linewidth=2, color='#8B5CF6', linestyle='--')

fpr, tpr, _ = roc_curve(filtered_kw['dissented'], filtered_kw['kw_v7_score'].abs())
ax.plot(fpr, tpr, label=f'Keyword baseline (AUC={auc_kw:.3f})',
        linewidth=2, color='#EF4444', linestyle=':')

ax.plot([0, 1], [0, 1], 'k--', label='Random (AUC=0.500)', alpha=0.4, linewidth=1)

ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves: Detecting FOMC Dissent', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig('plots/roc_validation_clean.png', dpi=200)
print("    Saved: plots/roc_validation_clean.png")

print("\n" + "="*70)
print("Done!")
print("="*70)