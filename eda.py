#!/usr/bin/env python3
"""
DESCRIPTIVE STATISTICS & VISUALIZATIONS
========================================
- Score distributions by speaker
- Bank presidents vs Board of Governors
- Dissents over time
- Unemployment gap vs speech scores
- Score heatmaps by speaker x year (CLEAN - no cell annotations)
- Timeline strips (CLEAN - circles only, era labels at top)
- GPT vs Claude comparison
- Fed district map of dissent
- Summary statistics for appendix (Table A1 & A2)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import os
import warnings
warnings.filterwarnings('ignore')

CACHE_DIR = 'data/cache'
PLOTS_DIR = 'plots/descriptive'
os.makedirs(PLOTS_DIR, exist_ok=True)

print("=" * 70)
print("DESCRIPTIVE STATISTICS & VISUALIZATIONS")
print("=" * 70)

# ============================================================================
# LOAD DATA
# ============================================================================

print("\n[1] Loading data...")

v8 = pd.read_csv(f'{CACHE_DIR}/gpt_dissent_scores_v8.csv')
v8['date'] = pd.to_datetime(v8['date'])

# Load Claude if available
has_claude = os.path.exists(f'{CACHE_DIR}/claude_dissent_scores_v8.csv')
if has_claude:
    claude = pd.read_csv(f'{CACHE_DIR}/claude_dissent_scores_v8.csv')
    claude['date'] = pd.to_datetime(claude['date'])
    v8 = v8.merge(
        claude[['speaker', 'date', 'claude_dissent_direction']],
        on=['speaker', 'date'], how='left'
    )
    print("    Loaded Claude scores")

# Load votes
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

def match_dissent(row, dissent_df):
    speaker_upper = row['speaker'].upper()
    for _, d in dissent_df[dissent_df['date'] == row['date']].iterrows():
        if d['name'] in speaker_upper:
            return d['direction']
    return 'none'

v8['dissent_direction'] = v8.apply(lambda r: match_dissent(r, dissent_df), axis=1)
v8['dissented'] = (v8['dissent_direction'] != 'none').astype(int)
v8['is_chair'] = v8['speaker'].str.upper().str.contains('CHAIR', na=False)
v8['is_bank_president'] = v8['district'].notna() & ~v8['is_chair']
v8['is_board'] = v8['district'].isna() & ~v8['is_chair']

# Load unemployment
unemp = pd.read_csv(f'{CACHE_DIR}/regional_unemployment_all.csv')
unemp['date'] = pd.to_datetime(unemp['date'])
unemp['year_month'] = unemp['date'].dt.to_period('M')
nat_unemp = unemp.groupby('year_month')['unemployment_rate'].mean().rename('nat_unemp')
unemp = unemp.merge(nat_unemp, on='year_month')
unemp['unemployment_gap'] = unemp['unemployment_rate'] - unemp['nat_unemp']

v8['year_month'] = v8['date'].dt.to_period('M')
v8 = v8.merge(
    unemp[['year_month', 'district', 'unemployment_gap']],
    on=['year_month', 'district'], how='left'
)

print(f"    Total observations: {len(v8)}")
print(f"    Bank presidents: {v8['is_bank_president'].sum()}")
print(f"    Board members: {v8['is_board'].sum()}")
print(f"    Chairs: {v8['is_chair'].sum()}")

# ============================================================================
# HELPER: consistent hawk/dove colormap
# ============================================================================

HAWK_DOVE_CMAP = mcolors.LinearSegmentedColormap.from_list('hawk_dove',
    ['#c0392b', '#e74c3c', '#f5b7b1', '#fdfefe', '#aed6f1', '#3498db', '#2471a3'], N=256)

def get_last_name(speaker):
    """Extract clean last name from FOMC speaker string."""
    DISTRICT_MAPPING_NAMES = [
        'syron', 'minehan', 'rosengren', 'mcdonough', 'geithner', 'dudley',
        'boehne', 'santomero', 'plosser', 'harker', 'jordan', 'pianalto',
        'mester', 'broaddus', 'lacker', 'barkin', 'forrestal', 'guynn',
        'lockhart', 'bostic', 'keehn', 'moskow', 'evans', 'melzer', 'poole',
        'bullard', 'stern', 'kocherlakota', 'kashkari', 'hoenig', 'george',
        'mcteer', 'fisher', 'kaplan', 'parry', 'yellen', 'williams', 'daly',
    ]
    for name in DISTRICT_MAPPING_NAMES:
        if name in str(speaker).lower():
            return name.capitalize()
    return speaker.split()[-1].title() if speaker else speaker

# ============================================================================
# [2] OVERALL SCORE DISTRIBUTION
# ============================================================================

print("\n[2] Overall score distribution...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].hist(v8['gpt_dissent_direction'], bins=21, range=(-10.5, 10.5),
             color='steelblue', edgecolor='white', alpha=0.8)
axes[0].set_title('All Speakers (incl. Chairs)', fontsize=13, fontweight='bold')
axes[0].set_xlabel('Policy Stance Score')
axes[0].set_ylabel('Count')
axes[0].axvline(0, color='red', linestyle='--', alpha=0.5)

bp = v8[v8['is_bank_president']]
axes[1].hist(bp['gpt_dissent_direction'], bins=21, range=(-10.5, 10.5),
             color='#2ecc71', edgecolor='white', alpha=0.8)
axes[1].set_title('Bank Presidents Only', fontsize=13, fontweight='bold')
axes[1].set_xlabel('Policy Stance Score')
axes[1].axvline(0, color='red', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/score_distributions.png', dpi=200)
print(f"    Saved: {PLOTS_DIR}/score_distributions.png")

# ============================================================================
# [3] BANK PRESIDENTS vs BOARD: Summary stats
# ============================================================================

print("\n[3] Bank presidents vs Board of Governors...")

print(f"\n    {'':>25} {'Bank Pres':>12} {'Chair':>12}")
print(f"    {'-'*49}")
for stat_name, stat_func in [
    ('N', lambda df: len(df)),
    ('Mean score', lambda df: df['gpt_dissent_direction'].mean()),
    ('Std dev', lambda df: df['gpt_dissent_direction'].std()),
    ('Mean |score|', lambda df: df['gpt_dissent_direction'].abs().mean()),
    ('% scored 0', lambda df: (df['gpt_dissent_direction'] == 0).mean() * 100),
    ('% hawkish (<0)', lambda df: (df['gpt_dissent_direction'] < 0).mean() * 100),
    ('% dovish (>0)', lambda df: (df['gpt_dissent_direction'] > 0).mean() * 100),
    ('Dissent rate', lambda df: df['dissented'].mean() * 100),
]:
    bp_val = stat_func(v8[v8['is_bank_president']])
    ch_val = stat_func(v8[v8['is_chair']])
    if isinstance(bp_val, int):
        print(f"    {stat_name:>25} {bp_val:>12,} {ch_val:>12,}")
    else:
        print(f"    {stat_name:>25} {bp_val:>12.2f} {ch_val:>12.2f}")

# ============================================================================
# [4] TOP SPEAKERS BY MEAN SCORE (bank presidents only)
# ============================================================================

print("\n[4] Speaker-level statistics (bank presidents, >=10 meetings)...")

speaker_stats = (
    v8[v8['is_bank_president']]
    .groupby(['speaker', 'district'])
    .agg(
        n_meetings=('gpt_dissent_direction', 'count'),
        mean_score=('gpt_dissent_direction', 'mean'),
        std_score=('gpt_dissent_direction', 'std'),
        mean_abs=('gpt_dissent_direction', lambda x: x.abs().mean()),
        pct_hawk=('gpt_dissent_direction', lambda x: (x < 0).mean() * 100),
        pct_dove=('gpt_dissent_direction', lambda x: (x > 0).mean() * 100),
        pct_zero=('gpt_dissent_direction', lambda x: (x == 0).mean() * 100),
        n_dissents=('dissented', 'sum'),
    )
    .reset_index()
)

speaker_stats = speaker_stats[speaker_stats['n_meetings'] >= 10].sort_values('mean_score')

print(f"\n    {'Speaker':<22} {'District':<15} {'N':>4} {'Mean':>6} {'|Mean|':>7} {'%Hawk':>6} {'%Dove':>6} {'%Zero':>6} {'Diss':>5}")
print(f"    {'-'*90}")

print("\n    === MOST HAWKISH ===")
for _, row in speaker_stats.head(10).iterrows():
    name = get_last_name(row['speaker'])
    print(f"    {name:<22} {row['district']:<15} {row['n_meetings']:>4.0f} {row['mean_score']:>+6.2f} {row['mean_abs']:>7.2f} {row['pct_hawk']:>5.1f}% {row['pct_dove']:>5.1f}% {row['pct_zero']:>5.1f}% {row['n_dissents']:>5.0f}")

print("\n    === MOST DOVISH ===")
for _, row in speaker_stats.tail(10).iloc[::-1].iterrows():
    name = get_last_name(row['speaker'])
    print(f"    {name:<22} {row['district']:<15} {row['n_meetings']:>4.0f} {row['mean_score']:>+6.2f} {row['mean_abs']:>7.2f} {row['pct_hawk']:>5.1f}% {row['pct_dove']:>5.1f}% {row['pct_zero']:>5.1f}% {row['n_dissents']:>5.0f}")

# ============================================================================
# [5] SPEAKER MEAN SCORE BAR CHART
# ============================================================================

print("\n[5] Speaker mean score bar chart...")

fig, ax = plt.subplots(figsize=(14, 8))

plot_data = speaker_stats.sort_values('mean_score')
colors = ['#e74c3c' if x < 0 else '#3498db' for x in plot_data['mean_score']]
names = [get_last_name(row['speaker']) + f" ({row['district'][:3]})"
         for _, row in plot_data.iterrows()]

ax.barh(range(len(plot_data)), plot_data['mean_score'], color=colors, alpha=0.8)
ax.set_yticks(range(len(plot_data)))
ax.set_yticklabels(names, fontsize=8)
ax.axvline(0, color='black', linewidth=0.8)
ax.set_xlabel('Mean Policy Stance Score', fontsize=12)
ax.set_title('Mean Policy Stance by Bank President (GPT-4o v8)\nNegative = Hawkish, Positive = Dovish',
             fontsize=13, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/speaker_mean_scores.png', dpi=200)
print(f"    Saved: {PLOTS_DIR}/speaker_mean_scores.png")

# ============================================================================
# [6] DISSENTS OVER TIME
# ============================================================================

print("\n[6] Dissents over time...")

fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

dissent_by_year = dissent_df.groupby(dissent_df['date'].dt.year).size()
years = range(1994, 2021)
dissent_counts = [dissent_by_year.get(y, 0) for y in years]

colors_yr = ['#95a5a6' if y < 2006 else '#3498db' for y in years]

axes[0].bar(years, dissent_counts, color=colors_yr, edgecolor='white', alpha=0.8)
axes[0].set_ylabel('Number of Dissent Votes', fontsize=12)
axes[0].set_title('Panel A: Formal Dissent Votes per Year', fontsize=13, fontweight='bold')
axes[0].axvline(2005.5, color='red', linestyle='--', alpha=0.6, label='Greenspan \u2192 Post-Greenspan')
axes[0].legend(fontsize=10)

bp_by_year = (
    v8[v8['is_bank_president']]
    .groupby(v8[v8['is_bank_president']]['date'].dt.year)
    .agg(
        mean_abs=('gpt_dissent_direction', lambda x: x.abs().mean()),
        std_score=('gpt_dissent_direction', 'std'),
    )
)

axes[1].plot(bp_by_year.index, bp_by_year['mean_abs'], 'o-', color='steelblue',
             linewidth=2, markersize=5, label='Mean |score|')
axes[1].fill_between(bp_by_year.index,
                     bp_by_year['mean_abs'] - bp_by_year['std_score'] * 0.3,
                     bp_by_year['mean_abs'] + bp_by_year['std_score'] * 0.3,
                     alpha=0.2, color='steelblue')
axes[1].set_ylabel('Mean |Policy Stance Score|', fontsize=12)
axes[1].set_xlabel('Year', fontsize=12)
axes[1].set_title('Panel B: Average Speech Intensity (Bank Presidents)', fontsize=13, fontweight='bold')
axes[1].axvline(2005.5, color='red', linestyle='--', alpha=0.6)
axes[1].legend(fontsize=10)

plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/dissents_over_time.png', dpi=200)
print(f"    Saved: {PLOTS_DIR}/dissents_over_time.png")

# ============================================================================
# [7] UNEMPLOYMENT GAP vs SPEECH SCORE (scatter)
# ============================================================================

print("\n[7] Unemployment gap vs speech score...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

bp_unemp = v8[(v8['is_bank_president']) & (v8['unemployment_gap'].notna())].copy()
bp_unemp['era'] = np.where(bp_unemp['date'] < pd.Timestamp('2006-02-01'), 'Greenspan', 'Post-Greenspan')

for i, (era, color) in enumerate([('Greenspan', '#95a5a6'), ('Post-Greenspan', '#3498db')]):
    subset = bp_unemp[bp_unemp['era'] == era]
    axes[i].scatter(subset['unemployment_gap'], subset['gpt_dissent_direction'],
                   alpha=0.15, s=15, color=color)

    z = np.polyfit(subset['unemployment_gap'], subset['gpt_dissent_direction'], 1)
    p = np.poly1d(z)
    x_range = np.linspace(subset['unemployment_gap'].min(), subset['unemployment_gap'].max(), 100)
    axes[i].plot(x_range, p(x_range), color='red', linewidth=2)

    r = subset['unemployment_gap'].corr(subset['gpt_dissent_direction'])
    axes[i].set_title(f'{era} Era (r={r:.3f})', fontsize=13, fontweight='bold')
    axes[i].set_xlabel('District Unemployment Gap (pp)', fontsize=11)
    axes[i].set_ylabel('Policy Stance Score', fontsize=11)
    axes[i].axhline(0, color='gray', linestyle=':', alpha=0.5)
    axes[i].axvline(0, color='gray', linestyle=':', alpha=0.5)
    axes[i].set_ylim(-10.5, 10.5)
    axes[i].grid(alpha=0.2)

plt.suptitle('Regional Unemployment Gap vs. Speech Score (Bank Presidents)',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/unemp_gap_vs_speech.png', dpi=200, bbox_inches='tight')
print(f"    Saved: {PLOTS_DIR}/unemp_gap_vs_speech.png")

# ============================================================================
# [8] HEATMAP: Speaker x Year (CLEAN - no cell annotations)
# ============================================================================

print("\n[8] Speaker-year heatmap (clean)...")

bp_all = v8[v8['is_bank_president']].copy()
bp_all['year'] = bp_all['date'].dt.year
bp_all['last_name'] = bp_all['speaker'].apply(get_last_name)

pivot = bp_all.groupby(['last_name', 'year'])['gpt_dissent_direction'].mean().unstack(fill_value=np.nan)

# Sort alphabetically
speaker_means = bp_all.groupby('last_name')['gpt_dissent_direction'].mean().sort_index()
# Keep only speakers with >= 10 meetings
speaker_counts = bp_all.groupby('last_name').size()
keep = speaker_counts[speaker_counts >= 10].index
speaker_means = speaker_means[speaker_means.index.isin(keep)]
pivot = pivot.reindex(speaker_means.index)

fig, ax = plt.subplots(figsize=(16, 10))

masked = np.ma.masked_invalid(pivot.values)

im = ax.pcolormesh(
    np.arange(pivot.columns.min(), pivot.columns.max() + 2) - 0.5,
    np.arange(len(pivot) + 1) - 0.5,
    masked,
    cmap=HAWK_DOVE_CMAP, vmin=-7, vmax=7,
    edgecolors='white', linewidth=1.5,
)

ax.set_yticks(range(len(pivot)))
ax.set_yticklabels(pivot.index, fontsize=11, fontfamily='serif')

years_list = sorted(pivot.columns)
ax.set_xticks(years_list)
ax.set_xticklabels(years_list, fontsize=10, rotation=45, ha='right')

# Greenspan / Post-Greenspan divider
ax.axvline(x=2005.5, color='#2c3e50', linewidth=2, linestyle='--', alpha=0.7)
ax.text(2004.5, -0.8, 'Greenspan', fontsize=10, ha='right',
        fontfamily='serif', fontstyle='italic', color='#555')
ax.text(2006.5, -0.8, 'Post-Greenspan', fontsize=10, ha='left',
        fontfamily='serif', fontstyle='italic', color='#555')

cbar = fig.colorbar(im, ax=ax, shrink=0.6, aspect=25, pad=0.02)
cbar.set_label('Mean Policy Stance Score', fontsize=11, fontfamily='serif')
cbar.set_ticks([-6, -3, 0, 3, 6])
cbar.set_ticklabels(['Hawkish\n(\u22126)', '\u22123', 'Neutral\n(0)', '+3', 'Dovish\n(+6)'])
cbar.ax.tick_params(labelsize=10)

ax.set_title('Policy Stance by Speaker and Year',
             fontsize=16, fontweight='bold', fontfamily='serif', pad=20)

ax.invert_yaxis()
ax.set_xlim(pivot.columns.min() - 0.5, pivot.columns.max() + 0.5)

plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/speaker_year_heatmap.png', dpi=200, bbox_inches='tight',
            facecolor='white')
print(f"    Saved: {PLOTS_DIR}/speaker_year_heatmap.png")

# ============================================================================
# [9] DISSENT vs NON-DISSENT BOX PLOT
# ============================================================================

print("\n[9] Score on dissent vs non-dissent days...")

fig, ax = plt.subplots(figsize=(10, 6))

dissenters = v8[v8['dissented'] == 1]
non_d_same = v8[(v8['dissented'] == 0) & (v8['speaker'].isin(dissenters['speaker'].unique()))]

data_box = [
    non_d_same['gpt_dissent_direction'].abs().dropna(),
    dissenters['gpt_dissent_direction'].abs().dropna(),
]

bp_plot = ax.boxplot(data_box, labels=['Non-Dissent\nMeetings', 'Dissent\nMeetings'],
                     patch_artist=True, widths=0.5)

bp_plot['boxes'][0].set_facecolor('#3498db')
bp_plot['boxes'][0].set_alpha(0.6)
bp_plot['boxes'][1].set_facecolor('#e74c3c')
bp_plot['boxes'][1].set_alpha(0.6)

ax.set_ylabel('|Policy Stance Score|', fontsize=12)
ax.set_title('Speech Intensity: Dissent vs Non-Dissent Meetings\n(Same speakers who dissented at least once)',
             fontsize=13, fontweight='bold')

for i, d in enumerate(data_box):
    ax.scatter(i + 1, d.mean(), color='black', s=100, zorder=5, marker='D')
    ax.annotate(f'Mean: {d.mean():.2f}', (i + 1.15, d.mean()), fontsize=10)

ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/dissent_vs_nondissent_box.png', dpi=200)
print(f"    Saved: {PLOTS_DIR}/dissent_vs_nondissent_box.png")

# ============================================================================
# [10] DISTRICT-LEVEL SUMMARY
# ============================================================================

print("\n[10] District-level summary...")

district_stats = (
    v8[v8['is_bank_president']]
    .groupby('district')
    .agg(
        n=('gpt_dissent_direction', 'count'),
        mean_score=('gpt_dissent_direction', 'mean'),
        mean_abs=('gpt_dissent_direction', lambda x: x.abs().mean()),
        n_dissents=('dissented', 'sum'),
        n_speakers=('speaker', 'nunique'),
    )
    .sort_values('mean_score')
)

print(f"\n    {'District':<18} {'N':>5} {'Speakers':>9} {'Mean':>7} {'|Mean|':>7} {'Dissents':>9}")
print(f"    {'-'*60}")
for dist, row in district_stats.iterrows():
    print(f"    {dist:<18} {row['n']:>5.0f} {row['n_speakers']:>9.0f} {row['mean_score']:>+7.2f} {row['mean_abs']:>7.2f} {row['n_dissents']:>9.0f}")

# ============================================================================
# [11] TIMELINE STRIPS (CLEAN - circles, era labels at top)
# ============================================================================

print("\n[11] Timeline strips (clean)...")

bp_tl = v8[v8['is_bank_president']].copy()
bp_tl['last_name'] = bp_tl['speaker'].apply(get_last_name)

# Select speakers: dissenters with >= 15 meetings
dissent_speakers = set(bp_tl[bp_tl['dissented'] == 1]['last_name'].unique())
meeting_counts = bp_tl.groupby('last_name').size()
enough_meetings = set(meeting_counts[meeting_counts >= 15].index)
show_speakers = sorted(dissent_speakers & enough_meetings)

# Sort alphabetically
speaker_order = (bp_tl[bp_tl['last_name'].isin(show_speakers)]
                 .groupby('last_name')['gpt_dissent_direction'].mean()
                 .sort_index())
show_speakers = speaker_order.index.tolist()

cmap = HAWK_DOVE_CMAP
norm = mcolors.Normalize(vmin=-8, vmax=8)

fig, ax = plt.subplots(figsize=(18, 10))

for i, speaker in enumerate(show_speakers):
    sdata = bp_tl[bp_tl['last_name'] == speaker].copy()

    # Non-dissent: small faded dots
    non_dissent = sdata[sdata['dissented'] == 0]
    colors_nd = [cmap(norm(s)) for s in non_dissent['gpt_dissent_direction']]
    ax.scatter(non_dissent['date'], [i] * len(non_dissent),
              c=colors_nd, s=25, alpha=0.5, edgecolors='none', zorder=2)

    # Dissent: larger dots with black border
    dissent = sdata[sdata['dissented'] == 1]
    if len(dissent) > 0:
        colors_d = [cmap(norm(s)) for s in dissent['gpt_dissent_direction']]
        ax.scatter(dissent['date'], [i] * len(dissent),
                  c=colors_d, s=180, alpha=0.9, edgecolors='black',
                  linewidth=1.5, zorder=3)

# Greenspan / Post-Greenspan vertical line
ax.axvline(x=pd.Timestamp('2006-02-01'), color='#2c3e50', linewidth=1.5,
           linestyle='--', alpha=0.5, zorder=1)

# Era labels at TOP of chart
ax.text(pd.Timestamp('2000-01-01'), -1.8, 'Greenspan',
        fontsize=11, ha='center', fontfamily='serif', fontstyle='italic',
        color='#555', fontweight='bold')
ax.text(pd.Timestamp('2013-06-01'), -1.8, 'Post-Greenspan',
        fontsize=11, ha='center', fontfamily='serif', fontstyle='italic',
        color='#555', fontweight='bold')

# GFC shading
ax.axvspan(pd.Timestamp('2007-12-01'), pd.Timestamp('2009-06-01'),
           alpha=0.08, color='gray', zorder=0)
ax.text(pd.Timestamp('2008-09-01'), -1.0, 'GFC', fontsize=9, ha='center',
        color='#888', fontfamily='serif', fontstyle='italic')

# Y-axis
ax.set_yticks(range(len(show_speakers)))
ax.set_yticklabels(show_speakers, fontsize=11, fontfamily='serif')

# X-axis
ax.xaxis.set_major_locator(mdates.YearLocator(4))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
ax.tick_params(axis='x', labelsize=11, pad=8)

# Subtle grid
ax.set_axisbelow(True)
ax.grid(axis='x', alpha=0.15)
for i in range(len(show_speakers)):
    ax.axhline(y=i, color='#eee', linewidth=0.5, zorder=0)

# Colorbar
sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, shrink=0.5, aspect=20, pad=0.02)
cbar.set_ticks([-6, -3, 0, 3, 6])
cbar.set_ticklabels(['Hawkish', '\u22123', 'Neutral', '+3', 'Dovish'])
cbar.ax.tick_params(labelsize=10)

# Legend
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#999',
           markersize=6, alpha=0.5, label='Regular meeting'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#999',
           markeredgecolor='black', markeredgewidth=1.5,
           markersize=12, label='Formal dissent vote'),
]
ax.legend(handles=legend_elements, loc='upper left', fontsize=10,
          framealpha=0.9, edgecolor='#ddd')

# Title
ax.set_title('Policy Stance Timeline by Speaker',
             fontsize=16, fontweight='bold', fontfamily='serif', pad=15)

ax.set_ylim(-2.5, len(show_speakers) - 0.5)
ax.invert_yaxis()
ax.set_xlim(pd.Timestamp('1993-06-01'), pd.Timestamp('2021-06-01'))

plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/timeline_strips.png', dpi=200, bbox_inches='tight',
            facecolor='white')
print(f"    Saved: {PLOTS_DIR}/timeline_strips.png")

# ============================================================================
# [12] GPT v8 vs CLAUDE
# ============================================================================

if has_claude:
    print("\n[12] GPT v8 vs Claude comparison...")

    valid = v8.dropna(subset=['gpt_dissent_direction', 'claude_dissent_direction'])
    r = valid['gpt_dissent_direction'].corr(valid['claude_dissent_direction'])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].scatter(valid['gpt_dissent_direction'], valid['claude_dissent_direction'],
                   alpha=0.1, s=10, color='steelblue')
    axes[0].plot([-10, 10], [-10, 10], 'r--', alpha=0.5)
    axes[0].set_xlabel('GPT-4o (v8) Score', fontsize=12)
    axes[0].set_ylabel('Claude Score', fontsize=12)
    axes[0].set_title(f'GPT-4o vs Claude Scores (r={r:.3f})', fontsize=13, fontweight='bold')
    axes[0].set_xlim(-10.5, 10.5)
    axes[0].set_ylim(-10.5, 10.5)
    axes[0].grid(alpha=0.2)

    axes[1].hist(valid['gpt_dissent_direction'], bins=21, range=(-10.5, 10.5),
                alpha=0.5, label='GPT-4o', color='steelblue', edgecolor='white')
    axes[1].hist(valid['claude_dissent_direction'], bins=21, range=(-10.5, 10.5),
                alpha=0.5, label='Claude', color='#e67e22', edgecolor='white')
    axes[1].set_xlabel('Policy Stance Score', fontsize=12)
    axes[1].set_ylabel('Count', fontsize=12)
    axes[1].set_title('Score Distributions: GPT-4o vs Claude', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=11)

    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/gpt_vs_claude.png', dpi=200)
    print(f"    Saved: {PLOTS_DIR}/gpt_vs_claude.png")

    valid_c = valid.copy()
    valid_c['gpt_dir'] = np.sign(valid_c['gpt_dissent_direction'])
    valid_c['claude_dir'] = np.sign(valid_c['claude_dissent_direction'])
    agree = (valid_c['gpt_dir'] == valid_c['claude_dir']).mean()

    print(f"\n    Correlation: r = {r:.3f}")
    print(f"    Direction agreement: {agree:.1%}")
else:
    print("\n[12] Claude scores not available - skipping")

# ============================================================================
# [13] FED DISTRICT MAP: US tile grid colored by district
# ============================================================================

print("\n[13] Fed district map (tile grid)...")

STATE_TO_DISTRICT = {
    'ME': 'Boston', 'NH': 'Boston', 'VT': 'Boston', 'MA': 'Boston',
    'RI': 'Boston', 'CT': 'Boston',
    'NY': 'New York', 'NJ': 'New York',
    'PA': 'Philadelphia', 'DE': 'Philadelphia',
    'OH': 'Cleveland', 'WV': 'Cleveland', 'KY': 'Cleveland',
    'VA': 'Richmond', 'MD': 'Richmond', 'DC': 'Richmond',
    'NC': 'Richmond', 'SC': 'Richmond',
    'GA': 'Atlanta', 'FL': 'Atlanta', 'AL': 'Atlanta',
    'TN': 'Atlanta', 'MS': 'Atlanta', 'LA': 'Atlanta',
    'IL': 'Chicago', 'IN': 'Chicago', 'MI': 'Chicago',
    'WI': 'Chicago', 'IA': 'Chicago',
    'MO': 'St. Louis', 'AR': 'St. Louis',
    'MN': 'Minneapolis', 'ND': 'Minneapolis', 'SD': 'Minneapolis',
    'MT': 'Minneapolis',
    'KS': 'Kansas City', 'NE': 'Kansas City', 'OK': 'Kansas City',
    'CO': 'Kansas City', 'WY': 'Kansas City', 'NM': 'Kansas City',
    'TX': 'Dallas',
    'CA': 'San Francisco', 'OR': 'San Francisco', 'WA': 'San Francisco',
    'NV': 'San Francisco', 'UT': 'San Francisco', 'AZ': 'San Francisco',
    'ID': 'San Francisco',
}

DISTRICT_NUMBERS = {
    'Boston': 1, 'New York': 2, 'Philadelphia': 3, 'Cleveland': 4,
    'Richmond': 5, 'Atlanta': 6, 'Chicago': 7, 'St. Louis': 8,
    'Minneapolis': 9, 'Kansas City': 10, 'Dallas': 11, 'San Francisco': 12,
}

TILE_GRID = {
    'AK': (0, 0), 'ME': (10, 0),
    'WI': (5, 1), 'VT': (9, 1), 'NH': (10, 1),
    'WA': (0, 1), 'ID': (1, 1), 'MT': (2, 1), 'ND': (3, 1), 'MN': (4, 1),
    'MI': (6, 1), 'NY': (8, 1), 'MA': (9, 2), 'RI': (10, 2),
    'OR': (0, 2), 'NV': (1, 2), 'WY': (2, 2), 'SD': (3, 2), 'IA': (4, 2),
    'IL': (5, 2), 'IN': (6, 2), 'OH': (7, 2), 'PA': (8, 2), 'CT': (9, 3), 'NJ': (10, 3),
    'CA': (0, 3), 'UT': (1, 3), 'CO': (2, 3), 'NE': (3, 3), 'MO': (4, 3),
    'KY': (5, 3), 'WV': (6, 3), 'VA': (7, 3), 'MD': (8, 3), 'DE': (9, 4), 'DC': (10, 4),
    'AZ': (1, 4), 'NM': (2, 4), 'KS': (3, 4), 'AR': (4, 4),
    'TN': (5, 4), 'NC': (6, 4), 'SC': (7, 4),
    'OK': (3, 5), 'LA': (4, 5), 'MS': (5, 5), 'AL': (6, 5), 'GA': (7, 5),
    'TX': (3, 6), 'FL': (7, 6),
    'HI': (0, 6),
}

dist_data = (
    v8[v8['is_bank_president']]
    .groupby('district')
    .agg(
        mean_score=('gpt_dissent_direction', 'mean'),
        mean_abs=('gpt_dissent_direction', lambda x: x.abs().mean()),
        n_dissents=('dissented', 'sum'),
        n_meetings=('gpt_dissent_direction', 'count'),
    )
    .reset_index()
)
dist_dict = dist_data.set_index('district').to_dict('index')

fig, axes = plt.subplots(1, 2, figsize=(20, 8))

for ax_idx, (metric, title, norm_use, cmap_use, fmt) in enumerate([
    ('mean_score', 'Mean Policy Stance by Fed District',
     mcolors.Normalize(vmin=-3, vmax=3), HAWK_DOVE_CMAP, '{:+.1f}'),
    ('n_dissents', 'Total Formal Dissent Votes by District',
     mcolors.Normalize(vmin=0, vmax=25), plt.cm.YlOrRd, '{:.0f}'),
]):
    ax = axes[ax_idx]
    ax.set_facecolor('white')

    for state, (col, row) in TILE_GRID.items():
        district = STATE_TO_DISTRICT.get(state)
        if district and district in dist_dict:
            val = dist_dict[district][metric]
            color = cmap_use(norm_use(val))
        elif district == 'New York':
            color = '#d5d8dc'
        else:
            color = '#eee'

        rect = mpatches.FancyBboxPatch(
            (col, -row), 0.9, 0.9,
            boxstyle="round,pad=0.02",
            facecolor=color, edgecolor='white', linewidth=2,
            zorder=2
        )
        ax.add_patch(rect)

        text_color = '#fff' if district and district in dist_dict and abs(dist_dict.get(district, {}).get(metric, 0)) > 1.5 else '#333'
        if metric == 'n_dissents' and district in dist_dict and dist_dict[district][metric] > 12:
            text_color = '#fff'
        ax.text(col + 0.45, -row + 0.45, state, ha='center', va='center',
                fontsize=7, fontweight='bold', color=text_color, zorder=3)

    ax.set_xlim(-0.5, 11.5)
    ax.set_ylim(-7.5, 1.5)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title(title, fontsize=14, fontweight='bold', fontfamily='serif', pad=15)

    sm = plt.cm.ScalarMappable(cmap=cmap_use, norm=norm_use)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, shrink=0.5, aspect=20, pad=0.03,
                        orientation='horizontal', location='bottom')
    if ax_idx == 0:
        cbar.set_ticks([-3, -1.5, 0, 1.5, 3])
        cbar.set_ticklabels(['Hawkish', '', 'Neutral', '', 'Dovish'])
    else:
        cbar.set_label('Number of dissent votes', fontsize=10)

axes[0].text(0, -7.2, 'Gray = New York (excluded per Bobrov 2024)',
             fontsize=8, color='#888', fontfamily='serif')

plt.suptitle('Federal Reserve Districts: Speech & Dissent Patterns (1994\u20132020)',
             fontsize=16, fontweight='bold', fontfamily='serif', y=1.02)
plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/fed_district_map.png', dpi=200, bbox_inches='tight',
            facecolor='white')
print(f"    Saved: {PLOTS_DIR}/fed_district_map.png")

# ============================================================================
# [14] APPENDIX TABLE A1: Summary Statistics for Regression Sample
# ============================================================================

print("\n[14] Summary statistics for appendix (Table A1)...")

# Build the merged regression sample (same as reg.py)
scores_banks = v8[v8['is_bank_president']].copy()
scores_banks = scores_banks[scores_banks['district'] != 'New York'].copy()

# Vote direction (numeric)
def get_vote_direction(row, dissent_df):
    speaker_upper = row['speaker'].upper()
    for _, d in dissent_df[dissent_df['date'] == row['date']].iterrows():
        if d['name'] in speaker_upper:
            if d['direction'] == 'tighter':
                return -1
            elif d['direction'] == 'easier':
                return +1
            else:
                return 0
    return 0

scores_banks['vote_direction'] = scores_banks.apply(
    lambda r: get_vote_direction(r, dissent_df), axis=1
)

# Merge unemployment_rate only (unemployment_gap already in v8 from section 1)
scores_banks = scores_banks.merge(
    unemp[['year_month', 'district', 'unemployment_rate']].drop_duplicates(),
    on=['year_month', 'district'], how='inner'
)

scores_banks['post_2006'] = (scores_banks['date'] >= pd.Timestamp('2006-02-01')).astype(int)

print(f"\n    Regression sample (excl. New York): {len(scores_banks)} observations")
print(f"    Districts: {scores_banks['district'].nunique()}")
print(f"    Speakers:  {scores_banks['speaker'].nunique()}")
print(f"    Meetings:  {scores_banks['date'].nunique()}")

# --- Panel A: Main Variables ---
print(f"\n    {'='*80}")
print(f"    TABLE A1: SUMMARY STATISTICS")
print(f"    {'='*80}")
print(f"\n    Panel A: Regression Variables")
print(f"    {'Variable':<30} {'N':>6} {'Mean':>8} {'SD':>8} {'Min':>8} {'Max':>8}")
print(f"    {'-'*70}")

stats_vars = [
    ('GPT Speech Score (v8)',       'gpt_dissent_direction'),
    ('Vote Direction (-1/0/+1)',    'vote_direction'),
    ('Unemployment Gap (pp)',       'unemployment_gap'),
    ('Dist. Unemployment Rate (%)', 'unemployment_rate'),
    ('Post-2006 Indicator',         'post_2006'),
]

if has_claude and 'claude_dissent_direction' in scores_banks.columns:
    stats_vars.insert(1, ('Claude Speech Score', 'claude_dissent_direction'))

for label, col in stats_vars:
    s = scores_banks[col].dropna()
    print(f"    {label:<30} {len(s):>6} {s.mean():>+8.3f} {s.std():>8.3f} {s.min():>8.3f} {s.max():>8.3f}")

# --- Panel B: Score Distribution ---
print(f"\n    Panel B: GPT Score Distribution (Bank Presidents, excl. NY)")
gpt = scores_banks['gpt_dissent_direction']
print(f"    {'Hawkish (< 0)':<30} {(gpt < 0).sum():>6} ({(gpt < 0).mean()*100:>5.1f}%)")
print(f"    {'Neutral (= 0)':<30} {(gpt == 0).sum():>6} ({(gpt == 0).mean()*100:>5.1f}%)")
print(f"    {'Dovish (> 0)':<30} {(gpt > 0).sum():>6} ({(gpt > 0).mean()*100:>5.1f}%)")

# --- Panel C: Vote Distribution ---
print(f"\n    Panel C: Vote Distribution (Bank Presidents, excl. NY)")
vd = scores_banks['vote_direction']
print(f"    {'Tighter Dissent (-1)':<30} {(vd == -1).sum():>6} ({(vd == -1).mean()*100:>5.1f}%)")
print(f"    {'Agreement (0)':<30} {(vd == 0).sum():>6} ({(vd == 0).mean()*100:>5.1f}%)")
print(f"    {'Easier Dissent (+1)':<30} {(vd == +1).sum():>6} ({(vd == +1).mean()*100:>5.1f}%)")

# --- Panel D: By Era ---
print(f"\n    Panel D: By Era")
print(f"    {'Era':<30} {'N':>6} {'GPT Mean':>9} {'GPT SD':>8} {'Unemp Gap':>10} {'Dissents':>9}")
print(f"    {'-'*75}")

for era_name, mask in [
    ('Greenspan (1994-2005)', scores_banks['post_2006'] == 0),
    ('Post-Greenspan (2006-2020)', scores_banks['post_2006'] == 1),
]:
    sub = scores_banks[mask]
    n_dissents = (sub['vote_direction'] != 0).sum()
    print(f"    {era_name:<30} {len(sub):>6} {sub['gpt_dissent_direction'].mean():>+9.3f} {sub['gpt_dissent_direction'].std():>8.3f} {sub['unemployment_gap'].mean():>+10.4f} {n_dissents:>9}")

# --- Panel E: By District ---
print(f"\n    Panel E: By District")
print(f"    {'District':<18} {'N':>5} {'GPT Mean':>9} {'GPT SD':>8} {'Unemp Gap':>10} {'Dissents':>9}")
print(f"    {'-'*62}")

dist_summary = scores_banks.groupby('district').agg(
    n=('gpt_dissent_direction', 'count'),
    mean=('gpt_dissent_direction', 'mean'),
    sd=('gpt_dissent_direction', 'std'),
    unemp_gap=('unemployment_gap', 'mean'),
    dissents=('vote_direction', lambda x: (x != 0).sum()),
).sort_values('mean')

for dist, row in dist_summary.iterrows():
    print(f"    {dist:<18} {row['n']:>5.0f} {row['mean']:>+9.3f} {row['sd']:>8.3f} {row['unemp_gap']:>+10.4f} {row['dissents']:>9.0f}")

# --- Save to CSV for easy LaTeX/Word table creation ---
summary_rows = []
for label, col in stats_vars:
    s = scores_banks[col].dropna()
    summary_rows.append({
        'Variable': label,
        'N': int(len(s)),
        'Mean': round(s.mean(), 3),
        'SD': round(s.std(), 3),
        'Min': round(s.min(), 3),
        'Max': round(s.max(), 3),
    })

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(f'{PLOTS_DIR}/summary_statistics.csv', index=False)
print(f"\n    Saved: {PLOTS_DIR}/summary_statistics.csv")

# --- Variable Definitions Table ---
print(f"\n    {'='*80}")
print(f"    TABLE A2: VARIABLE DEFINITIONS")
print(f"    {'='*80}")
print(f"    {'Variable':<30} {'Description':<50}")
print(f"    {'-'*80}")

defs = [
    ('GPT Speech Score',       'Policy stance from GPT-4o-mini, -10 (hawk) to +10 (dove)'),
    ('Claude Speech Score',    'Policy stance from Claude Sonnet, same scale'),
    ('Vote Direction',         'Formal vote: -1 = tighter, 0 = agree, +1 = easier'),
    ('Unemployment Gap',       'District unemp. rate minus national rate (pp)'),
    ('Unemployment Rate',      'District-level rate, pop-weighted from county BLS data'),
    ('Post-2006',              'Indicator = 1 for meetings after Jan 2006'),
    ('Speaker FE',             'Fixed effects for each bank president'),
    ('Meeting FE',             'Fixed effects for each FOMC meeting date'),
]

for var, desc in defs:
    print(f"    {var:<30} {desc:<50}")

# ============================================================================
# DONE
# ============================================================================

print("\n" + "=" * 70)
print(f"Plots saved to: {PLOTS_DIR}/")
print("Done!")
print("=" * 70)