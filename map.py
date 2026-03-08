#!/usr/bin/env python3
"""
Federal Reserve District Choropleths
  1) Mean policy stance by district (from GPT v8 scores)
  2) Dissent vote rate by district
All values computed dynamically from data files.
"""

import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os, sys

# ── mappings ────────────────────────────────────────────────────────────────

STATE_TO_DISTRICT = {
    'CT':'Boston','MA':'Boston','ME':'Boston','NH':'Boston','RI':'Boston','VT':'Boston',
    'NY':'New York',
    'DE':'Philadelphia','NJ':'Philadelphia','PA':'Philadelphia',
    'OH':'Cleveland','KY':'Cleveland',
    'DC':'Richmond','MD':'Richmond','NC':'Richmond','SC':'Richmond','VA':'Richmond','WV':'Richmond',
    'AL':'Atlanta','FL':'Atlanta','GA':'Atlanta','MS':'Atlanta','TN':'Atlanta',
    'IA':'Chicago','IL':'Chicago','IN':'Chicago','MI':'Chicago','WI':'Chicago',
    'AR':'St. Louis','MO':'St. Louis',
    'MN':'Minneapolis','MT':'Minneapolis','ND':'Minneapolis','SD':'Minneapolis',
    'CO':'Kansas City','KS':'Kansas City','NE':'Kansas City','NM':'Kansas City','OK':'Kansas City','WY':'Kansas City',
    'TX':'Dallas','LA':'Dallas',
    'AK':'San Francisco','AZ':'San Francisco','CA':'San Francisco','HI':'San Francisco',
    'ID':'San Francisco','NV':'San Francisco','OR':'San Francisco','UT':'San Francisco','WA':'San Francisco',
}

DISTRICT_NUM = {
    "Boston":1,"New York":2,"Philadelphia":3,"Cleveland":4,"Richmond":5,
    "Atlanta":6,"Chicago":7,"St. Louis":8,"Minneapolis":9,"Kansas City":10,
    "Dallas":11,"San Francisco":12,
}

DISTRICT_CENTROIDS = {
    "Boston": (42.8, -71.5),
    "Philadelphia": (40.3, -76.0),
    "Cleveland": (39.8, -82.5),
    "Richmond": (36.0, -79.0),
    "Atlanta": (32.5, -85.0),
    "Chicago": (42.5, -88.0),
    "St. Louis": (36.0, -92.5),
    "Minneapolis": (46.5, -101.0),
    "Kansas City": (37.5, -100.5),
    "Dallas": (31.5, -98.0),
    "San Francisco": (42.0, -118.0),
}

# ── load data ───────────────────────────────────────────────────────────────

print("[1] Loading scores...")

scores_path = "data/cache/gpt_dissent_scores_v8.csv"
if not os.path.exists(scores_path):
    sys.exit(f"ERROR: {scores_path} not found. Run from thesis root.")

scores = pd.read_csv(scores_path)
scores['date'] = pd.to_datetime(scores['date'])

scores_ex = scores[scores['district'] != 'New York'].copy()

district_means = scores_ex.groupby('district')['gpt_dissent_direction'].mean()
print(f"    Computed mean stance for {len(district_means)} districts")
for d in sorted(district_means.index, key=lambda x: DISTRICT_NUM.get(x, 99)):
    print(f"      {DISTRICT_NUM.get(d, '?'):>2}  {d:<16} {district_means[d]:+.3f}")

# ── load dissent votes ──────────────────────────────────────────────────────

print("\n[2] Loading dissent votes...")

dissent_path = "data/FOMC_Dissents_Data.xlsx"
if not os.path.exists(dissent_path):
    print(f"    WARNING: {dissent_path} not found — skipping dissent map")
    has_dissent = False
else:
    has_dissent = True
    votes = pd.read_excel(dissent_path, skiprows=3)
    votes["date"] = pd.to_datetime(votes["FOMC Meeting"])

    dissent_records = []
    for _, row in votes.iterrows():
        for col, direction in [("Dissenters Tighter", -1),
                                ("Dissenters Easier", +1),
                                ("Dissenters Other/Indeterminate", 0)]:
            if pd.notna(row.get(col)):
                for name in str(row[col]).split(", "):
                    dissent_records.append({
                        "date": row["date"],
                        "name": name.strip().upper(),
                        "vote_direction": direction,
                    })
    dissent_df = pd.DataFrame(dissent_records)

    scores_ex['vote_direction'] = 0
    for idx, row in scores_ex.iterrows():
        speaker_upper = row['speaker'].upper()
        meeting_dissents = dissent_df[dissent_df['date'] == row['date']]
        for _, d in meeting_dissents.iterrows():
            if d['name'] in speaker_upper:
                scores_ex.at[idx, 'vote_direction'] = d['vote_direction']

    district_dissent = scores_ex.groupby('district').apply(
        lambda g: (g['vote_direction'] != 0).mean(), include_groups=False
    )
    district_dissent_n = scores_ex.groupby('district').apply(
        lambda g: (g['vote_direction'] != 0).sum(), include_groups=False
    )

    print(f"    Computed dissent rate for {len(district_dissent)} districts")
    for d in sorted(district_dissent.index, key=lambda x: DISTRICT_NUM.get(x, 99)):
        print(f"      {DISTRICT_NUM.get(d, '?'):>2}  {d:<16} {district_dissent[d]:.3f}  ({int(district_dissent_n[d])} dissents)")

# ── helper: build a map ─────────────────────────────────────────────────────

BG_COLOR = '#f0d9a8'  # warm golden-orange ochre

def make_map(district_values, title_main, title_sub, colorscale, zmin, zmax,
             tickvals, ticktext, outname):
    states = list(STATE_TO_DISTRICT.keys())
    z_vals, hover = [], []
    for st in states:
        dist = STATE_TO_DISTRICT[st]
        num = DISTRICT_NUM[dist]
        if dist == "New York":
            z_vals.append(None)
            hover.append(f"District {num}: {dist}<br>(Excluded per Bobrov 2024)")
        elif dist in district_values:
            val = district_values[dist]
            z_vals.append(val)
            hover.append(f"District {num}: {dist}<br>Value: {val:+.3f}")
        else:
            z_vals.append(None)
            hover.append(f"District {num}: {dist}<br>No data")

    fig = go.Figure()

    fig.add_trace(go.Choropleth(
        locationmode='USA-states',
        locations=states,
        z=z_vals,
        colorscale=colorscale,
        zmin=zmin, zmax=zmax,
        text=hover,
        hoverinfo='text',
        marker_line_color='white',
        marker_line_width=1.5,
        colorbar=dict(
            thickness=14, len=0.4, x=0.93, y=0.5,
            tickvals=tickvals,
            ticktext=ticktext,
            tickfont=dict(size=10, family="Georgia"),
        ),
    ))

    lats = [c[0] for c in DISTRICT_CENTROIDS.values()]
    lons = [c[1] for c in DISTRICT_CENTROIDS.values()]
    nums = [str(DISTRICT_NUM[d]) for d in DISTRICT_CENTROIDS.keys()]

    fig.add_trace(go.Scattergeo(
        locationmode='USA-states',
        lat=lats, lon=lons,
        text=nums,
        mode='text',
        textfont=dict(size=12, color='#2c3e50', family='Georgia'),
        hoverinfo='skip',
        showlegend=False,
    ))

    legend_parts = [f"<b>{num}</b> {name}"
                    for name, num in sorted(DISTRICT_NUM.items(), key=lambda x: x[1])]
    legend_str = "   ".join(legend_parts)

    fig.update_layout(
        title=dict(
            text=(f"{title_main}"
                  f"<br><span style='font-size:11px;color:#7a6840'>{title_sub}</span>"),
            x=0.5, xanchor='center',
            font=dict(size=17, family="Georgia", color="#2c3e50"),
        ),
        geo=dict(
            scope='usa',
            projection=dict(type='albers usa'),
            showlakes=False,
            bgcolor=BG_COLOR,
            landcolor=BG_COLOR,
        ),
        width=1000, height=620,
        margin=dict(l=10, r=10, t=80, b=60),
        paper_bgcolor=BG_COLOR,
        annotations=[dict(
            text=legend_str,
            x=0.5, y=-0.02, xref='paper', yref='paper',
            showarrow=False, xanchor='center',
            font=dict(size=9, family='Georgia', color='#6b5c3e'),
        )],
    )

    os.makedirs("plots", exist_ok=True)
    fig.write_image(f"plots/{outname}.png", scale=3)
    fig.write_image(f"plots/{outname}.pdf")
    fig.write_image(f"plots/{outname}.svg")
    print(f"    Saved plots/{outname}.{{png,pdf,svg}}")

# ── MAP 1: Mean policy stance (red ↔ blue) ─────────────────────────────────

print("\n[3] Generating stance map...")

make_map(
    district_values=district_means.to_dict(),
    title_main="Federal Reserve Districts: Mean Policy Stance (GPT v8, 1994–2020)",
    title_sub="Gray = New York (excluded per Bobrov 2024). Numbers = Fed district.",
    colorscale=[
        [0.0, '#b71c1c'], [0.25, '#e57373'], [0.4, '#fce4ec'],
        [0.5, '#ffffff'], [0.6, '#bbdefb'], [0.75, '#42a5f5'], [1.0, '#1565c0'],
    ],
    zmin=-3, zmax=3,
    tickvals=[-3, 0, 3],
    ticktext=["Hawkish", "Neutral", "Dovish"],
    outname="fed_district_stance",
)

# ── MAP 2: Dissent rate (white → purple) ───────────────────────────────────

if has_dissent:
    print("\n[4] Generating dissent map...")

    dmax = float(max(district_dissent.values)) * 1.1

    make_map(
        district_values=district_dissent.to_dict(),
        title_main="Federal Reserve Districts: Dissent Vote Rate (1994–2020)",
        title_sub="Gray = New York (excluded). Fraction of meetings with a formal dissent.",
        colorscale=[
            [0.0, '#f5f0ff'], [0.25, '#d4b9f7'], [0.5, '#9b59b6'],
            [0.75, '#6c3483'], [1.0, '#2e0854'],
        ],
        zmin=0, zmax=dmax,
        tickvals=[0, round(float(max(district_dissent.values)), 2)],
        ticktext=["0%", f"{float(max(district_dissent.values))*100:.0f}%"],
        outname="fed_district_dissent",
    )

print("\n✅ Done!")