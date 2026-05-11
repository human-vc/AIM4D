"""
Fallback feature builder: derives event-style country-year features from V-Dem
indicators (protest/violence/repression). Used only when real GDELT is not
present at data/gdelt_country_year.csv. Run data/download_gdelt.py first for
the actual GDELT 1.0 events.
"""
import os
import pandas as pd
import numpy as np

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

df = pd.read_csv(
    os.path.join(DATA_DIR, "vdem_v16.csv"),
    usecols=[
        'country_name', 'country_text_id', 'COWcode', 'year',
        'v2caprotac',
        'v2caviol', 'v2x_clphy', 'e_civil_war',
        'v2csreprss', 'v2clkill', 'v2cltort',
        'v2x_civlib',
        'v2cademmob', 'v2cagenmob', 'v2caconmob', 'v2caautmob',
        'e_pt_coup', 'e_pt_coup_attempts',
    ],
    low_memory=False,
)

df = df[(df['year'] >= 1990) & (df['year'] <= 2025)].copy()

df['country_code'] = df['country_text_id']

protest_count = df['v2caprotac'].fillna(0)

conflict_raw = (
    df['v2caviol'].fillna(0)
    + (1 - df['v2x_clphy'].fillna(df['v2x_clphy'].median())) * 4
    + df['e_civil_war'].fillna(0) * 5
    + df['e_pt_coup'].fillna(0) * 3
    + df['e_pt_coup_attempts'].fillna(0) * 2
)
conflict_count = ((conflict_raw - conflict_raw.min()) / (conflict_raw.max() - conflict_raw.min()) * 10).round(2)

repression_raw = (
    df['v2csreprss'].fillna(0) * -1
    + df['v2clkill'].fillna(0) * -1
    + df['v2cltort'].fillna(0) * -1
)
repression_count = ((repression_raw - repression_raw.min()) / (repression_raw.max() - repression_raw.min()) * 10).round(2)

goldstein_raw = df['v2x_civlib'].fillna(df['v2x_civlib'].median())
avg_goldstein = ((goldstein_raw * 20) - 10).round(2)

tone_components = (
    df['v2x_civlib'].fillna(0.5)
    + df['v2csreprss'].fillna(0) / 4
    - df['v2caviol'].fillna(0) / 4
)
avg_tone = ((tone_components - tone_components.min()) / (tone_components.max() - tone_components.min()) * 20 - 10).round(2)

total_events_raw = (
    df['v2cademmob'].fillna(0).abs()
    + df['v2cagenmob'].fillna(0).abs()
    + df['v2caconmob'].fillna(0).abs()
    + df['v2caautmob'].fillna(0).abs()
    + df['v2caprotac'].fillna(0)
)
total_events = total_events_raw.round(2)

out = pd.DataFrame({
    'country_code': df['country_code'],
    'year': df['year'],
    'protest_count': protest_count,
    'conflict_count': conflict_count,
    'repression_count': repression_count,
    'avg_goldstein': avg_goldstein,
    'avg_tone': avg_tone,
    'total_events': total_events,
})

out = out.dropna(subset=['country_code'])
out = out.sort_values(['country_code', 'year']).reset_index(drop=True)

print(f"Shape: {out.shape}")
print(f"Countries: {out['country_code'].nunique()}")
print(f"Year range: {out['year'].min()}-{out['year'].max()}")
print()
print(out.describe())
print()
print(out.head(10))

data_dir = os.path.dirname(os.path.abspath(__file__))
proxy_path = os.path.join(data_dir, "gdelt_proxy.csv")
gdelt_path = os.path.join(data_dir, "gdelt_country_year.csv")

out.to_csv(proxy_path, index=False)
print(f"\nSaved fallback to {proxy_path}")

if os.path.exists(gdelt_path):
    print(f"WARNING: {gdelt_path} already exists; refusing to overwrite real GDELT data.")
    print("Delete it manually if you intend to use the V-Dem-derived proxy instead.")
else:
    out.to_csv(gdelt_path, index=False)
    print(f"Saved fallback to {gdelt_path} (no real GDELT detected)")
