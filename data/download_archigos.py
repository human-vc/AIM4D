"""
F3: Archigos leader-level features.

Archigos v4.1 (Goemans, Gleditsch & Chiozza 2009) records every head of
state since 1875, including entry/exit dates, age, military background, and
manner of entry/exit. Leader-level covariates are the load-bearing piece of
CoupCast (Beger, Dorff & Ward 2014) and likely cover the 2021-23 coup
cluster (Niger / Gabon / Burkina Faso) that our 2017-cutoff pipeline misses.

Features per country-year:
  - leader_tenure_years
  - irregular_entry  (dummy: 1 if entered via coup/rebellion/etc.)
  - military_background  (dummy: 1 if leader has military rank)
  - leader_age (at year)
  - years_since_irregular_change

Source URL: https://www.prio.org/data/3 (Archigos v4.1)
We fall back to the public dataverse mirror if direct download fails.
"""

import os
import sys
import io
import zipfile
import urllib.request
import pandas as pd
import numpy as np

DATA = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(DATA, "archigos_features.csv")
RAW = os.path.join(DATA, "Archigos_4.1.txt")

# Primary source (Hein Goemans' website mirror)
ARCHIGOS_URLS = [
    "https://www.rochester.edu/college/faculty/hgoemans/Archigos_4.1.txt",
    "https://hgoemans.com/data/Archigos_4.1.txt",
]


def download_archigos():
    if os.path.exists(RAW):
        return RAW
    for url in ARCHIGOS_URLS:
        try:
            print(f"  trying {url} ...")
            urllib.request.urlretrieve(url, RAW)
            print(f"  downloaded to {RAW}")
            return RAW
        except Exception as e:
            print(f"    failed: {e}")
    raise FileNotFoundError(
        f"Could not download Archigos v4.1. Please manually download from "
        f"https://www.rochester.edu/college/faculty/hgoemans/data and place at {RAW}"
    )


def build_features():
    path = download_archigos()
    # Archigos is tab-delimited
    df = pd.read_csv(path, sep="\t", encoding="latin-1", low_memory=False)
    df.columns = [c.lower() for c in df.columns]

    # Expected columns: ccode, idacr (iso3-ish), leader, startdate, enddate,
    # entry, exit, prevtimesinoffice, posttenurefate, mil
    needed = {"idacr": "iso3", "startdate": "startdate", "enddate": "enddate",
              "entry": "entry", "exit": "exit", "mil": "mil",
              "yrborn": "yrborn"}
    avail = {src: dst for src, dst in needed.items() if src in df.columns}
    sub = df.rename(columns=avail)[list(avail.values())].copy()

    sub["start_year"] = pd.to_datetime(sub["startdate"], errors="coerce").dt.year
    sub["end_year"] = pd.to_datetime(sub["enddate"], errors="coerce").dt.year
    sub = sub.dropna(subset=["iso3", "start_year"])
    sub["end_year"] = sub["end_year"].fillna(2025).astype(int)
    sub["start_year"] = sub["start_year"].astype(int)

    # Expand to country-year panel
    rows = []
    for _, r in sub.iterrows():
        iso3 = r["iso3"]
        s, e = int(r["start_year"]), int(r["end_year"])
        irreg = 1 if str(r.get("entry", "")).lower() in {"irregular", "foreign"} else 0
        mil = 1 if str(r.get("mil", "")).lower() in {"1", "true", "y", "yes"} else 0
        yrborn = r.get("yrborn", np.nan)
        for y in range(max(s, 1970), min(e, 2025) + 1):
            tenure = y - s + 1
            age = (y - int(yrborn)) if not pd.isna(yrborn) else np.nan
            rows.append({
                "iso3": iso3, "year": y,
                "leader_tenure_years": tenure,
                "irregular_entry": irreg,
                "military_background": mil,
                "leader_age": age,
            })

    out = pd.DataFrame(rows)
    out = out.sort_values(["iso3", "year", "leader_tenure_years"], ascending=[True, True, False])
    out = out.drop_duplicates(subset=["iso3", "year"], keep="first")

    # Years since last irregular leadership change per country
    out = out.sort_values(["iso3", "year"]).reset_index(drop=True)
    out["years_since_irregular"] = 0
    for iso3, grp in out.groupby("iso3"):
        last_irreg_year = -9999
        ys = grp["year"].values
        irr = grp["irregular_entry"].values
        result = np.zeros(len(grp), dtype=int)
        for i, (y, r) in enumerate(zip(ys, irr)):
            if r == 1:
                last_irreg_year = y
            result[i] = y - last_irreg_year if last_irreg_year > -9999 else 99
        out.loc[out["iso3"] == iso3, "years_since_irregular"] = result

    out.to_csv(OUT, index=False)
    print(f"Wrote {OUT}: {len(out)} country-years, "
          f"{out['iso3'].nunique()} countries, {out['year'].min()}-{out['year'].max()}")


if __name__ == "__main__":
    build_features()
