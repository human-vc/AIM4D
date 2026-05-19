"""
Download UCDP-GED (Uppsala Conflict Data Program — Georeferenced Event Dataset).

Used by robustness/ucdp_overlap_test.py to gate the transfer-learning plan.
GED records all violent events; we aggregate to country-year and flag state-based
conflict (type_of_violence == 1) crossing the 25-battle-deaths threshold per
Hegre et al. (2019) / ViEWS convention.

UCDP-GED is published under CC-BY 4.0. https://ucdp.uu.se/downloads/

Tries v25.1 then v24.1 then v23.1. Saves to data/ucdp_ged.csv (event-level).
"""

import io
import os
import sys
import urllib.request
import zipfile
import pandas as pd

DATA = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(DATA, "ucdp_ged.csv")

UCDP_URLS = [
    "https://ucdp.uu.se/downloads/ged/ged251-csv.zip",
    "https://ucdp.uu.se/downloads/ged/ged241-csv.zip",
    "https://ucdp.uu.se/downloads/ged/ged231-csv.zip",
    "https://ucdp.uu.se/downloads/ged/ged221-csv.zip",
]


def main():
    if os.path.exists(OUT):
        print(f"UCDP-GED already present: {OUT}")
        return OUT
    headers = {"User-Agent": "Mozilla/5.0 AIM4D-research"}
    for url in UCDP_URLS:
        print(f"  trying {url[:80]} ...")
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = resp.read()
            print(f"    downloaded {len(data)/1e6:.1f} MB")
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                csv_names = [n for n in z.namelist() if n.endswith(".csv")]
                if not csv_names:
                    print("    zip has no .csv files")
                    continue
                with z.open(csv_names[0]) as f:
                    df = pd.read_csv(f, low_memory=False)
                df.to_csv(OUT, index=False)
                print(f"  saved {len(df)} events to {OUT}")
                print(f"  columns: {list(df.columns)[:10]}...")
                print(f"  years: {df['year'].min()}-{df['year'].max()}" if "year" in df.columns else "")
                return OUT
        except Exception as e:
            print(f"    failed: {type(e).__name__}: {str(e)[:140]}")
    print()
    print("MANUAL DOWNLOAD:")
    print("  1. Go to https://ucdp.uu.se/downloads/")
    print("  2. Download the GED 'csv' version (latest available)")
    print(f"  3. Unzip and place the .csv at {OUT}")
    print("  4. Re-run this script (or skip this step — it will find it)")
    sys.exit(1)


if __name__ == "__main__":
    main()
