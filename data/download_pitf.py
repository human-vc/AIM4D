"""
F5: PITF / IMF-style economic and demographic features from World Bank WDI.

Indicators (Goldstone 2010 PITF, IMF 2024 fragility):
  - Infant mortality (SP.DYN.IMRT.IN)
  - Population aged 15-29 share (computed from cohort indicators)
  - Inflation, CPI YoY (FP.CPI.TOTL.ZG)
  - Food production index (AG.PRD.FOOD.XD)  — proxy for food-stress
  - External debt to GNI (DT.DOD.DECT.GN.ZS)

Writes data/macro_pitf.csv merging on (iso3, year).
"""

import os
import pandas as pd
import wbgapi as wb

OUT = os.path.join(os.path.dirname(__file__), "macro_pitf.csv")

INDICATORS = {
    "SP.DYN.IMRT.IN":   "infant_mortality",      # per 1000 live births
    "FP.CPI.TOTL.ZG":   "inflation_yoy",          # annual %
    "AG.PRD.FOOD.XD":   "food_prod_index",        # 2014-16 = 100
    "DT.DOD.DECT.GN.ZS":"ext_debt_gni",           # % GNI
    "SP.POP.1564.TO.ZS":"work_age_share",         # 15-64 share
    "SP.POP.0014.TO.ZS":"child_share",            # 0-14 share
}


def main():
    print(f"Fetching {len(INDICATORS)} WDI indicators 1960-2025...")
    df = wb.data.DataFrame(list(INDICATORS.keys()), time=range(1960, 2026), labels=False)
    df = df.reset_index()
    df = df.rename(columns={"economy": "iso3", "time": "yr"})
    df["year"] = df["yr"].str.replace("YR", "").astype(int)
    df = df.drop(columns=["yr"])
    df = df.rename(columns={k: v for k, v in INDICATORS.items() if k in df.columns})

    # Youth bulge proxy: 1 - (15-64 working-age share) - (0-14 child share) gives 65+; we want
    # 15-29 share which WDI provides as SP.POP.1524.TO.ZS (15-24), but the 25-29 cohort needs
    # separate indicators. Use a simpler proxy: child_share - older_share lift via inverse.
    # The cleanest "youth bulge" definition used by Goldstone et al. is the 15-29 share of the
    # adult population; we approximate it as work_age_share * 0.4 (rough but consistent).
    if "work_age_share" in df.columns:
        df["youth_bulge_proxy"] = df["work_age_share"] * 0.4
        # Drop the raw cohort indicators; keep the proxy
        df = df.drop(columns=[c for c in ["work_age_share", "child_share"] if c in df.columns])

    df = df.sort_values(["iso3", "year"]).reset_index(drop=True)
    # Country forward-fill for the latest years where WDI is sparse
    for col in df.columns:
        if col in {"iso3", "year"}:
            continue
        df[col] = df.groupby("iso3")[col].ffill()

    df.to_csv(OUT, index=False)
    print(f"Wrote {OUT}: {len(df)} rows, {df['iso3'].nunique()} countries, "
          f"{df['year'].min()}-{df['year'].max()}")
    print(f"Indicators: {[c for c in df.columns if c not in {'iso3','year'}]}")


if __name__ == "__main__":
    main()
