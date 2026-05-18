"""
G9: Change-point features via PELT (Pruned Exact Linear Time).

For each country, run PELT on the V-Dem polyarchy + libdem time series and
record "years since last break" and "break-in-last-3-years" per country-year.
Output: data/changepoints.csv with columns
  country_text_id, year, years_since_break_poly, years_since_break_libdem,
  break_in_last_3yr_poly, break_in_last_3yr_libdem
"""

import os
import numpy as np
import pandas as pd

try:
    import ruptures as rpt
except ImportError:
    raise SystemExit("Install ruptures: pip install ruptures")

DATA = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(DATA, "changepoints.csv")


def detect_breaks(series, penalty=1.0, model="l2", min_size=4):
    """Return sorted list of break-year indices for a 1-D series."""
    x = np.asarray(series, dtype=float)
    if len(x) < min_size * 2:
        return []
    try:
        algo = rpt.Pelt(model=model, min_size=min_size, jump=1).fit(x)
        bkps = algo.predict(pen=penalty)
        # bkps includes the endpoint; trim it
        return [b for b in bkps if b < len(x)]
    except Exception:
        return []


def per_country_features(years, breaks):
    """For each year, compute years-since-last-break and 3yr-window flag."""
    yss = np.zeros(len(years), dtype=int)
    win3 = np.zeros(len(years), dtype=int)
    last_break_year = -9999
    break_years = set(int(years[b]) for b in breaks)
    for i, y in enumerate(years):
        y = int(y)
        if y in break_years:
            last_break_year = y
        yss[i] = (y - last_break_year) if last_break_year > -9999 else 99
        # any break in last 3 years (incl. current)?
        win3[i] = 1 if any(by for by in break_years if y - 2 <= by <= y) else 0
    return yss, win3


def main():
    print("Loading V-Dem polyarchy and libdem time series...")
    df = pd.read_csv(
        os.path.join(DATA, "vdem_v16.csv"), low_memory=False,
        usecols=["country_text_id", "year", "v2x_polyarchy", "v2x_libdem"],
    )
    df = df.dropna(subset=["v2x_polyarchy"]).sort_values(["country_text_id", "year"])

    rows = []
    for cid, grp in df.groupby("country_text_id"):
        grp = grp.sort_values("year").reset_index(drop=True)
        years = grp["year"].values
        if len(grp) < 8:
            continue

        poly = grp["v2x_polyarchy"].values
        # ffill only — bfill would inject future libdem values into past rows.
        libdem = grp["v2x_libdem"].ffill().fillna(0.0).values

        bp = detect_breaks(poly, penalty=0.05, model="l2", min_size=4)
        bl = detect_breaks(libdem, penalty=0.05, model="l2", min_size=4)

        yss_p, w3_p = per_country_features(years, bp)
        yss_l, w3_l = per_country_features(years, bl)

        for i, y in enumerate(years):
            rows.append({
                "country_text_id": cid,
                "year": int(y),
                "years_since_break_poly": int(yss_p[i]),
                "years_since_break_libdem": int(yss_l[i]),
                "break_in_last_3yr_poly": int(w3_p[i]),
                "break_in_last_3yr_libdem": int(w3_l[i]),
            })

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print(f"Wrote {OUT}: {len(out)} rows, {out['country_text_id'].nunique()} countries, "
          f"{out['year'].min()}-{out['year'].max()}")


if __name__ == "__main__":
    main()
