"""
Downloads GDELT 1.0 events for 1990-2025 and aggregates to country-year.

GDELT 1.0 archive structure:
  1979-2005: data/events/YYYY.zip            (yearly, ~37-160 MB each)
  2006-2013-03: data/events/YYYYMM.zip       (monthly, ~10-100 MB each)
  2013-04 onwards: data/events/YYYYMMDD.export.CSV.zip  (daily, ~5-10 MB each)

Output: data/gdelt_country_year.csv with columns
  country_code (ISO3), year, protest_count, conflict_count, repression_count,
  total_events, avg_goldstein, avg_tone, num_mentions.

CAMEO root codes used:
  14 -> Protest
  17 -> Coerce (govt repression)
  18, 19, 20 -> Assault / Fight / Use Unconventional Mass Violence

Resume-capable: per-file aggregates land in data/gdelt_cache/ and are skipped
on re-run. Set GDELT_WORKERS to control parallelism (default 8).
"""

import io
import os
import sys
import zipfile
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta

import pandas as pd
import requests

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(DATA_DIR, "gdelt_cache")
OUTPUT_CSV = os.path.join(DATA_DIR, "gdelt_country_year.csv")
BASE_URL = "http://data.gdeltproject.org/events"

YEAR_START = 1990
YEAR_END = 2025
WORKERS = int(os.environ.get("GDELT_WORKERS", "8"))
CHUNK_SIZE = 500_000
REQ_TIMEOUT = 120

USECOLS = [1, 28, 30, 31, 34, 51]
COLNAMES = ["sqldate", "event_root", "goldstein", "num_mentions", "avg_tone", "geo_fips"]

PROTEST_ROOT = {14}
CONFLICT_ROOTS = {18, 19, 20}
REPRESSION_ROOT = {17}

FIPS_TO_ISO3 = {
    "AF": "AFG", "AL": "ALB", "AG": "DZA", "AN": "AND", "AO": "AGO", "AC": "ATG",
    "AR": "ARG", "AM": "ARM", "AS": "AUS", "AU": "AUT", "AJ": "AZE", "BF": "BHS",
    "BA": "BHR", "BG": "BGD", "BB": "BRB", "BO": "BLR", "BE": "BEL", "BH": "BLZ",
    "BN": "BEN", "BT": "BTN", "BL": "BOL", "BK": "BIH", "BC": "BWA", "BR": "BRA",
    "BX": "BRN", "BU": "BGR", "UV": "BFA", "BM": "MMR", "BY": "BDI", "CB": "KHM",
    "CM": "CMR", "CA": "CAN", "CV": "CPV", "CT": "CAF", "CD": "TCD", "CI": "CHL",
    "CH": "CHN", "CO": "COL", "CN": "COM", "CF": "COG", "CG": "COD", "CS": "CRI",
    "IV": "CIV", "HR": "HRV", "CU": "CUB", "CY": "CYP", "EZ": "CZE", "DA": "DNK",
    "DJ": "DJI", "DO": "DMA", "DR": "DOM", "EC": "ECU", "EG": "EGY", "ES": "SLV",
    "EK": "GNQ", "ER": "ERI", "EN": "EST", "ET": "ETH", "FJ": "FJI", "FI": "FIN",
    "FR": "FRA", "GB": "GAB", "GA": "GMB", "GG": "GEO", "GM": "DEU", "GH": "GHA",
    "GR": "GRC", "GJ": "GRD", "GT": "GTM", "GV": "GIN", "PU": "GNB", "GY": "GUY",
    "HA": "HTI", "HO": "HND", "HK": "HKG", "HU": "HUN", "IC": "ISL", "IN": "IND",
    "ID": "IDN", "IR": "IRN", "IZ": "IRQ", "EI": "IRL", "IS": "ISR", "IT": "ITA",
    "JM": "JAM", "JA": "JPN", "JO": "JOR", "KZ": "KAZ", "KE": "KEN", "KR": "KIR",
    "KN": "PRK", "KS": "KOR", "KU": "KWT", "KG": "KGZ", "LA": "LAO", "LG": "LVA",
    "LE": "LBN", "LT": "LSO", "LI": "LBR", "LY": "LBY", "LS": "LIE", "LH": "LTU",
    "LU": "LUX", "MK": "MKD", "MA": "MDG", "MI": "MWI", "MY": "MYS", "MV": "MDV",
    "ML": "MLI", "MT": "MLT", "RM": "MHL", "MR": "MRT", "MP": "MUS", "MX": "MEX",
    "FM": "FSM", "MD": "MDA", "MN": "MCO", "MG": "MNG", "MJ": "MNE", "MO": "MAR",
    "MZ": "MOZ", "WA": "NAM", "NR": "NRU", "NP": "NPL", "NL": "NLD", "NZ": "NZL",
    "NU": "NIC", "NG": "NER", "NI": "NGA", "NO": "NOR", "MU": "OMN", "PK": "PAK",
    "PS": "PLW", "PM": "PAN", "PP": "PNG", "PA": "PRY", "PE": "PER", "RP": "PHL",
    "PL": "POL", "PO": "PRT", "QA": "QAT", "RO": "ROU", "RS": "RUS", "RW": "RWA",
    "SC": "KNA", "ST": "LCA", "VC": "VCT", "WS": "WSM", "SM": "SMR", "TP": "STP",
    "SA": "SAU", "SG": "SEN", "RI": "SRB", "SE": "SYC", "SL": "SLE", "SN": "SGP",
    "LO": "SVK", "SI": "SVN", "BP": "SLB", "SO": "SOM", "SF": "ZAF", "OD": "SSD",
    "SP": "ESP", "CE": "LKA", "SU": "SDN", "NS": "SUR", "WZ": "SWZ", "SW": "SWE",
    "SZ": "CHE", "SY": "SYR", "TW": "TWN", "TI": "TJK", "TZ": "TZA", "TH": "THA",
    "TT": "TLS", "TO": "TGO", "TN": "TON", "TD": "TTO", "TS": "TUN", "TU": "TUR",
    "TX": "TKM", "TV": "TUV", "UG": "UGA", "UP": "UKR", "AE": "ARE", "UK": "GBR",
    "US": "USA", "UY": "URY", "UZ": "UZB", "NH": "VUT", "VT": "VAT", "VE": "VEN",
    "VM": "VNM", "YM": "YEM", "ZA": "ZMB", "ZI": "ZWE",
}


def url_for_stamp(stamp):
    return f"{BASE_URL}/{stamp}.export.CSV.zip" if len(stamp) == 8 else f"{BASE_URL}/{stamp}.zip"


def build_stamps():
    stamps = []
    for y in range(YEAR_START, 2006):
        stamps.append(str(y))
    for y in range(2006, 2013):
        for m in range(1, 13):
            stamps.append(f"{y}{m:02d}")
    for m in range(1, 4):
        stamps.append(f"2013{m:02d}")
    d = date(2013, 4, 1)
    end = date(YEAR_END, 12, 31)
    while d <= end:
        stamps.append(d.strftime("%Y%m%d"))
        d += timedelta(days=1)
    return stamps


def aggregate_chunk(chunk):
    chunk = chunk.dropna(subset=["sqldate", "event_root", "geo_fips"]).copy()
    chunk["year"] = pd.to_numeric(chunk["sqldate"], errors="coerce") // 10_000
    chunk["event_root"] = pd.to_numeric(chunk["event_root"], errors="coerce")
    chunk["goldstein"] = pd.to_numeric(chunk["goldstein"], errors="coerce")
    chunk["num_mentions"] = pd.to_numeric(chunk["num_mentions"], errors="coerce").fillna(0)
    chunk["avg_tone"] = pd.to_numeric(chunk["avg_tone"], errors="coerce")
    chunk = chunk.dropna(subset=["year", "event_root", "geo_fips"])
    chunk["year"] = chunk["year"].astype(int)
    chunk = chunk[(chunk["year"] >= YEAR_START) & (chunk["year"] <= YEAR_END)]

    chunk["is_protest"] = chunk["event_root"].astype(int).isin(PROTEST_ROOT).astype(int)
    chunk["is_conflict"] = chunk["event_root"].astype(int).isin(CONFLICT_ROOTS).astype(int)
    chunk["is_repression"] = chunk["event_root"].astype(int).isin(REPRESSION_ROOT).astype(int)

    grouped = chunk.groupby(["geo_fips", "year"]).agg(
        protest_count=("is_protest", "sum"),
        conflict_count=("is_conflict", "sum"),
        repression_count=("is_repression", "sum"),
        total_events=("event_root", "count"),
        goldstein_sum=("goldstein", "sum"),
        goldstein_n=("goldstein", "count"),
        tone_sum=("avg_tone", "sum"),
        tone_n=("avg_tone", "count"),
        num_mentions=("num_mentions", "sum"),
    ).reset_index()
    return grouped


def process_stamp(stamp):
    cache_path = os.path.join(CACHE_DIR, f"{stamp}.parquet")
    if os.path.exists(cache_path):
        return stamp, "cached", None

    url = url_for_stamp(stamp)
    try:
        resp = requests.get(url, timeout=REQ_TIMEOUT)
    except requests.RequestException as e:
        return stamp, "network_error", str(e)

    if resp.status_code != 200:
        return stamp, f"http_{resp.status_code}", None

    try:
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            inner = zf.namelist()[0]
            with zf.open(inner) as fh:
                aggs = []
                for chunk in pd.read_csv(
                    fh, sep="\t", header=None, usecols=USECOLS, names=COLNAMES,
                    chunksize=CHUNK_SIZE, low_memory=False, on_bad_lines="skip",
                    dtype=str,
                ):
                    aggs.append(aggregate_chunk(chunk))
    except (zipfile.BadZipFile, pd.errors.EmptyDataError, ValueError) as e:
        return stamp, "parse_error", str(e)

    if not aggs:
        empty = pd.DataFrame(columns=[
            "geo_fips", "year", "protest_count", "conflict_count", "repression_count",
            "total_events", "goldstein_sum", "goldstein_n", "tone_sum", "tone_n", "num_mentions"
        ])
        empty.to_parquet(cache_path)
        return stamp, "empty", None

    combined = pd.concat(aggs, ignore_index=True)
    final = combined.groupby(["geo_fips", "year"]).sum().reset_index()
    final.to_parquet(cache_path)
    return stamp, "ok", len(final)


def merge_cache():
    parts = []
    for fn in sorted(os.listdir(CACHE_DIR)):
        if fn.endswith(".parquet"):
            parts.append(pd.read_parquet(os.path.join(CACHE_DIR, fn)))
    if not parts:
        raise RuntimeError("No cached aggregates found")
    merged = pd.concat(parts, ignore_index=True)
    final = merged.groupby(["geo_fips", "year"]).sum().reset_index()
    final["avg_goldstein"] = (final["goldstein_sum"] / final["goldstein_n"].clip(lower=1)).round(2)
    final["avg_tone"] = (final["tone_sum"] / final["tone_n"].clip(lower=1)).round(2)
    final = final.drop(columns=["goldstein_sum", "goldstein_n", "tone_sum", "tone_n"])
    final["country_code"] = final["geo_fips"].map(FIPS_TO_ISO3)
    final = final.dropna(subset=["country_code"])
    cols = ["country_code", "year", "protest_count", "conflict_count", "repression_count",
            "total_events", "avg_goldstein", "avg_tone", "num_mentions"]
    out = final[cols].sort_values(["country_code", "year"]).reset_index(drop=True)
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {len(out)} country-years to {OUTPUT_CSV}")
    print(f"Countries: {out['country_code'].nunique()}, Years: {out['year'].min()}-{out['year'].max()}")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--merge-only", action="store_true", help="Skip download, merge existing cache")
    args = parser.parse_args()

    os.makedirs(CACHE_DIR, exist_ok=True)

    if args.merge_only:
        merge_cache()
        return

    stamps = build_stamps()
    todo = [s for s in stamps if not os.path.exists(os.path.join(CACHE_DIR, f"{s}.parquet"))]
    print(f"GDELT download: {len(stamps)} total stamps, {len(todo)} remaining "
          f"({len(stamps) - len(todo)} cached). Workers={WORKERS}.")

    if not todo:
        print("Nothing to download. Merging cache.")
        merge_cache()
        return

    completed = 0
    errors = []
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(process_stamp, s): s for s in todo}
        for fut in as_completed(futures):
            stamp = futures[fut]
            try:
                _, status, info = fut.result()
            except Exception as e:
                status, info = "exception", str(e)
            completed += 1
            if status not in ("ok", "cached", "empty"):
                errors.append((stamp, status, info))
            if completed % 50 == 0 or completed == len(todo):
                print(f"  [{completed}/{len(todo)}] last={stamp} status={status} errors={len(errors)}")

    if errors:
        print(f"\n{len(errors)} failures. First 10:")
        for s, st, info in errors[:10]:
            print(f"  {s}: {st} {info or ''}")
        print("Re-run the script to retry failed stamps (cached stamps are skipped).")

    merge_cache()


if __name__ == "__main__":
    main()
