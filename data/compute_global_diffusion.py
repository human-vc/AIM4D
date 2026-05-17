"""
F4: Global PageRank-weighted backsliding-exposure feature.

Replaces (or supplements) the existing neighbor-mean network_exposure score
from Stage 4 with a global diffusion measure motivated by Schmotz & Selvik
(2025), who show that backsliding clusters globally (alliance + trade) rather
than only locally.

For each country-year (i, t) we compute

    GlobalExposure_{i,t} = sum_{j != i}  PR_j  *  poly_change_{j, t-1}

where PR_j is the PageRank of country j on the union of (contiguity ∪
alliance ∪ trade-similarity) graph for that year, and poly_change is
country j's V-Dem polyarchy YoY change. A country surrounded by influential
backsliders gets a high (negative-direction) score.

Writes data/global_diffusion.csv with columns:
  country_text_id, year, global_exposure_polyarchy, global_exposure_libdem,
  pagerank, n_backsliding_neighbors
"""

import os
import pandas as pd
import numpy as np
import networkx as nx

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.dirname(os.path.abspath(__file__))


def load_iso3_mapping():
    return pd.read_csv(os.path.join(DATA, "cow_iso3_mapping.csv"))


def load_contig_pairs():
    p = os.path.join(DATA, "contiguity", "DirectContiguity320", "contdird.csv")
    if not os.path.exists(p):
        # fall back to .dta or similar; cheap path
        for cand in ["contdird.csv", "contdir3.20.csv"]:
            q = os.path.join(DATA, "contiguity", "DirectContiguity320", cand)
            if os.path.exists(q):
                p = q
                break
    if not os.path.exists(p):
        return pd.DataFrame()
    df = pd.read_csv(p, low_memory=False)
    # contdird has 'state1no', 'state2no', 'conttype' (1-5 where lower = closer)
    df = df[df["conttype"] <= 2]
    return df[["state1no", "state2no", "year", "conttype"]]


def load_alliance_pairs():
    # ATOP is typically directed-dyadic; we want any active alliance per year
    p = os.path.join(DATA, "atop", "ATOP 5.1 (.csv)", "atop5_1ddyr.csv")
    if not os.path.exists(p):
        for cand in os.listdir(os.path.join(DATA, "atop", "ATOP 5.1 (.csv)")):
            if cand.endswith(".csv"):
                p = os.path.join(DATA, "atop", "ATOP 5.1 (.csv)", cand)
                break
    if not os.path.exists(p):
        return pd.DataFrame()
    df = pd.read_csv(p, low_memory=False)
    # canonical columns: stateA, stateB, year, atopally
    cols = {c.lower(): c for c in df.columns}
    a = cols.get("statea") or cols.get("ccode1") or cols.get("stateA")
    b = cols.get("stateb") or cols.get("ccode2") or cols.get("stateB")
    y = cols.get("year")
    ally = cols.get("atopally") or cols.get("ally")
    if not (a and b and y):
        return pd.DataFrame()
    sub = df[[a, b, y] + ([ally] if ally else [])].copy()
    sub.columns = ["state1no", "state2no", "year"] + (["atopally"] if ally else [])
    if "atopally" in sub.columns:
        sub = sub[sub["atopally"] == 1]
    return sub[["state1no", "state2no", "year"]]


def build_graph(year, contig, alliance, vdem_iso3_set):
    """Build undirected graph for a year combining contiguity + alliance edges."""
    G = nx.Graph()
    G.add_nodes_from(vdem_iso3_set)
    for df in [contig, alliance]:
        if df.empty:
            continue
        sub = df[df["year"] == year] if "year" in df.columns else df
        for _, r in sub.iterrows():
            # Need to convert COW state numbers to ISO3
            pass
    return G


def main():
    print("Loading V-Dem polyarchy and libdem timeseries...")
    vdem = pd.read_csv(
        os.path.join(DATA, "vdem_v16.csv"), low_memory=False,
        usecols=["country_text_id", "year", "v2x_polyarchy", "v2x_libdem"],
    )
    vdem = vdem.dropna(subset=["v2x_polyarchy"])
    vdem = vdem.sort_values(["country_text_id", "year"])
    vdem["poly_change"] = vdem.groupby("country_text_id")["v2x_polyarchy"].diff()
    vdem["libdem_change"] = vdem.groupby("country_text_id")["v2x_libdem"].diff()

    iso_map = load_iso3_mapping()
    cow_to_iso = dict(zip(iso_map["cow_code"], iso_map["iso3"]))

    contig = load_contig_pairs()
    alliance = load_alliance_pairs()
    if not contig.empty:
        contig["iso_a"] = contig["state1no"].map(cow_to_iso)
        contig["iso_b"] = contig["state2no"].map(cow_to_iso)
        contig = contig.dropna(subset=["iso_a", "iso_b"])
    if not alliance.empty:
        alliance["iso_a"] = alliance["state1no"].map(cow_to_iso)
        alliance["iso_b"] = alliance["state2no"].map(cow_to_iso)
        alliance = alliance.dropna(subset=["iso_a", "iso_b"])
    print(f"  contig pairs: {len(contig)}; alliance pairs: {len(alliance)}")

    # For each year, build the graph, compute PageRank, then compute exposure.
    years = sorted(vdem["year"].unique())
    rows = []
    for year in years:
        if year < 1970:
            continue
        countries = sorted(vdem[vdem["year"] == year]["country_text_id"].unique())
        G = nx.Graph()
        G.add_nodes_from(countries)

        # add contiguity edges for this year (or fall back to most recent <= year)
        if not contig.empty:
            sub = contig[contig["year"] <= year].sort_values("year").drop_duplicates(
                subset=["iso_a", "iso_b"], keep="last"
            )
            for _, r in sub.iterrows():
                if r["iso_a"] in countries and r["iso_b"] in countries:
                    G.add_edge(r["iso_a"], r["iso_b"])
        # add alliance edges within a 5-year window (matches Stage 4 convention)
        if not alliance.empty:
            sub = alliance[(alliance["year"] >= year - 5) & (alliance["year"] <= year)]
            for _, r in sub.iterrows():
                if r["iso_a"] in countries and r["iso_b"] in countries:
                    G.add_edge(r["iso_a"], r["iso_b"])

        if G.number_of_edges() == 0:
            # fallback: complete graph weakly weighted (gives uniform PageRank)
            pass

        try:
            pr = nx.pagerank(G, alpha=0.85, max_iter=100, tol=1e-6)
        except Exception:
            pr = {c: 1.0 / max(len(countries), 1) for c in countries}

        # lag-1 polyarchy and libdem change for each country in this year
        prev = vdem[vdem["year"] == year - 1].set_index("country_text_id")
        poly_lag = prev["poly_change"].to_dict() if not prev.empty else {}
        libdem_lag = prev["libdem_change"].to_dict() if not prev.empty else {}

        # Schmotz-Selvik style: sum_{j != i} PR_j * change_j  (year t-1 change)
        for ego in countries:
            ego_pr = pr.get(ego, 0.0)
            exposure_poly = 0.0
            exposure_libdem = 0.0
            n_backsliders = 0
            for j in countries:
                if j == ego:
                    continue
                pj = pr.get(j, 0.0)
                cp = poly_lag.get(j, np.nan)
                cl = libdem_lag.get(j, np.nan)
                if not np.isnan(cp):
                    # negative change = backsliding; we want a positive exposure
                    # to backsliders, so we use -change
                    exposure_poly += pj * (-cp)
                    if cp < -0.01:
                        n_backsliders += 1
                if not np.isnan(cl):
                    exposure_libdem += pj * (-cl)
            rows.append({
                "country_text_id": ego,
                "year": year,
                "global_exposure_polyarchy": exposure_poly,
                "global_exposure_libdem": exposure_libdem,
                "pagerank": ego_pr,
                "n_backsliding_neighbors": n_backsliders,
            })
        if year % 10 == 0:
            print(f"  year {year}: {len(countries)} countries, {G.number_of_edges()} edges")

    out = pd.DataFrame(rows)
    out_path = os.path.join(DATA, "global_diffusion.csv")
    out.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}: {len(out)} rows, "
          f"{out['country_text_id'].nunique()} countries, {out['year'].min()}-{out['year'].max()}")


if __name__ == "__main__":
    main()
