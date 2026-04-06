import sys
import os
import numpy as np
import pandas as pd
from scipy import stats
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

WINDOW = 8
MIN_WINDOW = 5
EWS_THRESHOLD_VAR = 0.90
EWS_THRESHOLD_AR1 = 0.85
LEAD_YEARS = 5
TRAIN_CUTOFF = 2019

KNOWN_EPISODES = {
    "Hungary": {"onset": 2010, "peak": 2018, "type": "backsliding"},
    "Türkiye": {"onset": 2013, "peak": 2017, "type": "backsliding"},
    "Poland": {"onset": 2015, "peak": 2019, "type": "backsliding"},
    "Venezuela": {"onset": 2005, "peak": 2013, "type": "backsliding"},
    "Tunisia": {"onset": 2021, "peak": 2023, "type": "backsliding"},
    "Burma/Myanmar": {"onset": 2021, "peak": 2022, "type": "coup"},
    "Mali": {"onset": 2020, "peak": 2021, "type": "coup"},
    "Burkina Faso": {"onset": 2022, "peak": 2022, "type": "coup"},
}


def load_nscm_residuals():
    base = os.path.dirname(os.path.abspath(__file__))

    sys.path.insert(0, os.path.join(base, "..", "stage4_nscm"))
    from estimate import (load_all_data, build_spatial_edges, build_spatiotemporal_graph,
                          INETARNet, FACTOR_COLS, BETA_COLS, STATE_COLS)

    df, mapping = load_all_data()
    feature_cols = FACTOR_COLS + BETA_COLS + ["gdp_pc", "urbanization"]

    years_all = sorted(df["year"].unique())
    years_use = [y for y in years_all if y >= 1990]

    complete = df.groupby("country_text_id").apply(
        lambda g: g[g["year"].isin(years_use)].dropna(subset=feature_cols + STATE_COLS)["year"].nunique()
    )
    countries_iso3 = sorted(complete[complete >= len(years_use) * 0.8].index.tolist())

    contig_pairs, alliance_by_year = build_spatial_edges(mapping, countries_iso3)

    (x, y, edge_index, spatial_ei, temporal_ei,
     mask_train, mask_test, node_country, node_year, N, T) = \
        build_spatiotemporal_graph(df, countries_iso3, years_use, contig_pairs, alliance_by_year, feature_cols)

    in_dim = x.shape[1]
    model = INETARNet(in_dim)

    model_path = os.path.join(base, "..", "stage4_nscm", "model.pt")
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, weights_only=True))
    else:
        from estimate import train_model
        model = train_model(x, y, edge_index, mask_train, mask_test, in_dim)

    model.eval()
    with torch.no_grad():
        full_ei = torch.cat([spatial_ei, temporal_ei], dim=1)
        y_pred_full, y_pred_local, _, _ = model(x, full_ei)
        _, h_ego = model.encode(x, full_ei)
        y_pred_domestic = F.softmax(model.local_logits(h_ego), dim=-1)

        resid_full = (y - y_pred_full).numpy()
        resid_domestic = (y - y_pred_domestic).numpy()

    cname_map = df.drop_duplicates("country_text_id").set_index("country_text_id")["country_name"]

    rows = []
    for nid in range(len(node_country)):
        row = {
            "country_text_id": node_country[nid],
            "country_name": cname_map.get(node_country[nid], node_country[nid]),
            "year": node_year[nid],
        }
        for k in range(resid_domestic.shape[1]):
            row[f"resid_domestic_{k}"] = resid_domestic[nid, k]
            row[f"resid_full_{k}"] = resid_full[nid, k]
        row["resid_composite"] = np.mean(np.abs(resid_domestic[nid]))
        rows.append(row)

    return pd.DataFrame(rows)


def load_factor_residuals():
    base = os.path.dirname(os.path.abspath(__file__))
    scores = pd.read_csv(os.path.join(base, "..", "stage4_nscm", "contagion_scores.csv"))
    factors = pd.read_csv(os.path.join(base, "..", "stage1_factors", "country_year_factors.csv"))
    states = pd.read_csv(os.path.join(base, "..", "stage3_msvar", "country_year_states.csv"))

    state_cols = [c for c in states.columns if c.startswith("prob_state_")]
    merged = factors.merge(scores[["country_text_id", "year", "contagion_score", "domestic_score"]],
                           on=["country_text_id", "year"], how="inner")
    merged = merged.merge(states[["country_name", "year"] + state_cols],
                          on=["country_name", "year"], how="inner")

    factor_cols = ["factor_1", "factor_2", "factor_3", "factor_4"]
    for k, fc in enumerate(factor_cols):
        country_mean = merged.groupby("country_text_id")[fc].transform("mean")
        merged[f"resid_domestic_{k}"] = (merged[fc] - country_mean) * merged["domestic_score"]

    merged["resid_composite"] = sum(
        merged[f"resid_domestic_{k}"].abs() for k in range(4)
    ) / 4.0

    return merged


def compute_rolling_ews(series, window=WINDOW, min_window=MIN_WINDOW):
    n = len(series)
    variances = np.full(n, np.nan)
    ar1s = np.full(n, np.nan)
    var_trends = np.full(n, np.nan)
    ar1_trends = np.full(n, np.nan)

    for t in range(min_window, n):
        start = max(0, t - window)
        chunk = series[start:t + 1]
        if len(chunk) < min_window:
            continue

        variances[t] = np.var(chunk, ddof=1) if len(chunk) > 1 else 0

        if len(chunk) >= 3 and np.std(chunk) > 1e-10:
            ar1s[t] = np.corrcoef(chunk[:-1], chunk[1:])[0, 1]

    trend_window = min(10, n)
    for t in range(min_window + 2, n):
        start = max(min_window, t - trend_window)
        v_slice = variances[start:t + 1]
        a_slice = ar1s[start:t + 1]

        v_valid = ~np.isnan(v_slice)
        if v_valid.sum() >= 4:
            tau, _ = stats.kendalltau(np.arange(v_valid.sum()), v_slice[v_valid])
            var_trends[t] = tau

        a_valid = ~np.isnan(a_slice)
        if a_valid.sum() >= 4:
            tau, _ = stats.kendalltau(np.arange(a_valid.sum()), a_slice[a_valid])
            ar1_trends[t] = tau

    return variances, ar1s, var_trends, ar1_trends


def run_ews():
    print("=== Stage 5: Early Warning Signals (Critical Slowing Down) ===\n")

    print("Loading domestic residuals from factor-based decomposition...")
    df = load_factor_residuals()
    countries = sorted(df["country_name"].unique())
    print(f"Countries: {len(countries)}")

    all_ews = []
    for country in countries:
        cdf = df[df["country_name"] == country].sort_values("year")
        if len(cdf) < MIN_WINDOW + 2:
            continue

        years = cdf["year"].values
        series = cdf["resid_composite"].values

        variances, ar1s, var_trends, ar1_trends = compute_rolling_ews(series)

        for t in range(len(years)):
            all_ews.append({
                "country_name": country,
                "country_text_id": cdf["country_text_id"].iloc[0],
                "year": int(years[t]),
                "rolling_var": variances[t],
                "rolling_ar1": ar1s[t],
                "var_trend": var_trends[t],
                "ar1_trend": ar1_trends[t],
            })

    ews_df = pd.DataFrame(all_ews).dropna(subset=["rolling_var", "rolling_ar1"])

    train_data = ews_df[ews_df["year"] <= TRAIN_CUTOFF]
    var_threshold = train_data["rolling_var"].quantile(EWS_THRESHOLD_VAR)
    ar1_threshold = train_data["rolling_ar1"].quantile(EWS_THRESHOLD_AR1)
    print(f"\nThresholds (from training period ≤{TRAIN_CUTOFF}):")
    print(f"  Variance > {var_threshold:.5f} (p{int(EWS_THRESHOLD_VAR*100)})")
    print(f"  AR(1) > {ar1_threshold:.3f} (p{int(EWS_THRESHOLD_AR1*100)})")

    ews_df["high_var"] = ews_df["rolling_var"] > var_threshold
    ews_df["high_ar1"] = ews_df["rolling_ar1"] > ar1_threshold
    ews_df["rising_var"] = ews_df["var_trend"] > 0.2
    ews_df["rising_ar1"] = ews_df["ar1_trend"] > 0.1

    ews_df["ews_alert"] = (
        (ews_df["high_var"] & ews_df["high_ar1"]) |
        (ews_df["high_var"] & ews_df["rising_ar1"]) |
        (ews_df["high_ar1"] & ews_df["rising_var"])
    )

    output_dir = os.path.dirname(os.path.abspath(__file__))
    ews_df.to_csv(os.path.join(output_dir, "ews_signals.csv"), index=False)
    print(f"Saved {len(ews_df)} EWS records")

    print(f"\n{'='*60}")
    print(f"Validation against known episodes")
    print(f"{'='*60}\n")

    hits = 0
    total = 0
    for country, info in KNOWN_EPISODES.items():
        onset = info["onset"]
        pre = ews_df[
            (ews_df["country_name"] == country) &
            (ews_df["year"] >= onset - LEAD_YEARS) &
            (ews_df["year"] < onset)
        ]

        if len(pre) == 0:
            print(f"  {country} ({info['type']} {onset}): NO DATA")
            continue

        total += 1
        any_alert = pre["ews_alert"].any()

        if any_alert:
            hits += 1
            alert_years = sorted(pre[pre["ews_alert"]]["year"].tolist())
            lead_time = onset - alert_years[0]
            print(f"  {country} ({info['type']} {onset}): DETECTED")
            print(f"    First alert: {alert_years[0]} ({lead_time}yr lead)")
            print(f"    Alert years: {alert_years}")
        else:
            max_var_pct = (train_data["rolling_var"] < pre["rolling_var"].max()).mean()
            max_ar1_pct = (train_data["rolling_ar1"] < pre["rolling_ar1"].max()).mean()
            print(f"  {country} ({info['type']} {onset}): MISSED")
            print(f"    Max var pct: {max_var_pct:.1%}, Max AR1 pct: {max_ar1_pct:.1%}")

    sensitivity = hits / total if total > 0 else 0
    print(f"\n  Sensitivity: {hits}/{total} ({sensitivity:.0%})")

    print(f"\n{'='*60}")
    print(f"False positive analysis")
    print(f"{'='*60}\n")

    known_countries = set(KNOWN_EPISODES.keys())
    known_onset_years = {c: info["onset"] for c, info in KNOWN_EPISODES.items()}

    all_alerts = ews_df[ews_df["ews_alert"]].copy()
    tp_alerts = all_alerts[
        all_alerts.apply(lambda r: r["country_name"] in known_countries and
                         known_onset_years.get(r["country_name"], 9999) - LEAD_YEARS <= r["year"] < known_onset_years.get(r["country_name"], 9999),
                         axis=1)
    ]
    fp_alerts = all_alerts[~all_alerts.index.isin(tp_alerts.index)]

    total_country_years = len(ews_df)
    alert_rate = len(all_alerts) / total_country_years
    fp_rate = len(fp_alerts) / total_country_years

    print(f"  Total alerts: {len(all_alerts)} / {total_country_years} country-years ({alert_rate:.1%})")
    print(f"  True positives: {len(tp_alerts)}")
    print(f"  False positives: {len(fp_alerts)} ({fp_rate:.1%} FP rate)")

    if sensitivity > 0 and fp_rate > 0:
        precision = len(tp_alerts) / len(all_alerts) if len(all_alerts) > 0 else 0
        print(f"  Precision: {precision:.1%}")
        print(f"  F1 score: {2 * precision * sensitivity / (precision + sensitivity):.3f}" if (precision + sensitivity) > 0 else "")

    stable_democracies = ["Denmark", "Sweden", "Norway", "Switzerland", "Finland",
                          "Germany", "Canada", "New Zealand", "Uruguay", "Belgium"]
    stable_alerts = fp_alerts[fp_alerts["country_name"].isin(stable_democracies)]
    if len(stable_alerts) > 0:
        print(f"\n  False alerts in stable democracies:")
        for _, r in stable_alerts.iterrows():
            print(f"    {r['country_name']} ({int(r['year'])}): var={r['rolling_var']:.5f}, AR1={r['rolling_ar1']:.3f}")
    else:
        print(f"\n  No false alerts in stable democracies")

    print(f"\n{'='*60}")
    print(f"Current alerts (post-{TRAIN_CUTOFF})")
    print(f"{'='*60}\n")

    recent = ews_df[(ews_df["year"] > TRAIN_CUTOFF) & (ews_df["ews_alert"])].sort_values(["year", "country_name"])
    if len(recent) == 0:
        print("  No active alerts")
    else:
        for year in sorted(recent["year"].unique()):
            yr_alerts = recent[recent["year"] == year]
            countries_str = ", ".join(yr_alerts["country_name"].tolist())
            print(f"  {int(year)} ({len(yr_alerts)} alerts): {countries_str}")

    print(f"\n{'='*60}")
    print(f"Case study EWS trajectories")
    print(f"{'='*60}\n")

    for country in ["Hungary", "Türkiye", "Poland", "United States of America", "Denmark"]:
        sub = ews_df[ews_df["country_name"] == country].sort_values("year").tail(10)
        if len(sub) == 0:
            continue
        print(f"{country}:")
        for _, r in sub.iterrows():
            alert = " *** ALERT" if r["ews_alert"] else ""
            oos = " (OOS)" if r["year"] > TRAIN_CUTOFF else ""
            print(f"  {int(r['year'])}{oos}: var={r['rolling_var']:.5f}, AR1={r['rolling_ar1']:.3f}, "
                  f"Δvar={r['var_trend']:.3f}, ΔAR1={r['ar1_trend']:.3f}{alert}")
        print()

    return ews_df


if __name__ == "__main__":
    ews_df = run_ews()
