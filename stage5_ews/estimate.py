import sys
import os
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

WINDOW = 8
MIN_WINDOW = 5
EWS_THRESHOLD_VAR = 0.7
EWS_THRESHOLD_AR1 = 0.7
LEAD_YEARS = 5

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


def load_residuals():
    base = os.path.dirname(os.path.abspath(__file__))
    scores = pd.read_csv(os.path.join(base, "..", "stage4_nscm", "contagion_scores.csv"))
    factors = pd.read_csv(os.path.join(base, "..", "stage1_factors", "country_year_factors.csv"))

    spillover_cols = [c for c in scores.columns if c.startswith("spillover_state_")]
    factor_cols = ["factor_1", "factor_2", "factor_3", "factor_4"]

    merged = scores.merge(factors[["country_text_id", "year"] + factor_cols],
                          on=["country_text_id", "year"], how="inner")

    for k, fcol in enumerate(factor_cols):
        scol = f"spillover_state_{k}" if f"spillover_state_{k}" in merged.columns else None
        if scol:
            merged[f"domestic_resid_{k+1}"] = merged[fcol] - merged[scol] * merged["contagion_score"]
        else:
            merged[f"domestic_resid_{k+1}"] = merged[fcol]

    merged["composite_resid"] = sum(
        merged[f"domestic_resid_{k+1}"] for k in range(4)
    ) / 4.0

    return merged


def compute_ews(series, window=WINDOW, min_window=MIN_WINDOW):
    n = len(series)
    variances = np.full(n, np.nan)
    ar1s = np.full(n, np.nan)

    for t in range(min_window, n):
        start = max(0, t - window)
        chunk = series[start:t + 1]
        if len(chunk) < min_window:
            continue

        variances[t] = np.var(chunk, ddof=1)

        if len(chunk) >= 3 and np.std(chunk) > 1e-10:
            ar1s[t] = np.corrcoef(chunk[:-1], chunk[1:])[0, 1]

    return variances, ar1s


def kendall_trend(series):
    valid = ~np.isnan(series)
    if valid.sum() < 5:
        return 0.0, 1.0
    x = np.arange(len(series))[valid]
    y = series[valid]
    tau, p = stats.kendalltau(x, y)
    return tau, p


def run_ews():
    print("=== Stage 5: Early Warning Signals (Critical Slowing Down) ===\n")

    df = load_residuals()
    resid_cols = [f"domestic_resid_{k+1}" for k in range(4)] + ["composite_resid"]
    countries = sorted(df["country_name"].unique())
    print(f"Countries: {len(countries)}")
    print(f"Residual columns: {resid_cols}")
    print(f"Window: {WINDOW} years, Min: {MIN_WINDOW}")

    all_ews = []

    for country in countries:
        cdf = df[df["country_name"] == country].sort_values("year")
        if len(cdf) < MIN_WINDOW + 2:
            continue

        years = cdf["year"].values

        for rcol in resid_cols:
            series = cdf[rcol].values
            variances, ar1s = compute_ews(series, WINDOW, MIN_WINDOW)

            var_tau, var_p = kendall_trend(variances)
            ar1_tau, ar1_p = kendall_trend(ar1s)

            for t in range(len(years)):
                all_ews.append({
                    "country_name": country,
                    "country_text_id": cdf["country_text_id"].iloc[0],
                    "year": int(years[t]),
                    "residual_type": rcol,
                    "rolling_var": variances[t] if not np.isnan(variances[t]) else None,
                    "rolling_ar1": ar1s[t] if not np.isnan(ar1s[t]) else None,
                    "var_trend_tau": var_tau,
                    "var_trend_p": var_p,
                    "ar1_trend_tau": ar1_tau,
                    "ar1_trend_p": ar1_p,
                })

    ews_df = pd.DataFrame(all_ews)

    composite = ews_df[ews_df["residual_type"] == "composite_resid"].copy()
    composite = composite.dropna(subset=["rolling_var", "rolling_ar1"])

    if len(composite) > 0:
        var_q = composite["rolling_var"].quantile(EWS_THRESHOLD_VAR)
        ar1_q = composite["rolling_ar1"].quantile(EWS_THRESHOLD_AR1)
        composite["high_var"] = composite["rolling_var"] > var_q
        composite["high_ar1"] = composite["rolling_ar1"] > ar1_q
        composite["ews_alert"] = composite["high_var"] & composite["high_ar1"]
    else:
        composite["ews_alert"] = False

    output_dir = os.path.dirname(os.path.abspath(__file__))
    ews_df.to_csv(os.path.join(output_dir, "ews_signals.csv"), index=False)
    composite.to_csv(os.path.join(output_dir, "ews_composite.csv"), index=False)
    print(f"Saved {len(ews_df)} EWS records, {len(composite)} composite records")

    print(f"\n=== Validation against known episodes ===\n")
    hits = 0
    total = 0
    for country, info in KNOWN_EPISODES.items():
        onset = info["onset"]
        pre_window = composite[
            (composite["country_name"] == country) &
            (composite["year"] >= onset - LEAD_YEARS) &
            (composite["year"] < onset)
        ]

        if len(pre_window) == 0:
            print(f"  {country} ({info['type']} {onset}): NO DATA in pre-onset window")
            continue

        total += 1
        any_alert = pre_window["ews_alert"].any()
        var_rising = pre_window["var_trend_tau"].iloc[-1] > 0
        ar1_rising = pre_window["ar1_trend_tau"].iloc[-1] > 0

        max_var = pre_window["rolling_var"].max()
        max_ar1 = pre_window["rolling_ar1"].max()
        var_pct = (composite["rolling_var"] < max_var).mean() if max_var == max_var else 0
        ar1_pct = (composite["rolling_ar1"] < max_ar1).mean() if max_ar1 == max_ar1 else 0

        detected = any_alert or (var_rising and ar1_rising)
        if detected:
            hits += 1

        status = "DETECTED" if detected else "MISSED"
        print(f"  {country} ({info['type']} {onset}): {status}")
        print(f"    Pre-onset var percentile: {var_pct:.1%}, AR1 percentile: {ar1_pct:.1%}")
        print(f"    Var trend: tau={pre_window['var_trend_tau'].iloc[-1]:.3f}, "
              f"AR1 trend: tau={pre_window['ar1_trend_tau'].iloc[-1]:.3f}")
        if any_alert:
            alert_years = pre_window[pre_window["ews_alert"]]["year"].tolist()
            print(f"    Alert years: {alert_years}")

    if total > 0:
        print(f"\n  Detection rate: {hits}/{total} ({hits/total:.0%})")

    print(f"\n=== Current alerts (2023-2025) ===")
    recent_alerts = composite[
        (composite["year"] >= 2023) & (composite["ews_alert"] == True)
    ].sort_values("year")

    if len(recent_alerts) == 0:
        print("  No active alerts")
    else:
        for _, row in recent_alerts.iterrows():
            print(f"  {row['country_name']} ({int(row['year'])}): "
                  f"var={row['rolling_var']:.4f} (>{var_q:.4f}), "
                  f"AR1={row['rolling_ar1']:.3f} (>{ar1_q:.3f})")

    print(f"\n=== Countries with strongest rising CSD (all time) ===")
    country_trends = composite.groupby("country_name").agg(
        var_tau=("var_trend_tau", "last"),
        ar1_tau=("ar1_trend_tau", "last"),
    ).reset_index()
    country_trends["combined_tau"] = country_trends["var_tau"] + country_trends["ar1_tau"]
    top_csd = country_trends.sort_values("combined_tau", ascending=False).head(15)
    for _, row in top_csd.iterrows():
        print(f"  {row['country_name']}: var_tau={row['var_tau']:.3f}, ar1_tau={row['ar1_tau']:.3f}")

    print(f"\n=== Case study EWS trajectories ===")
    for country in ["Hungary", "Türkiye", "Poland", "United States of America"]:
        sub = composite[composite["country_name"] == country].sort_values("year").tail(10)
        if len(sub) == 0:
            continue
        print(f"\n{country}:")
        for _, r in sub.iterrows():
            alert = " *** ALERT" if r.get("ews_alert", False) else ""
            print(f"  {int(r['year'])}: var={r['rolling_var']:.4f}, AR1={r['rolling_ar1']:.3f}{alert}")

    return ews_df, composite


if __name__ == "__main__":
    ews_df, composite = run_ews()
