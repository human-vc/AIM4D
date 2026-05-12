"""
Bootstrap 95% confidence intervals for the paper's headline metrics.

Loads stage5_ews/ews_signals.csv (must have combined_risk + label populated)
and V-Dem v16 for the FH/Polity outcome cross-checks. Resamples 1,000x with
replacement; reports point estimate and percentile [2.5, 97.5] CI.
Outputs robustness/bootstrap_cis.csv.
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

OUT = os.path.dirname(os.path.abspath(__file__))
RNG = np.random.default_rng(42)
N_BOOT = 1000


def bootstrap_auc(y, s, fn=roc_auc_score, n_boot=N_BOOT):
    y = np.asarray(y)
    s = np.asarray(s)
    n = len(y)
    out = []
    for _ in range(n_boot):
        idx = RNG.integers(0, n, n)
        yy, ss = y[idx], s[idx]
        if yy.sum() == 0 or yy.sum() == len(yy):
            continue
        out.append(fn(yy, ss))
    if not out:
        return np.nan, np.nan, np.nan
    arr = np.array(out)
    return float(arr.mean()), float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def wilson_ci(k, n, alpha=0.05):
    if n == 0:
        return 0.0, 0.0
    from scipy.stats import beta
    lo = beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hi = beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return float(lo), float(hi)


def main():
    rows = []

    print("=" * 70)
    print("BOOTSTRAP 95% CIs ON HEADLINE METRICS")
    print("=" * 70)

    ews_path = os.path.join(OUT, "..", "stage5_ews", "ews_signals.csv")
    ews = pd.read_csv(ews_path)
    if "combined_risk" not in ews.columns:
        raise RuntimeError("ews_signals.csv missing combined_risk column; rerun stage 5")
    if "label" not in ews.columns:
        raise RuntimeError("ews_signals.csv missing label column")

    valid = ews.dropna(subset=["combined_risk", "label"])
    y = valid["label"].astype(int).values
    s = valid["combined_risk"].values

    print(f"\nFull panel: n={len(valid)}, n_positive={int(y.sum())} ({y.mean():.1%})\n")

    auc, lo, hi = bootstrap_auc(y, s, roc_auc_score)
    print(f"  AUC-ROC (in-sample):  {auc:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
    rows.append({"metric": "auc_roc_in_sample", "point": auc, "ci_low": lo, "ci_high": hi, "n": len(y)})

    auc_pr, lo, hi = bootstrap_auc(y, s, average_precision_score)
    print(f"  AUC-PR  (in-sample):  {auc_pr:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
    rows.append({"metric": "auc_pr_in_sample", "point": auc_pr, "ci_low": lo, "ci_high": hi, "n": len(y)})

    cutoff = 2017
    oos = valid[valid["year"] > cutoff]
    if len(oos) > 0 and oos["label"].sum() > 1:
        auc, lo, hi = bootstrap_auc(oos["label"].astype(int).values, oos["combined_risk"].values, roc_auc_score)
        print(f"  AUC-ROC (2017 OOS):   {auc:.3f}  95% CI [{lo:.3f}, {hi:.3f}]  n={len(oos)}")
        rows.append({"metric": "auc_roc_strict_oos_2017", "point": auc, "ci_low": lo, "ci_high": hi, "n": len(oos)})

    try:
        from stage5_ews.estimate import KNOWN_EPISODES, LEAD_YEARS
    except Exception:
        LEAD_YEARS = 5
        KNOWN_EPISODES = {}

    if KNOWN_EPISODES:
        episodes = []
        for country, info in KNOWN_EPISODES.items():
            onset = info["onset"]
            sub = valid[(valid["country_name"] == country)
                        & (valid["year"] >= onset - LEAD_YEARS)
                        & (valid["year"] < onset)]
            if len(sub) == 0:
                continue
            max_risk = sub["combined_risk"].max()
            train_risks = valid[valid["year"] <= 2021]["combined_risk"]
            thresh = train_risks.quantile(0.80)
            episodes.append({
                "country": country,
                "type": info["type"],
                "max_risk": float(max_risk),
                "detected": int(max_risk >= thresh),
            })

        if episodes:
            ep_df = pd.DataFrame(episodes)
            for stratum_name, mask in [
                ("in_sample_watch_all", np.ones(len(ep_df), dtype=bool)),
                ("in_sample_watch_backsliding", ep_df["type"].eq("backsliding").values),
                ("in_sample_watch_coup", ep_df["type"].eq("coup").values),
            ]:
                if mask.sum() == 0:
                    continue
                hits = int(ep_df[mask]["detected"].sum())
                n = int(mask.sum())
                point = hits / n
                lo, hi = wilson_ci(hits, n)
                print(f"  {stratum_name:<28}: {hits}/{n} ({point:.0%})  95% CI [{lo:.0%}, {hi:.0%}]")
                rows.append({"metric": stratum_name, "point": point, "ci_low": lo, "ci_high": hi, "n": n})

    loeo_path = os.path.join(OUT, "..", "stage5_ews", "loeo_results.csv")
    if os.path.exists(loeo_path):
        loeo = pd.read_csv(loeo_path)
        loeo_strata = [
            ("loeo_watch_all", loeo["detected_watch"], np.ones(len(loeo), dtype=bool)),
            ("loeo_watch_backsliding", loeo["detected_watch"], loeo["type"].eq("backsliding").values),
            ("loeo_watch_coup", loeo["detected_watch"], loeo["type"].eq("coup").values),
        ]
        for stratum_name, det_col, mask in loeo_strata:
            if mask.sum() == 0:
                continue
            hits = int(det_col[mask].sum())
            n = int(mask.sum())
            point = hits / n
            lo, hi = wilson_ci(hits, n)
            print(f"  {stratum_name:<28}: {hits}/{n} ({point:.0%})  95% CI [{lo:.0%}, {hi:.0%}]")
            rows.append({"metric": stratum_name, "point": point, "ci_low": lo, "ci_high": hi, "n": n})
    else:
        print(f"\n  [note] {loeo_path} not present — true LOEO CIs skipped.")
        print(f"         Rerun stage 5 after the patch below to dump LOEO results.")

    vdem_path = os.path.join(OUT, "..", "data", "vdem_v16.csv")
    fh_cols = ["country_text_id", "year", "e_fh_pr"]
    fh = pd.read_csv(vdem_path, usecols=fh_cols, low_memory=False).dropna()
    fh["e_fh_pr"] = fh["e_fh_pr"].astype(float)
    fh = fh.sort_values(["country_text_id", "year"])
    fh["fh_3yr_change"] = fh.groupby("country_text_id")["e_fh_pr"].diff(3)
    fh["fh_decline_3yr_2pt"] = (fh["fh_3yr_change"] >= 2).astype(int)

    merged_fh = valid.merge(fh, on=["country_text_id", "year"], how="inner").dropna(subset=["fh_decline_3yr_2pt"])
    if merged_fh["fh_decline_3yr_2pt"].sum() > 1:
        auc, lo, hi = bootstrap_auc(merged_fh["fh_decline_3yr_2pt"].astype(int).values,
                                     merged_fh["combined_risk"].values, roc_auc_score)
        print(f"  FH 3yr decline AUC:   {auc:.3f}  95% CI [{lo:.3f}, {hi:.3f}]  n={len(merged_fh)}")
        rows.append({"metric": "auc_fh_3yr_decline_2pt", "point": auc, "ci_low": lo, "ci_high": hi, "n": len(merged_fh)})

    polity = pd.read_csv(vdem_path, usecols=["country_text_id", "year", "e_polity2"], low_memory=False).dropna()
    polity["e_polity2"] = polity["e_polity2"].astype(float)
    polity = polity.sort_values(["country_text_id", "year"])
    polity["polity_3yr_change"] = polity.groupby("country_text_id")["e_polity2"].diff(3)
    polity["polity_decline_3yr_3pt"] = (polity["polity_3yr_change"] <= -3).astype(int)

    merged_polity = valid.merge(polity, on=["country_text_id", "year"], how="inner").dropna(subset=["polity_decline_3yr_3pt"])
    if merged_polity["polity_decline_3yr_3pt"].sum() > 1:
        auc, lo, hi = bootstrap_auc(merged_polity["polity_decline_3yr_3pt"].astype(int).values,
                                     merged_polity["combined_risk"].values, roc_auc_score)
        print(f"  Polity 3yr decline:   {auc:.3f}  95% CI [{lo:.3f}, {hi:.3f}]  n={len(merged_polity)}")
        rows.append({"metric": "auc_polity_3yr_decline_3pt", "point": auc, "ci_low": lo, "ci_high": hi, "n": len(merged_polity)})

    out = pd.DataFrame(rows)
    out_path = os.path.join(OUT, "bootstrap_cis.csv")
    out.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
