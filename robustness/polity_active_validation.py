"""
Polity-active subset validation.

The full-sample Polity cross-validation (robustness/polity_validation.py)
returns AUC ~0.53 — the paper's defense is that Polity is too coarse to see
gradual erosion (e.g., Hungary stays at Polity 10 throughout 2010-2018).
This script tests that defense empirically. It restricts the validation to
country-years where Polity is *responsive* (its score has actually moved)
and asks: when Polity does record movement, does the AIM4D risk score
predict its declines?

Two operationalizations of "Polity-active":
  (1) Country-level: countries with any Polity variance over their history.
  (2) Country-year level: country-years preceded by Polity movement in the
      past 10 years (purely backward-looking).

Outputs robustness/polity_active_validation.csv and prints AUCs with
1,000-replicate bootstrap CIs.
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
RNG = np.random.default_rng(42)
N_BOOT = 1000


def bootstrap_auc_ci(y_true, y_score, n_boot=N_BOOT, alpha=0.05):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n = len(y_true)
    aucs = []
    for _ in range(n_boot):
        idx = RNG.integers(0, n, n)
        yt, ys = y_true[idx], y_score[idx]
        if yt.sum() == 0 or yt.sum() == len(yt):
            continue
        aucs.append(roc_auc_score(yt, ys))
    if not aucs:
        return (np.nan, np.nan, np.nan)
    aucs = np.array(aucs)
    return float(aucs.mean()), float(np.percentile(aucs, 100 * alpha / 2)), float(np.percentile(aucs, 100 * (1 - alpha / 2)))


def main():
    print("=" * 70)
    print("POLITY-ACTIVE SUBSET VALIDATION")
    print("=" * 70)

    vdem_path = os.path.join(os.path.dirname(__file__), "..", "data", "vdem_v16.csv")
    polity = pd.read_csv(
        vdem_path, low_memory=False,
        usecols=["country_name", "country_text_id", "year", "e_polity2"],
    )
    polity = polity.dropna(subset=["e_polity2"]).copy()
    polity["e_polity2"] = polity["e_polity2"].astype(float)
    polity = polity.sort_values(["country_text_id", "year"]).reset_index(drop=True)

    g = polity.groupby("country_text_id", group_keys=False)
    for h in (3, 5):
        polity[f"polity_change_{h}yr"] = g["e_polity2"].diff(h)
        polity[f"polity_decline_{h}yr_2pt"] = (polity[f"polity_change_{h}yr"] <= -2).astype(int)
        polity[f"polity_decline_{h}yr_3pt"] = (polity[f"polity_change_{h}yr"] <= -3).astype(int)

    polity["polity_past10_max"] = g["e_polity2"].transform(
        lambda x: x.shift(1).rolling(10, min_periods=5).max()
    )
    polity["polity_past10_min"] = g["e_polity2"].transform(
        lambda x: x.shift(1).rolling(10, min_periods=5).min()
    )
    polity["polity_past10_range"] = polity["polity_past10_max"] - polity["polity_past10_min"]
    polity["polity_active_year"] = (polity["polity_past10_range"] >= 1).astype(int)

    country_variance = polity.groupby("country_text_id")["e_polity2"].nunique()
    polity["polity_active_country"] = polity["country_text_id"].map(
        (country_variance > 1).astype(int)
    )

    ews_path = os.path.join(os.path.dirname(__file__), "..", "stage5_ews", "ews_signals.csv")
    if not os.path.exists(ews_path):
        raise FileNotFoundError(f"Run stage5_ews/estimate.py first: {ews_path} not found")
    ews = pd.read_csv(ews_path)
    if "combined_risk" not in ews.columns:
        raise RuntimeError("ews_signals.csv is missing the combined_risk column")

    merged = ews[["country_text_id", "year", "combined_risk"]].merge(
        polity, on=["country_text_id", "year"], how="inner",
    ).dropna(subset=["combined_risk"])

    print(f"\nMerged panel: {len(merged):,} country-years, "
          f"{merged['country_text_id'].nunique()} countries")
    print(f"Polity-active countries: {int(merged['polity_active_country'].max() and country_variance.gt(1).sum())} / "
          f"{merged['country_text_id'].nunique()}")

    rows = []
    print("\n--- Full sample (paper baseline) ---")
    for h in (3, 5):
        for thr in (2, 3):
            col = f"polity_decline_{h}yr_{thr}pt"
            sub = merged.dropna(subset=[col])
            if sub[col].sum() == 0 or sub[col].nunique() < 2:
                continue
            auc = roc_auc_score(sub[col], sub["combined_risk"])
            print(f"  {h}yr decline ≥{thr}pt:  n={len(sub):,}  n_pos={int(sub[col].sum())}  AUC={auc:.3f}")
            rows.append({
                "subset": "full",
                "horizon_yr": h,
                "decline_threshold": thr,
                "n": len(sub),
                "n_positive": int(sub[col].sum()),
                "auc": auc,
                "auc_ci_low": np.nan,
                "auc_ci_high": np.nan,
            })

    for subset_name, mask_col in (
        ("active_country", "polity_active_country"),
        ("active_year", "polity_active_year"),
    ):
        print(f"\n--- Polity-active subset: {subset_name} ---")
        sub_panel = merged[merged[mask_col] == 1]
        if len(sub_panel) < 30:
            print(f"  Subset too small ({len(sub_panel)} rows). Skipping.")
            continue
        for h in (3, 5):
            for thr in (2, 3):
                col = f"polity_decline_{h}yr_{thr}pt"
                sub = sub_panel.dropna(subset=[col])
                if len(sub) < 30 or sub[col].sum() == 0 or sub[col].nunique() < 2:
                    continue
                auc = roc_auc_score(sub[col], sub["combined_risk"])
                auc_mean, lo, hi = bootstrap_auc_ci(sub[col].values, sub["combined_risk"].values)
                n_pos = int(sub[col].sum())
                base = sub[col].mean()
                print(f"  {h}yr decline ≥{thr}pt:  n={len(sub):,}  n_pos={n_pos} ({base:.1%})  "
                      f"AUC={auc:.3f}  95% CI [{lo:.3f}, {hi:.3f}]")
                rows.append({
                    "subset": subset_name,
                    "horizon_yr": h,
                    "decline_threshold": thr,
                    "n": len(sub),
                    "n_positive": n_pos,
                    "auc": auc,
                    "auc_ci_low": lo,
                    "auc_ci_high": hi,
                })

    print("\n--- Spearman ρ (risk vs −Polity change, active-country subset) ---")
    sub_panel = merged[merged["polity_active_country"] == 1]
    for h in (3, 5):
        change_col = f"polity_change_{h}yr"
        sub = sub_panel.dropna(subset=[change_col])
        if len(sub) < 30:
            continue
        rho, p = spearmanr(sub["combined_risk"], -sub[change_col])
        print(f"  {h}yr:  n={len(sub):,}  ρ={rho:.3f}  p={p:.4f}")

    out = pd.DataFrame(rows)
    out_path = os.path.join(OUTPUT_DIR, "polity_active_validation.csv")
    out.to_csv(out_path, index=False)
    print(f"\nWrote {out_path}")

    print("\n--- Interpretation guide ---")
    print("  active_country AUC > 0.65: strong evidence framework predicts declines Polity can see")
    print("  active_country AUC 0.55-0.65: moderate, consistent with FH validation")
    print("  active_country AUC < 0.55: weak; framework may track V-Dem-specific signals")
    print("  active_country AUC < 0.50: framework anti-predicts; revisit construct validity")


if __name__ == "__main__":
    main()
