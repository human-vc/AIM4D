"""
Pre-onset detection lead-time distribution.

For each detected autocratization episode, computes the earliest year in
the pre-onset window where the framework's combined_risk first crossed the
watch (P80), warning (P95), and alert (P98) thresholds. Reports the
distribution of lead times so reviewers can see whether detections are
consistently early or just barely-on-time.

Outputs robustness/detection_lead_times.csv and prints a summary.
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

OUT = os.path.dirname(os.path.abspath(__file__))
LEAD_YEARS = 5
TRAIN_CUTOFF = 2021


def main():
    from stage5_ews.estimate import KNOWN_EPISODES

    ews_path = os.path.join(OUT, "..", "stage5_ews", "ews_signals.csv")
    ews = pd.read_csv(ews_path)
    if "combined_risk" not in ews.columns:
        raise RuntimeError("ews_signals.csv missing combined_risk; rerun stage 5")

    train = ews[ews["year"] <= TRAIN_CUTOFF]["combined_risk"]
    thresh = {
        "watch": train.quantile(0.80),
        "warning": train.quantile(0.95),
        "alert": train.quantile(0.98),
    }
    print(f"Thresholds: watch={thresh['watch']:.4f}  warning={thresh['warning']:.4f}  alert={thresh['alert']:.4f}\n")

    rows = []
    for country, info in KNOWN_EPISODES.items():
        onset = info["onset"]
        ep_type = info["type"]
        pre = ews[(ews["country_name"] == country)
                  & (ews["year"] >= onset - LEAD_YEARS)
                  & (ews["year"] < onset)].sort_values("year")
        if len(pre) == 0:
            rows.append({
                "country": country, "type": ep_type, "onset": onset,
                "watch_lead": np.nan, "warning_lead": np.nan, "alert_lead": np.nan,
                "max_risk": np.nan,
            })
            continue

        max_risk = float(pre["combined_risk"].max())
        leads = {}
        for tier, t in thresh.items():
            hits = pre[pre["combined_risk"] >= t]
            leads[tier] = (onset - int(hits["year"].min())) if len(hits) > 0 else np.nan

        rows.append({
            "country": country, "type": ep_type, "onset": onset,
            "watch_lead": leads["watch"],
            "warning_lead": leads["warning"],
            "alert_lead": leads["alert"],
            "max_risk": max_risk,
        })

    df = pd.DataFrame(rows)
    out_path = os.path.join(OUT, "detection_lead_times.csv")
    df.to_csv(out_path, index=False)

    print("Per-episode lead times (years before onset; NaN = never crossed threshold)\n")
    print(df.sort_values(["type", "onset"]).to_string(index=False))

    print("\n=== Distribution summary ===")
    for tier in ("watch", "warning", "alert"):
        s = df[f"{tier}_lead"].dropna()
        if len(s) == 0:
            continue
        detected = len(s)
        total = len(df.dropna(subset=["max_risk"]))
        print(f"\n  {tier.upper()} tier:")
        print(f"    detected: {detected}/{total}")
        print(f"    lead time: mean={s.mean():.2f}yr  median={s.median():.1f}yr  "
              f"min={int(s.min())}  max={int(s.max())}")
        print(f"    distribution: {dict(s.astype(int).value_counts().sort_index())}")

    print("\n=== By episode type (watch tier) ===")
    for ep_type in ("backsliding", "coup"):
        sub = df[df["type"] == ep_type]["watch_lead"].dropna()
        if len(sub) == 0:
            continue
        print(f"  {ep_type:<14} n={len(sub):>2}  mean lead={sub.mean():.2f}yr  median={sub.median():.1f}yr")

    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
