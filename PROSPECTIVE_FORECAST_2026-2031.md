# Pre-Registered Prospective Autocratization Forecast 2026–2031

This document locks in falsifiable predictions made by the AIM4D framework
(Crainic, Yee & Sharma, in prep.) on the date below. The framework is
described in the manuscript bundled in this repository; the predictions below
are generated **without any post-2025 V-Dem data** and are intended to be
evaluated against V-Dem v17+ / ERT v16+ / Polity-VI releases through 2031.

## Registration metadata

| Field | Value |
|---|---|
| Pre-registration date | 2026-05-17 |
| Model commit | `6d0c68a91010030dac348684f60f9813e51497ad` |
| Repository | <https://github.com/YCRG-Labs/AIM4D> |
| Training data cutoff | V-Dem v16 (2025-12-31 snapshot), upstream stages 1-4 fit on year ≤ 2019; Stage 5 meta-learner fit on year ≤ 2019, post-onset country-years excluded |
| Forecasting horizon | 6 years (2026-2031) |
| Headline OOS AUC (validation 2020-2025) | 0.933, 95% cluster-bootstrap CI [0.850, 0.983] |
| Number of training episodes | 46 (V-Dem v16 ERT-style backsliding + coup episodes 1996-2024) |

## The predictions

The 25 country-level risk scores below are the **single locked top-25 ranking**
produced by the framework's calibrated meta-ensemble (LR + 20-seed GB + RF +
ExtraTrees + CatBoost) on 2025 features. Risk is a calibrated probability of
autocratization onset between 2026 and 2031, on the same scale as the model's
in-sample tier thresholds (watch = 0.075, warning = 0.20, alert = 0.40).

| Rank | Country | Risk | Tier | Key signals |
|---:|---|---:|---|---|
| 1 | Bolivia | 0.476 | warning | CSD + mv-CSD + ELEC |
| 2 | Argentina | 0.463 | warning | CSD + mv-CSD + ELEC |
| 3 | Niger | 0.456 | warning | CSD |
| 4 | Iran | 0.409 | watch | meta-only |
| 5 | Uganda | 0.409 | watch | meta-only |
| 6 | Malaysia | 0.382 | watch | CSD |
| 7 | Chad | 0.377 | watch | CSD + mv-CSD |
| 8 | South Korea | 0.375 | watch | CSD + ELEC |
| 9 | Botswana | 0.364 | watch | CSD + mv-CSD + ELEC |
| 10 | Mexico | 0.363 | watch | CSD + mv-CSD + ELEC |
| 11 | Türkiye | 0.361 | watch | CSD + mv-CSD |
| 12 | Burma / Myanmar | 0.297 | watch | CSD |
| 13 | Colombia | 0.292 | watch | meta-only |
| 14 | Tanzania | 0.279 | watch | ELEC |
| 15 | Guinea-Bissau | 0.278 | watch | ELEC |
| 16 | Brazil | 0.273 | watch | CSD + mv-CSD |
| 17 | Democratic Republic of the Congo | 0.250 | watch | CSD + mv-CSD |
| 18 | Philippines | 0.249 | watch | CSD + ELEC |
| 19 | Tunisia | 0.247 | watch | CSD + ELEC |
| 20 | Central African Republic | 0.241 | watch | CSD + ELEC |
| 21 | Mongolia | 0.236 | watch | meta-only |
| 22 | Somalia | 0.231 | watch | meta-only |
| 23 | Burkina Faso | 0.229 | watch | CSD |
| 24 | Guinea | 0.219 | watch | CSD + ELEC |
| 25 | Senegal | 0.218 | watch | mv-CSD + ELEC |

The full prospective scoreboard for all 138 evaluable countries is bundled at
`stage5_ews/ews_signals.csv` at the registration commit hash above.

## Falsification criteria

A country in the top-25 is counted as a **true positive** if, by 2031-12-31,
**any** of the following has occurred:

1. V-Dem v17+ codes the country as having entered an Episodes of Regime
   Transformation (ERT) v16+ autocratization episode whose first year falls
   in 2026-2031, **or**
2. V-Dem `v2x_polyarchy` declines by ≥ 0.10 over any contiguous 5-year window
   contained within 2026-2031, **or**
3. V-Dem `v2x_regime` decreases by ≥ 1 step (e.g., liberal democracy →
   electoral democracy) at any point in 2026-2031, **or**
4. Polity-VI declines by ≥ 3 points over any contiguous 3-year window
   contained within 2026-2031, **or**
5. The country experiences a successful coup, executive self-coup, or
   constitutional rupture coded by the Cline Center Coup d'État Project,
   Marsteintredet & Malamud Self-Coups Database, or COW/Archigos with onset
   in 2026-2031.

A country is a **false positive** if none of (1)-(5) occurs by 2031-12-31.

## Evaluation metrics (to be reported in the post-2031 follow-up)

| Metric | What it tells you | Locked threshold |
|---|---|---|
| Precision @ 5 | Hit rate among most-confident predictions | — |
| Precision @ 10 | Standard top-10 forecast scoreboard | — |
| Precision @ 25 | Full pre-registered list | — |
| Brier score | Calibration quality of risk values | — |
| Hit at warning tier (top 3) | High-confidence true positives | — |

For each metric the framework's performance will be compared against the
in-sample expected value computed from the validation period (2020-2025
held-out evaluation), as reported in the manuscript.

## Update policy

This file is **frozen**: it is not edited after the registration date. If the
predictions need to be regenerated for any reason (bugfix, data update),
**a new dated pre-registration file is created** rather than overwriting this
one. The git history of this file documents the immutability.

## How to verify the registration

To confirm the predictions on this page were generated from the code state at
commit `6d0c68a` (or whichever hash above), check out that commit and rerun:

```bash
git checkout 6d0c68a
python3 stage1_factors/extract.py
python3 stage2_betas/estimate.py
python3 stage3_msvar/estimate.py
python3 stage4_nscm/estimate.py
python3 stage5_ews/estimate.py
# the "Prospective Risk Ranking" block at end of stage5 output should match
# the table above, modulo a few tail ranks that may shift by a row due to
# rounding (the model is deterministic; only stacked-ensemble seeds vary).
```

## External timestamp

For an independent timestamp of this registration outside the GitHub history,
this file's SHA-256 is uploaded to:

- Open Science Framework: <https://osf.io/> (upload at registration; URL filled
  in once the user has uploaded)
- arXiv (as part of the bundled manuscript): URL filled in upon submission

The hash on the OSF/arXiv copy must match the hash on this file at the
registration commit. Anyone can verify with:

```bash
sha256sum PROSPECTIVE_FORECAST_2026-2031.md
```
