"""
Baseline comparison + staged ablation study.

Compares AIM4D against simple baselines and shows incremental value of
each pipeline component via progressive ablation.

Methodological basis:
  - Goldstone et al. (2010, AJPS): parsimonious 4-variable logistic as benchmark
  - Ward, Greenhill & Bakke (2010): OOS prediction as standard
  - Muchlinski et al. (2016): ML vs logistic for rare events
  - Hegre et al. (2019, ViEWS): exemplary EWS evaluation framework

Baselines:
  1. Naive: predict by base rate
  2. Logistic regression: polyarchy + log GDP per capita + region FE
  3. PITF-style: regime type + infant mortality proxy + neighborhood
  4. Factors only: POET factors + logistic
  5. Factors + HMM: regime probabilities + logistic
  6. Full AIM4D (no network): stages 1-3 + EWS
  7. Full AIM4D (with network): stages 1-5

Reports:
  - AUC-ROC, AUC-PR, Brier score for each model
  - Detection rate (sensitivity) at fixed FPR
  - Precision at top-K countries
  - Expanding-window temporal CV for each baseline
"""

import sys
import os
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from stage5_ews.estimate import KNOWN_EPISODES, LEAD_YEARS, TRAIN_CUTOFF

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Temporal CV windows (same as main pipeline)
WINDOWS = [
    (2005, 2008), (2008, 2011), (2011, 2014),
    (2014, 2017), (2017, 2020), (2020, 2023),
]


def build_labels(df):
    """Build binary labels from known episodes."""
    known_w = {}
    for c, info in KNOWN_EPISODES.items():
        for y in range(info["onset"] - LEAD_YEARS, info["onset"] + 1):
            known_w[(c, y)] = True
    df["label"] = df.apply(
        lambda r: 1 if (r.get("country_name", ""), r.get("year", 0)) in known_w else 0, axis=1
    )
    return df


def load_vdem_panel():
    """Load V-Dem data for baseline models."""
    vdem_path = os.path.join(os.path.dirname(__file__), "..", "data", "vdem_v16.csv")
    cols = ["country_name", "country_text_id", "year",
            "v2x_polyarchy", "v2x_regime", "v2x_libdem",
            "v2x_partipdem", "v2x_delibdem", "v2x_egaldem"]
    available = pd.read_csv(vdem_path, low_memory=False, nrows=1).columns
    cols = [c for c in cols if c in available]
    vdem = pd.read_csv(vdem_path, low_memory=False, usecols=cols)
    vdem = vdem[vdem["year"] >= 1990]
    return vdem


def load_macro_panel():
    """Load macro covariates."""
    path = os.path.join(os.path.dirname(__file__), "..", "data", "macro_covariates.csv")
    if not os.path.exists(path):
        return None
    macro = pd.read_csv(path)
    return macro


def load_factors():
    """Load stage 1 factors."""
    path = os.path.join(os.path.dirname(__file__), "..", "stage1_factors", "country_year_factors.csv")
    return pd.read_csv(path)


def load_states():
    """Load stage 3 regime probabilities."""
    path = os.path.join(os.path.dirname(__file__), "..", "stage3_msvar", "country_year_states.csv")
    return pd.read_csv(path)


def load_contagion():
    """Load stage 4 contagion scores."""
    path = os.path.join(os.path.dirname(__file__), "..", "stage4_nscm", "contagion_scores.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def load_ews():
    """Load EWS signals — run full stage 5 to get all columns including combined_risk."""
    try:
        from stage5_ews.estimate import run_ews
        df = run_ews()
        return df
    except Exception:
        path = os.path.join(os.path.dirname(__file__), "..", "stage5_ews", "ews_signals.csv")
        return pd.read_csv(path)


def neighborhood_polyarchy(vdem):
    """Compute mean polyarchy of geographic neighbors (simple proxy: region average)."""
    # Use country_text_id first letter as crude region proxy
    # Better: use contiguity data, but this is a baseline
    vdem["region"] = vdem["country_text_id"].str[:2]
    vdem["neighbor_polyarchy"] = vdem.groupby(["region", "year"])["v2x_polyarchy"].transform("mean")
    return vdem


def evaluate_model(y_true, y_pred, model_name):
    """Compute standard evaluation metrics."""
    results = {"model": model_name}

    if len(np.unique(y_true)) < 2 or len(y_true) == 0:
        return {**results, "auc_roc": np.nan, "auc_pr": np.nan, "brier": np.nan}

    try:
        results["auc_roc"] = roc_auc_score(y_true, y_pred)
    except ValueError:
        results["auc_roc"] = np.nan

    try:
        results["auc_pr"] = average_precision_score(y_true, y_pred)
    except ValueError:
        results["auc_pr"] = np.nan

    try:
        results["brier"] = brier_score_loss(y_true, np.clip(y_pred, 0, 1))
    except ValueError:
        results["brier"] = np.nan

    results["base_rate"] = y_true.mean()
    results["n"] = len(y_true)
    results["n_positive"] = int(y_true.sum())

    # Lift at top 10%
    if len(y_pred) > 10:
        top_10_thresh = np.percentile(y_pred, 90)
        top_10 = y_true[y_pred >= top_10_thresh]
        results["precision_top10"] = top_10.mean() if len(top_10) > 0 else 0
        results["lift_top10"] = results["precision_top10"] / max(results["base_rate"], 1e-6)

    return results


def temporal_cv(df, feature_cols, model_name):
    """Expanding-window temporal CV matching the main pipeline."""
    aucs = []
    for train_end, test_end in WINDOWS:
        train = df[(df["year"] <= train_end) & df[feature_cols].notna().all(axis=1)]
        test = df[(df["year"] > train_end) & (df["year"] <= test_end) & df[feature_cols].notna().all(axis=1)]

        if train["label"].sum() < 3 or test["label"].sum() == 0 or test["label"].nunique() < 2:
            continue

        X_train = train[feature_cols].values
        y_train = train["label"].values
        X_test = test[feature_cols].values
        y_test = test["label"].values

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
            model.fit(X_train_s, y_train)

        y_pred = model.predict_proba(X_test_s)[:, 1]
        try:
            auc = roc_auc_score(y_test, y_pred)
            aucs.append(auc)
        except ValueError:
            continue

    return np.mean(aucs) if aucs else np.nan, np.std(aucs) if aucs else np.nan


def detection_rate(df, risk_col):
    """Compute episode detection rate."""
    hits, total = 0, 0
    for country, info in KNOWN_EPISODES.items():
        onset = info["onset"]
        pre = df[(df["country_name"] == country) &
                 (df["year"] >= onset - LEAD_YEARS) & (df["year"] < onset)]
        if len(pre) == 0:
            continue
        total += 1
        threshold = df[df["year"] <= TRAIN_CUTOFF][risk_col].quantile(0.95)
        if pre[risk_col].max() >= threshold:
            hits += 1
    return hits, total


def run_baseline_comparison():
    print("=" * 70)
    print("ROBUSTNESS CHECK: Baseline Comparison + Staged Ablation")
    print("=" * 70)
    print()
    print("Methodological basis:")
    print("  Goldstone et al. (2010) PITF logistic benchmark;")
    print("  Ward et al. (2010) OOS prediction standard;")
    print("  Hegre et al. (2019) ViEWS evaluation framework")
    print()

    # Load all data sources
    print("Loading data...")
    vdem = load_vdem_panel()
    macro = load_macro_panel()
    factors = load_factors()
    states = load_states()
    contagion = load_contagion()
    ews = load_ews()

    # Build unified panel
    panel = vdem.copy()
    if macro is not None:
        macro_cols = ["gdp_pc", "gdp_growth", "urbanization"]
        avail = [c for c in macro_cols if c in macro.columns]
        panel = panel.merge(
            macro[["iso3", "year"] + avail].rename(columns={"iso3": "country_text_id"}),
            on=["country_text_id", "year"], how="left",
        )
        for c in avail:
            panel[c] = panel[c].fillna(panel[c].median())
        panel["log_gdp_pc"] = np.log1p(panel.get("gdp_pc", 0))

    factor_cols_list = [c for c in factors.columns if c.startswith("factor_")]
    panel = panel.merge(factors[["country_text_id", "year"] + factor_cols_list],
                        on=["country_text_id", "year"], how="left")

    state_prob_cols = [c for c in states.columns if c.startswith("prob_state_")]
    panel = panel.merge(states[["country_text_id", "year"] + state_prob_cols],
                        on=["country_text_id", "year"], how="left")

    if contagion is not None and "contagion_score" in contagion.columns:
        panel = panel.merge(
            contagion[["country_text_id", "year", "contagion_score"]],
            on=["country_text_id", "year"], how="left",
        )
        panel["contagion_score"] = panel["contagion_score"].fillna(0)

    # Add EWS features
    ews_cols = ["csd_index", "combined_risk"]
    ews_avail = [c for c in ews_cols if c in ews.columns]
    if ews_avail:
        panel = panel.merge(ews[["country_text_id", "year"] + ews_avail],
                            on=["country_text_id", "year"], how="left")

    panel = neighborhood_polyarchy(panel)
    panel = build_labels(panel)

    print(f"Panel: {len(panel)} country-years, {panel['label'].sum()} positive labels")

    # Define models
    models = {}

    # Baseline 1: Naive (base rate)
    models["naive"] = {"features": [], "description": "Predict base rate"}

    # Baseline 2: Logistic (polyarchy + GDP)
    b2_feats = ["v2x_polyarchy"]
    if "log_gdp_pc" in panel.columns:
        b2_feats.append("log_gdp_pc")
    if "neighbor_polyarchy" in panel.columns:
        b2_feats.append("neighbor_polyarchy")
    models["logistic_simple"] = {
        "features": b2_feats,
        "description": "Polyarchy + log GDP + neighborhood (Goldstone-style)",
    }

    # Baseline 3: PITF-style (regime type + macro)
    b3_feats = ["v2x_polyarchy"]
    if "v2x_regime" in panel.columns:
        b3_feats.append("v2x_regime")
    if "gdp_growth" in panel.columns:
        b3_feats.append("gdp_growth")
    if "neighbor_polyarchy" in panel.columns:
        b3_feats.append("neighbor_polyarchy")
    models["pitf_style"] = {
        "features": b3_feats,
        "description": "Regime type + GDP growth + neighborhood",
    }

    # Ablation 1: Factors only
    f_feats = [c for c in factor_cols_list if c in panel.columns]
    if f_feats:
        models["factors_only"] = {
            "features": f_feats,
            "description": "POET factors (stage 1) + logistic",
        }

    # Ablation 2: Factors + regime probs
    fr_feats = f_feats + [c for c in state_prob_cols if c in panel.columns]
    if len(fr_feats) > len(f_feats):
        models["factors_hmm"] = {
            "features": fr_feats,
            "description": "Factors + HMM regime probs (stages 1-3)",
        }

    # Ablation 3: Factors + regime + network
    frn_feats = fr_feats.copy()
    if "contagion_score" in panel.columns:
        frn_feats.append("contagion_score")
        models["factors_hmm_network"] = {
            "features": frn_feats,
            "description": "Factors + HMM + network contagion (stages 1-4)",
        }

    # Full AIM4D
    if "combined_risk" in panel.columns:
        models["aim4d_full"] = {
            "features": ["combined_risk"],
            "description": "Full AIM4D pipeline (stages 1-5)",
            "precomputed": True,
        }

    # Run evaluation
    all_results = []
    train_data = panel[panel["year"] <= TRAIN_CUTOFF]
    all_data = panel.dropna(subset=["label"])

    for name, config in models.items():
        print(f"\n{'='*50}")
        print(f"Model: {name}")
        print(f"  {config['description']}")
        print(f"{'='*50}")

        features = config["features"]

        if name == "naive":
            y_pred = np.full(len(all_data), all_data["label"].mean())
            y_true = all_data["label"].values
            result = evaluate_model(y_true, y_pred, name)
            result["cv_auc_mean"] = all_data["label"].mean()
            result["cv_auc_std"] = 0.0
            result["detection_rate"] = 0.0
            all_results.append(result)
            print(f"  Base rate: {result['base_rate']:.4f}")
            continue

        if config.get("precomputed"):
            # AIM4D full: use combined_risk directly
            valid = all_data.dropna(subset=features)
            y_true = valid["label"].values
            y_pred = valid[features[0]].values
            result = evaluate_model(y_true, y_pred, name)

            # Detection rate
            hits, total = detection_rate(valid, features[0])
            result["detection_rate"] = hits / total if total > 0 else 0
            result["detections"] = f"{hits}/{total}"

            # Temporal CV (use same windows)
            cv_aucs = []
            for train_end, test_end in WINDOWS:
                w_test = valid[(valid["year"] > train_end) & (valid["year"] <= test_end)]
                if w_test["label"].sum() > 0 and w_test["label"].nunique() > 1:
                    try:
                        cv_aucs.append(roc_auc_score(w_test["label"], w_test[features[0]]))
                    except ValueError:
                        pass
            result["cv_auc_mean"] = np.mean(cv_aucs) if cv_aucs else np.nan
            result["cv_auc_std"] = np.std(cv_aucs) if cv_aucs else np.nan

            all_results.append(result)
        else:
            # Standard logistic model
            avail_feats = [f for f in features if f in panel.columns]
            valid = all_data.dropna(subset=avail_feats + ["label"])

            if len(avail_feats) == 0 or valid["label"].sum() < 3:
                print(f"  Insufficient data")
                continue

            # Full evaluation
            train = valid[valid["year"] <= TRAIN_CUTOFF]
            X_train = train[avail_feats].values
            y_train = train["label"].values
            X_all = valid[avail_feats].values
            y_all = valid["label"].values

            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train)
            X_all_s = scaler.transform(X_all)

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = LogisticRegressionCV(cv=3, max_iter=1000, random_state=42)
                model.fit(X_train_s, y_train)

            y_pred = model.predict_proba(X_all_s)[:, 1]
            result = evaluate_model(y_all, y_pred, name)

            # Detection rate
            valid_with_pred = valid.copy()
            valid_with_pred["pred_risk"] = y_pred
            hits, total = detection_rate(valid_with_pred, "pred_risk")
            result["detection_rate"] = hits / total if total > 0 else 0
            result["detections"] = f"{hits}/{total}"

            # Temporal CV
            cv_mean, cv_std = temporal_cv(valid, avail_feats, name)
            result["cv_auc_mean"] = cv_mean
            result["cv_auc_std"] = cv_std

            # Feature importance
            if hasattr(model, 'coef_'):
                coefs = dict(zip(avail_feats, model.coef_[0]))
                top = sorted(coefs.items(), key=lambda x: -abs(x[1]))[:5]
                print(f"  Top features: {[(f, f'{c:+.3f}') for f, c in top]}")

            all_results.append(result)

        r = all_results[-1]
        print(f"  AUC-ROC: {r.get('auc_roc', np.nan):.3f}")
        print(f"  AUC-PR:  {r.get('auc_pr', np.nan):.3f}")
        print(f"  Brier:   {r.get('brier', np.nan):.4f}")
        print(f"  CV AUC:  {r.get('cv_auc_mean', np.nan):.3f} +/- {r.get('cv_auc_std', np.nan):.3f}")
        if "detection_rate" in r:
            print(f"  Detection rate: {r['detection_rate']:.0%} ({r.get('detections', '')})")

    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY TABLE: Baseline Comparison + Staged Ablation")
    print(f"{'='*70}")
    summary = pd.DataFrame(all_results)
    cols_show = ["model", "auc_roc", "auc_pr", "brier", "cv_auc_mean", "cv_auc_std",
                 "detection_rate", "n_positive"]
    cols_avail = [c for c in cols_show if c in summary.columns]
    print(summary[cols_avail].to_string(index=False, float_format="%.3f"))

    # Incremental value analysis
    print(f"\n{'='*50}")
    print("INCREMENTAL VALUE OF EACH PIPELINE STAGE")
    print(f"{'='*50}")

    stage_order = ["logistic_simple", "factors_only", "factors_hmm",
                   "factors_hmm_network", "aim4d_full"]
    stage_names = {
        "logistic_simple": "Baseline (polyarchy + GDP)",
        "factors_only": "+ Stage 1 (POET factors)",
        "factors_hmm": "+ Stage 3 (HMM regimes)",
        "factors_hmm_network": "+ Stage 4 (network contagion)",
        "aim4d_full": "+ Stage 5 (EWS meta-learner)",
    }

    prev_auc = None
    for sname in stage_order:
        row = summary[summary["model"] == sname]
        if len(row) == 0:
            continue
        auc = row["auc_roc"].iloc[0]
        delta = f"(+{auc - prev_auc:.3f})" if prev_auc is not None and not np.isnan(auc) else ""
        label = stage_names.get(sname, sname)
        print(f"  {label:45s} AUC={auc:.3f} {delta}")
        if not np.isnan(auc):
            prev_auc = auc

    summary.to_csv(os.path.join(OUTPUT_DIR, "baseline_comparison_results.csv"), index=False)
    print(f"\nSaved to robustness/baseline_comparison_results.csv")

    return summary


if __name__ == "__main__":
    run_baseline_comparison()
