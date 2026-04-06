import sys
import os
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

FACTOR_COLS = ["factor_1", "factor_2", "factor_3", "factor_4"]
MIN_OBS = 30


def load_factor_scores():
    path = os.path.join(os.path.dirname(__file__), "..", "stage1_factors", "country_year_factors.csv")
    return pd.read_csv(path)


def compute_loo_global(df, country):
    others = df[df["country_name"] != country]
    return others.groupby("year")[FACTOR_COLS].mean()


def kalman_tvp_univariate(y, x, q_var, r_var):
    T = len(y)

    valid = min(10, T)
    x_init = x[:valid]
    y_init = y[:valid]
    mask = np.isfinite(x_init) & np.isfinite(y_init) & (np.abs(x_init) > 1e-10)
    if mask.sum() >= 2:
        beta_0 = np.sum(x_init[mask] * y_init[mask]) / np.sum(x_init[mask] ** 2)
    else:
        beta_0 = 1.0
    P_0 = 1.0

    beta_filt = np.zeros(T)
    P_filt = np.zeros(T)
    beta_pred = np.zeros(T)
    P_pred = np.zeros(T)

    for t in range(T):
        if t == 0:
            bp = beta_0
            Pp = P_0 + q_var
        else:
            bp = beta_filt[t - 1]
            Pp = P_filt[t - 1] + q_var

        beta_pred[t] = bp
        P_pred[t] = Pp

        y_hat = x[t] * bp
        F_t = x[t] ** 2 * Pp + r_var
        K_t = Pp * x[t] / F_t

        beta_filt[t] = bp + K_t * (y[t] - y_hat)
        P_filt[t] = (1 - K_t * x[t]) * Pp

    beta_smooth = np.zeros(T)
    P_smooth = np.zeros(T)
    beta_smooth[-1] = beta_filt[-1]
    P_smooth[-1] = P_filt[-1]

    for t in range(T - 2, -1, -1):
        if P_pred[t + 1] > 1e-15:
            J = P_filt[t] / P_pred[t + 1]
        else:
            J = 0.0
        beta_smooth[t] = beta_filt[t] + J * (beta_smooth[t + 1] - beta_pred[t + 1])
        P_smooth[t] = P_filt[t] + J ** 2 * (P_smooth[t + 1] - P_pred[t + 1])

    return beta_smooth, P_smooth


def tvp_loglik_uni(params, y, x):
    q_var = np.exp(params[0])
    r_var = np.exp(params[1])
    T = len(y)

    valid = min(10, T)
    x_init = x[:valid]
    y_init = y[:valid]
    mask = np.isfinite(x_init) & np.isfinite(y_init) & (np.abs(x_init) > 1e-10)
    if mask.sum() >= 2:
        bp = np.sum(x_init[mask] * y_init[mask]) / np.sum(x_init[mask] ** 2)
    else:
        bp = 1.0
    Pp = 1.0 + q_var

    ll = 0.0
    for t in range(T):
        y_hat = x[t] * bp
        F_t = x[t] ** 2 * Pp + r_var
        v_t = y[t] - y_hat

        if F_t > 1e-15:
            ll += -0.5 * (np.log(2 * np.pi) + np.log(F_t) + v_t ** 2 / F_t)

        K_t = Pp * x[t] / F_t if F_t > 1e-15 else 0.0
        bp = bp + K_t * v_t
        Pp = (1 - K_t * x[t]) * Pp + q_var

    return -ll


def garch11_variance(series):
    T = len(series)
    omega = np.var(series) * 0.05
    alpha = 0.1
    beta = 0.85
    h = np.full(T, np.var(series))
    for t in range(1, T):
        h[t] = omega + alpha * series[t - 1] ** 2 + beta * h[t - 1]
        h[t] = max(h[t], 1e-10)
    return h


def ewma_correlation(z_i, z_f, lam=0.94):
    T = len(z_i)
    q_ii = np.ones(T)
    q_ff = np.ones(T)
    q_if = np.full(T, np.corrcoef(z_i, z_f)[0, 1] if len(z_i) > 2 else 0.0)
    rho = np.zeros(T)

    for t in range(1, T):
        q_ii[t] = (1 - lam) * z_i[t - 1] ** 2 + lam * q_ii[t - 1]
        q_ff[t] = (1 - lam) * z_f[t - 1] ** 2 + lam * q_ff[t - 1]
        q_if[t] = (1 - lam) * z_i[t - 1] * z_f[t - 1] + lam * q_if[t - 1]

    denom = np.sqrt(q_ii * q_ff)
    denom = np.maximum(denom, 1e-10)
    rho = q_if / denom
    return np.clip(rho, -0.999, 0.999)


def dcc_garch_beta(y, x):
    h_y = garch11_variance(y)
    h_x = garch11_variance(x)

    sigma_y = np.sqrt(h_y)
    sigma_x = np.sqrt(h_x)

    z_y = y / np.maximum(sigma_y, 1e-10)
    z_x = x / np.maximum(sigma_x, 1e-10)

    rho = ewma_correlation(z_y, z_x)

    beta_dcc = rho * sigma_y / np.maximum(sigma_x, 1e-10)
    return beta_dcc, rho, sigma_y, sigma_x


def estimate_country_factor_beta(y, x):
    init_params = np.array([np.log(0.05), np.log(np.var(y) * 0.5 + 1e-6)])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = minimize(
            tvp_loglik_uni, init_params, args=(y, x),
            method="L-BFGS-B",
            bounds=[(-8, 2), (-8, 5)],
        )

    q_var = np.exp(result.x[0])
    r_var = np.exp(result.x[1])

    beta_kalman, P_smooth = kalman_tvp_univariate(y, x, q_var, r_var)

    beta_dcc, rho, sig_y, sig_x = dcc_garch_beta(y, x)

    beta_combined = 0.5 * beta_kalman + 0.5 * beta_dcc

    return beta_combined, P_smooth, q_var, r_var, result.fun, rho


def estimate_all_betas():
    df = load_factor_scores()
    print(f"Loaded {len(df)} country-years, {df['country_name'].nunique()} countries")

    results = []
    diagnostics = []
    countries = df["country_name"].unique()

    for i, country in enumerate(countries):
        cdf = df[df["country_name"] == country].sort_values("year")
        if len(cdf) < MIN_OBS:
            continue

        years = cdf["year"].values
        y_all = cdf[FACTOR_COLS].values
        gf = compute_loo_global(df, country).loc[years].values

        country_betas = np.zeros((len(years), len(FACTOR_COLS)))

        for k, fcol in enumerate(FACTOR_COLS):
            dy = np.diff(y_all[:, k])
            dx = np.diff(gf[:, k])

            beta_smooth, P_smooth, q_var, r_var, nll, rho = estimate_country_factor_beta(dy, dx)

            country_betas[0, k] = beta_smooth[0]
            country_betas[1:, k] = beta_smooth

            diagnostics.append({
                "country": country,
                "factor": fcol,
                "q_state": q_var,
                "r_obs": r_var,
                "snr": q_var / r_var if r_var > 1e-15 else 0,
                "neg_loglik": nll,
                "T": len(years),
                "beta_mean": beta_smooth.mean(),
                "beta_std": beta_smooth.std(),
            })

        for t_idx, year in enumerate(years):
            row = {
                "country_name": country,
                "country_text_id": cdf["country_text_id"].iloc[0],
                "year": int(year),
            }
            for k, fcol in enumerate(FACTOR_COLS):
                row[f"beta_{fcol}"] = country_betas[t_idx, k]
            results.append(row)

        if (i + 1) % 25 == 0 or i == 0:
            print(f"  [{i+1}/{len(countries)}] {country}: done")

    beta_df = pd.DataFrame(results)
    diag_df = pd.DataFrame(diagnostics)

    output_dir = os.path.dirname(os.path.abspath(__file__))
    beta_df.to_csv(os.path.join(output_dir, "country_year_betas.csv"), index=False)
    diag_df.to_csv(os.path.join(output_dir, "estimation_diagnostics.csv"), index=False)

    print(f"\nEstimated betas for {beta_df['country_name'].nunique()} countries")
    print(f"Output: {len(beta_df)} country-years x {len(FACTOR_COLS)} factor betas")

    print(f"\nBeta summary by factor:")
    for fcol in FACTOR_COLS:
        col = f"beta_{fcol}"
        latest = beta_df[beta_df["year"] == beta_df["year"].max()][col]
        print(f"  {fcol}: mean={latest.mean():.3f}, std={latest.std():.3f}, "
              f"range=[{latest.min():.3f}, {latest.max():.3f}]")

    return beta_df, diag_df


if __name__ == "__main__":
    beta_df, diag_df = estimate_all_betas()

    print("\n=== Hungary betas over time ===")
    hun = beta_df[beta_df["country_name"] == "Hungary"].sort_values("year")
    for _, row in hun.tail(10).iterrows():
        print(f"  {int(row['year'])}: F1={row['beta_factor_1']:.3f} F2={row['beta_factor_2']:.3f} "
              f"F3={row['beta_factor_3']:.3f} F4={row['beta_factor_4']:.3f}")

    print("\n=== Turkey betas over time ===")
    tur = beta_df[beta_df["country_name"] == "Türkiye"].sort_values("year")
    if len(tur) == 0:
        tur = beta_df[beta_df["country_name"].str.contains("Turk", case=False)].sort_values("year")
    for _, row in tur.tail(10).iterrows():
        print(f"  {int(row['year'])}: F1={row['beta_factor_1']:.3f} F2={row['beta_factor_2']:.3f} "
              f"F3={row['beta_factor_3']:.3f} F4={row['beta_factor_4']:.3f}")

    print("\n=== Poland betas over time ===")
    pol = beta_df[beta_df["country_name"] == "Poland"].sort_values("year")
    for _, row in pol.tail(10).iterrows():
        print(f"  {int(row['year'])}: F1={row['beta_factor_1']:.3f} F2={row['beta_factor_2']:.3f} "
              f"F3={row['beta_factor_3']:.3f} F4={row['beta_factor_4']:.3f}")

    print("\n=== 2025 Factor 1 beta rankings ===")
    latest = beta_df[beta_df["year"] == 2025].sort_values("beta_factor_1")
    print("Bottom 5 (decoupling from global democratic trend):")
    for _, row in latest.head(5).iterrows():
        print(f"  {row['country_name']}: {row['beta_factor_1']:.3f}")
    print("Top 5 (amplifying global democratic trend):")
    for _, row in latest.tail(5).iterrows():
        print(f"  {row['country_name']}: {row['beta_factor_1']:.3f}")

    print("\n=== Cross-country beta variation ===")
    for fcol in FACTOR_COLS:
        col = f"beta_{fcol}"
        cv = beta_df.groupby("country_name")[col].std().mean()
        print(f"  {fcol}: avg within-country std = {cv:.4f}")
