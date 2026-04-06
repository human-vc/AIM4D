# AIM4D

Five-stage econometric-causal framework for forecasting democratic decline. Accepted to the [AIM4D Conference](https://sites.nd.edu/aim3d/knowledge-network/aim4d-conference/) at Notre Dame (October 2026).

**Authors:** Brandon Yee, Jacob Crainic, Krishna Sharma

## Pipeline

| Stage | Method | Output |
|---|---|---|
| 1 | POET factor extraction + Bai-Ng IC + varimax rotation on 332 V-Dem indicators | 4 latent democratic factors (institutional entrenchment, executive overreach, civil society suppression, informational autocratization) |
| 2 | DCC-GARCH + Kalman TVP on first-differenced leave-one-out global factors | Time-varying democratic betas per country-year |
| 3 | 5-state Gaussian HMM with AR(1) lags + TVTP (LASSO-selected GDELT/macro covariates) | Regime state probabilities and transition dynamics |
| 4 | INE-TARNet on spatio-temporal graph (contiguity + alliance + trade networks) | Per-factor contagion scores, domestic vs. network decomposition |
| 5 | Critical slowing down (variance, AR1, kurtosis) on NSCM domestic residuals | Continuous CSD risk index, early warning signals |

## Key Results

- **Stage 1:** 4 factors, 55% cumulative variance, Denmark at top, North Korea at bottom
- **Stage 2:** Hungary betas peak 2018, Poland reverses 2023, Turkey steadily declining
- **Stage 3:** Weighted kappa 0.67 vs V-Dem RoW, GDELT Goldstein scale + repression count drive transitions
- **Stage 4:** Network ablation shows 26.3% MSE improvement from spatial features, Hungary 70% contagion
- **Stage 5:** AUC-ROC 0.74, 62% sensitivity on 8 known episodes, 3.2x lift over random

## Data

- **V-Dem v16** (28K country-years, 4618 indicators) — downloaded via R `vdemdata` package
- **COW Contiguity** (82K dyad-years) — direct download
- **ATOP Alliances v5.1** (137K dyad-years) — direct download
- **World Bank** (GDP, trade, urbanization, military spending, resource rents) — via `wbgapi`
- **GDELT proxies** (protest, conflict, repression, Goldstein, tone) — derived from V-Dem indicators

## Setup

```bash
pip install -r requirements.txt
```

V-Dem data requires R with the `vdemdata` package:
```bash
Rscript -e 'remotes::install_github("vdemins/vdemdata")'
```

## Running

Each stage depends on the previous. Run in order:

```bash
python data/download_vdem.py
python data/download_networks.py
python stage1_factors/extract.py
python stage2_betas/estimate.py
python stage3_msvar/estimate.py
python stage4_nscm/estimate.py
python stage5_ews/estimate.py
```

Stage 3 takes ~10 min (HMM fitting), Stage 4 takes ~5 min (GNN training). Others are fast.

## Structure

```
AIM4D/
  data/                     # Raw data + download scripts
  stage1_factors/           # POET factor extraction
  stage2_betas/             # Time-varying democratic betas
  stage3_msvar/             # Markov-switching regime classification
  stage4_nscm/              # Network causal model + contagion scores
  stage5_ews/               # Early warning signals
  requirements.txt
```

## Citation

Yee, B., Crainic, J., & Sharma, K. (2026). Democratic Factor Betas, Network Contagion, and Early Warning: A Five-Stage Econometric-Causal Framework for Forecasting Democratic Decline. Presented at AIM4D, University of Notre Dame.
