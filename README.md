# AIM4D

Five-stage econometric-causal framework for forecasting democratic decline.

## Pipeline

| Stage | Method | Output |
|---|---|---|
| 1 | POET factor extraction + Bai-Ng IC + varimax rotation on 332 V-Dem indicators | 4 latent democratic factors (institutional entrenchment, executive overreach, civil society suppression, informational autocratization) |
| 2 | DCC-GARCH + Kalman TVP on first-differenced leave-one-out global factors | Time-varying democratic betas per country-year |
| 3 | 5-state Gaussian HMM with AR(1) lags + TVTP (LASSO-selected GDELT/macro covariates) | Regime state probabilities and transition dynamics |
| 4 | INE-TARNet on spatio-temporal graph (contiguity + alliance + trade networks) | Per-factor contagion scores, domestic vs. network decomposition |
| 5 | Critical slowing down (variance, AR1, kurtosis) on NSCM domestic residuals | Continuous CSD risk index, early warning signals |

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
