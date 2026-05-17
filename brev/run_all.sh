#!/usr/bin/env bash
# Full AIM4D pipeline + robustness suite on Brev.
# Streams all output live to stdout AND to brev/run_all.log (use `tail`
# from another shell if you disconnect; the file persists).
#
# Usage:
#   bash brev/run_all.sh           # everything (fast + slow)
#   FAST_ONLY=1 bash brev/run_all.sh    # skip Task E and Task F (4-8 hr saved)
#   SKIP_GDELT=1 bash brev/run_all.sh   # skip GDELT re-download if cached
#
# Expected wall-time (16 vCPU Brev):
#   fast loop (stages 1-5 + dsp_ablation + bootstrap + contagion sweep): ~1 hr
#   + Task E (4 folds, full refit per fold): ~2-3 hr
#   + Task F (5 episodes, full refit per episode): ~2-3 hr
#   TOTAL: ~5-7 hr

set -euo pipefail
cd "$(dirname "$0")/.."
LOG="brev/run_all.log"
: > "$LOG"

# unbuffered output so we see logs live in both stdout and the file
exec > >(stdbuf -oL tee -a "$LOG") 2>&1

echo "================================================================"
echo "AIM4D full pipeline run started $(date -u +%FT%TZ)"
echo "Host: $(hostname)  CPUs: $(nproc 2>/dev/null || echo '?')  RAM: $(free -h 2>/dev/null | awk '/Mem:/ {print $2}' || echo '?')"
echo "FAST_ONLY=${FAST_ONLY:-0}  SKIP_GDELT=${SKIP_GDELT:-0}"
echo "================================================================"

# --------- environment ---------
if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

# --------- data prerequisites ---------
# V-Dem, COW contiguity, ATOP must already be on disk (uploaded via brev/upload.sh).
# GDELT can be regenerated if missing.
for f in data/vdem_v16.csv data/contiguity/DirectContiguity320 data/atop; do
  if [[ ! -e $f ]]; then
    echo "[ERROR] Missing $f. Run brev/upload.sh from your laptop first." >&2
    exit 1
  fi
done

if [[ ! -f data/gdelt_country_year.csv ]] && [[ "${SKIP_GDELT:-0}" != "1" ]]; then
  echo "--- Downloading GDELT (first run only; ~30 min on 16 vCPU) ---"
  python3 -u data/download_gdelt.py
fi

# --------- canonical pipeline (cutoff=2019) ---------
echo
echo "================================================================"
echo "STAGE 1: POET factor extraction"
echo "================================================================"
python3 -u stage1_factors/extract.py

echo
echo "================================================================"
echo "STAGE 2: Kalman + DCC-GARCH betas"
echo "================================================================"
python3 -u stage2_betas/estimate.py

echo
echo "================================================================"
echo "STAGE 3: MS-VAR HMM regime classification"
echo "================================================================"
python3 -u stage3_msvar/estimate.py

echo
echo "================================================================"
echo "STAGE 4: INE-TARNet network contagion"
echo "================================================================"
python3 -u stage4_nscm/estimate.py

echo
echo "================================================================"
echo "STAGE 5: Multi-channel early warning + meta-learner"
echo "================================================================"
python3 -u stage5_ews/estimate.py

# --------- robustness (fast) ---------
echo
echo "================================================================"
echo "ROBUSTNESS: DSP ablation"
echo "================================================================"
python3 -u robustness/dsp_ablation.py

echo
echo "================================================================"
echo "ROBUSTNESS: cluster-bootstrap 95% CIs"
echo "================================================================"
python3 -u robustness/bootstrap_cis.py

echo
echo "================================================================"
echo "ROBUSTNESS: multi-seed Stage 4 contagion sweep (10 seeds)"
echo "================================================================"
python3 -u robustness/contagion_seed_sweep.py

echo
echo "================================================================"
echo "ROBUSTNESS: network weight stability sweep (10 seeds)"
echo "================================================================"
python3 -u robustness/network_seed_sweep.py || echo "  (skipped — script optional)"

if [[ "${FAST_ONLY:-0}" == "1" ]]; then
  echo
  echo "FAST_ONLY=1 set — skipping Task E and Task F"
  echo "Total wall-time: $(( SECONDS / 60 )) min"
  echo "Output: $LOG"
  exit 0
fi

# --------- expensive validation ---------
echo
echo "================================================================"
echo "TASK E: real expanding-window CV (4 folds, full refit per fold)"
echo "Expected: 2-3 hr"
echo "================================================================"
python3 -u robustness/expanding_window_cv.py

echo
echo "================================================================"
echo "TASK F: 5-episode full-pipeline LOEO"
echo "Expected: 2-3 hr"
echo "================================================================"
python3 -u robustness/sample_pipeline_loeo.py

echo
echo "================================================================"
echo "DONE. Total wall-time: $(( SECONDS / 60 )) min"
echo "Output: $LOG"
echo "================================================================"
