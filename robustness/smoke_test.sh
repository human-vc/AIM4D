#!/usr/bin/env bash
# Smoke-test every robustness script with AIM4D_QUICK=1.
# Verifies each script runs end-to-end and writes its expected output CSV.
# Each script gets a timeout; failures are reported but don't abort the run.
#
# Usage:
#   bash robustness/smoke_test.sh           # runs all
#   bash robustness/smoke_test.sh fast      # skip the GNN counterfactual
#
# Expected total runtime: ~10-15 min with AIM4D_QUICK=1.

set -u
cd "$(dirname "$0")/.." || exit 1

export AIM4D_QUICK=1
export AIM4D_HMM_RESTARTS=10

LOG_DIR="brev"
mkdir -p "$LOG_DIR"
SKIP_GNN="${1:-}"

PASSED=()
FAILED=()

run_one() {
    local name="$1"
    local cmd="$2"
    local timeout_s="$3"
    local out_csv="$4"
    echo
    echo "=== SMOKE-TEST: $name (timeout ${timeout_s}s) ==="
    if timeout "${timeout_s}" bash -c "$cmd" > "$LOG_DIR/smoke_${name}.log" 2>&1; then
        if [ -n "$out_csv" ] && [ ! -f "$out_csv" ]; then
            echo "  FAIL: ran but did not write $out_csv"
            FAILED+=("$name (no output)")
            tail -20 "$LOG_DIR/smoke_${name}.log"
        else
            echo "  PASS"
            PASSED+=("$name")
        fi
    else
        echo "  FAIL (exit $? or timeout)"
        FAILED+=("$name")
        tail -20 "$LOG_DIR/smoke_${name}.log"
    fi
}

# 1. lead_time_auc — fast, reads existing ews_signals.csv
run_one "lead_time_auc" \
    "python3 -u robustness/lead_time_auc.py" \
    120 \
    "robustness/lead_time_auc.csv"

# 2. alternate_labels — fast, reads existing ews_signals.csv + V-Dem
run_one "alternate_labels" \
    "python3 -u robustness/alternate_labels.py" \
    180 \
    "robustness/alternate_labels.csv"

# 3. permutation_importance_oos — refits ensemble, ~3-5 min in QUICK mode
run_one "permutation_importance_oos" \
    "python3 -u robustness/permutation_importance_oos.py" \
    600 \
    "robustness/permutation_importance_oos.csv"

# 4. elastic_net_robustness — 2 Stage 5 reruns in QUICK mode
run_one "elastic_net_robustness" \
    "python3 -u robustness/elastic_net_robustness.py" \
    600 \
    "robustness/elastic_net_robustness.csv"

# 5. dsp_imputation_robustness — 2 Stage 5 reruns in QUICK mode
run_one "dsp_imputation_robustness" \
    "python3 -u robustness/dsp_imputation_robustness.py" \
    600 \
    "robustness/dsp_imputation_robustness.csv"

# 6. hyperparameter_sensitivity — 13 Stage 5 reruns; QUICK mode keeps it tractable
run_one "hyperparameter_sensitivity" \
    "python3 -u robustness/hyperparameter_sensitivity.py" \
    1800 \
    "robustness/hyperparameter_sensitivity.csv"

# 7. gnn_counterfactual — refits Stage 4 once; skipped in 'fast' mode
if [ "$SKIP_GNN" != "fast" ]; then
    run_one "gnn_counterfactual" \
        "python3 -u robustness/gnn_counterfactual.py" \
        900 \
        "robustness/gnn_counterfactual.csv"
else
    echo "Skipping gnn_counterfactual (fast mode)"
fi

# 8. Task F sample (heavy) — verify 1 episode runs without crash
run_one "sample_pipeline_loeo_smoke" \
    "AIM4D_SMOKE_LIMIT=1 python3 -u robustness/sample_pipeline_loeo.py" \
    1800 \
    ""

echo
echo "================================================================"
echo "SMOKE TEST SUMMARY"
echo "================================================================"
echo "PASSED (${#PASSED[@]}): ${PASSED[*]}"
echo "FAILED (${#FAILED[@]}): ${FAILED[*]}"
if [ ${#FAILED[@]} -gt 0 ]; then
    echo
    echo "Logs for failed runs:"
    for f in "${FAILED[@]}"; do
        echo "  $LOG_DIR/smoke_${f%% *}.log"
    done
    exit 1
fi
echo
echo "All smoke tests passed. Safe to run full configurations on Brev."
