#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/retro/ARON"
PY="/home/retro/anaconda3/envs/pyg/bin/python"
RUNNER="${ROOT}/experiments/reversegnn-compactness/run_heart_pair_scorer_editor.py"
LOG_DIR="${ROOT}/experiments/reversegnn-compactness/results"
STAMP="${STAMP:-20260501}"
LOG_FILE="${LOG_DIR}/heart_two_decoder_remove_smoke_${STAMP}_tmux.log"

EPOCHS="${EPOCHS:-150}"
SEEDS=(${SEEDS:-0})
DATASETS=(${DATASETS:-cora citeseer})
MAX_WORKERS="${MAX_WORKERS:-2}"
MLP_PAIR_MAX_ROWS="${MLP_PAIR_MAX_ROWS:-4}"

mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

cd "${ROOT}"

echo "[two-decoder-remove-smoke] started $(date -Is)"
echo "[two-decoder-remove-smoke] log=${LOG_FILE}"
echo "[two-decoder-remove-smoke] epochs=${EPOCHS} datasets=${DATASETS[*]} seeds=${SEEDS[*]} workers=${MAX_WORKERS}"
echo "[two-decoder-remove-smoke] goal: prove actual [EDIT-GRAPH] remove > 0 before full harder-remove runs"

run_case() {
  local name="$1"
  shift
  local prefix="heart_two_decoder_remove_smoke_${name}_${STAMP}"

  echo
  echo "[case] ${name} prefix=${prefix}"
  "${PY}" "${RUNNER}" \
    --prefix "${prefix}" \
    --datasets "${DATASETS[@]}" \
    --seeds "${SEEDS[@]}" \
    --epochs "${EPOCHS}" \
    --max-workers "${MAX_WORKERS}" \
    --mlp-pair-max-rows "${MLP_PAIR_MAX_ROWS}" \
    --configs two_decoder_pred_remove \
    --decoded-add-ratio 0.0 \
    --decoded-remove-ratio 0.03 \
    --extra_flag=--skip_oom_epoch \
    "$@"
}

# Control: current constraints plus stronger ratio. This should tell us whether
# the old path was simply under-budgeted.
run_case "default_r03"

# Same cluster/C0p constraints, but allow degree-floor removals.
run_case "floor0_r03" \
  --extra_flag=--decoded_degree_floor \
  --extra_flag=0

# Force the bottom 5% of existing-edge scores, capped for safety.
run_case "floor0_q05_mr100" \
  --decoded-remove-ratio 0.0 \
  --extra_flag=--decoded_degree_floor \
  --extra_flag=0 \
  --extra_flag=--decoded_remove_quantile \
  --extra_flag=0.05 \
  --extra_flag=--decoded_max_remove_per_round \
  --extra_flag=100

# Relax cluster/C0p constraints as a last sanity check. This affects add and
# remove globally, but add_ratio=0 here, so only removal is relaxed.
run_case "relaxed_floor0_q05_mr100" \
  --decoded-remove-ratio 0.0 \
  --extra_flag=--decoded_degree_floor \
  --extra_flag=0 \
  --extra_flag=--decoded_remove_quantile \
  --extra_flag=0.05 \
  --extra_flag=--decoded_max_remove_per_round \
  --extra_flag=100 \
  --extra_flag=--decoded_allow_cross_cluster \
  --extra_flag=--decoded_no_c0p_endpoint

echo
echo "[two-decoder-remove-smoke] finished $(date -Is)"
