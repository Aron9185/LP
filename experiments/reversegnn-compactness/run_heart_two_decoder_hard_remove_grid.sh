#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/retro/ARON"
PY="/home/retro/anaconda3/envs/pyg/bin/python"
RUNNER="${ROOT}/experiments/reversegnn-compactness/run_heart_pair_scorer_editor.py"
LOG_DIR="${ROOT}/experiments/reversegnn-compactness/results"
STAMP="${STAMP:-20260501}"
LOG_FILE="${LOG_DIR}/heart_two_decoder_hard_remove_grid_${STAMP}_tmux.log"

EPOCHS="${EPOCHS:-700}"
SEEDS=(${SEEDS:-0 1 2})
DATASETS=(${DATASETS:-cora citeseer})
MAX_WORKERS="${MAX_WORKERS:-2}"
MLP_PAIR_MAX_ROWS="${MLP_PAIR_MAX_ROWS:-4}"
CONFIGS=(${CONFIGS:-two_decoder_pred_remove})
RELAX_REMOVE_CONSTRAINTS="${RELAX_REMOVE_CONSTRAINTS:-0}"

mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

cd "${ROOT}"

echo "[two-decoder-hard-remove-grid] started $(date -Is)"
echo "[two-decoder-hard-remove-grid] log=${LOG_FILE}"
echo "[two-decoder-hard-remove-grid] epochs=${EPOCHS} datasets=${DATASETS[*]} seeds=${SEEDS[*]} workers=${MAX_WORKERS}"
echo "[two-decoder-hard-remove-grid] configs=${CONFIGS[*]} relax_remove_constraints=${RELAX_REMOVE_CONSTRAINTS}"

extra_common=(
  --extra_flag=--skip_oom_epoch
  --extra_flag=--decoded_degree_floor
  --extra_flag=0
)

if [[ "${RELAX_REMOVE_CONSTRAINTS}" == "1" ]]; then
  # The runner has add_ratio controlled per case. For remove-heavy testing this
  # lets removals consider all existing edges instead of only same-cluster/C0p.
  extra_common+=(
    --extra_flag=--decoded_allow_cross_cluster
    --extra_flag=--decoded_no_c0p_endpoint
  )
fi

run_case() {
  local name="$1"
  local add_ratio="$2"
  local remove_ratio="$3"
  local prefix="heart_two_decoder_hard_remove_${name}_${STAMP}"

  echo
  echo "[case] ${name} add=${add_ratio} remove=${remove_ratio} prefix=${prefix}"
  "${PY}" "${RUNNER}" \
    --prefix "${prefix}" \
    --datasets "${DATASETS[@]}" \
    --seeds "${SEEDS[@]}" \
    --epochs "${EPOCHS}" \
    --max-workers "${MAX_WORKERS}" \
    --mlp-pair-max-rows "${MLP_PAIR_MAX_ROWS}" \
    --configs "${CONFIGS[@]}" \
    --decoded-add-ratio "${add_ratio}" \
    --decoded-remove-ratio "${remove_ratio}" \
    "${extra_common[@]}"
}

run_case "floor0_add001_rm001" 0.01 0.01
run_case "floor0_add001_rm003" 0.01 0.03
run_case "floor0_add001_rm005" 0.01 0.05
run_case "floor0_add0005_rm005" 0.005 0.05

echo
echo "[two-decoder-hard-remove-grid] finished $(date -Is)"
