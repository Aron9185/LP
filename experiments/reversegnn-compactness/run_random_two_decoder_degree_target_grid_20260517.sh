#!/usr/bin/env bash
set -euo pipefail

cd /home/retro/ARON

PY=/home/retro/anaconda3/envs/pyg/bin/python
RUN=experiments/reversegnn-compactness/run_heart_pair_scorer_editor.py

STAMP="${STAMP:-20260517}"
SEEDS=(${SEEDS:-0 1 2 3 4})
DATASETS=(${DATASETS:-cora citeseer})
MAX_WORKERS="${MAX_WORKERS:-1}"
MLP_PAIR_MAX_ROWS="${MLP_PAIR_MAX_ROWS:-4}"
EPOCHS="${EPOCHS:-700}"
WAIT_FOR_SESSION="${WAIT_FOR_SESSION:-}"

COMMON=(
  --datasets "${DATASETS[@]}"
  --seeds "${SEEDS[@]}"
  --epochs "${EPOCHS}"
  --max-workers "${MAX_WORKERS}"
  --mlp-pair-max-rows "${MLP_PAIR_MAX_ROWS}"
  --configs two_decoder_pred
  --decoded-add-ratio 0.01
  --decoded-remove-ratio 0.0
  --editor-pull-strength 0.25
  --decoded-graph-aug-bound 0.10
  --extra_flag=--split_mode
  --extra_flag=random
  --extra_flag=--skip_oom_epoch
  --extra_flag=--decoded_require_c0p_noncompact_endpoint
)

if [[ -n "${WAIT_FOR_SESSION}" ]]; then
  echo "[degree-target-grid] waiting for tmux session '${WAIT_FOR_SESSION}' to finish"
  while tmux has-session -t "${WAIT_FOR_SESSION}" 2>/dev/null; do
    sleep 60
  done
fi

run_variant() {
  local name="$1"
  local target="$2"
  local prefix="random_two_decoder_degree_target_${name}_${STAMP}"

  echo
  date
  echo "[degree-target-grid] prefix=${prefix} target=${target}"
  "${PY}" "${RUN}" \
    --prefix "${prefix}" \
    "${COMMON[@]}" \
    --extra_flag=--decoded_add_degree_target \
    --extra_flag="${target}"
}

echo "[degree-target-grid] started $(date -Is)"
echo "[degree-target-grid] seeds=${SEEDS[*]} datasets=${DATASETS[*]}"

run_variant c0p_noncompact_soft025_cap010_dtarget2 2
run_variant c0p_noncompact_soft025_cap010_dtarget3 3
run_variant c0p_noncompact_soft025_cap010_dtarget4 4

echo "[degree-target-grid] finished $(date -Is)"
