#!/usr/bin/env bash
set -euo pipefail

cd /home/retro/ARON

PY=/home/retro/anaconda3/envs/pyg/bin/python
RUN=experiments/reversegnn-compactness/run_heart_pair_scorer_editor.py

STAMP="${STAMP:-20260516}"
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
  --extra_flag=--split_mode
  --extra_flag=random
  --extra_flag=--skip_oom_epoch
)

if [[ -n "${WAIT_FOR_SESSION}" ]]; then
  echo "[pull-endpoint-grid] waiting for tmux session '${WAIT_FOR_SESSION}' to finish"
  while tmux has-session -t "${WAIT_FOR_SESSION}" 2>/dev/null; do
    sleep 60
  done
fi

run_variant() {
  local name="$1"
  shift
  local prefix="random_two_decoder_pull_endpoint_${name}_${STAMP}"

  echo
  date
  echo "[pull-endpoint-grid] prefix=${prefix} $*"
  "${PY}" "${RUN}" \
    --prefix "${prefix}" \
    "${COMMON[@]}" \
    "$@"
}

echo "[pull-endpoint-grid] started $(date -Is)"
echo "[pull-endpoint-grid] seeds=${SEEDS[*]} datasets=${DATASETS[*]}"

run_variant current_pull100 \
  --editor-pull-strength 1.0 \
  --decoded-graph-aug-bound -1

run_variant softpull025 \
  --editor-pull-strength 0.25 \
  --decoded-graph-aug-bound -1

run_variant softpull010 \
  --editor-pull-strength 0.10 \
  --decoded-graph-aug-bound -1

run_variant c0p_noncompact_soft025 \
  --editor-pull-strength 0.25 \
  --decoded-graph-aug-bound -1 \
  --extra_flag=--decoded_require_c0p_noncompact_endpoint

run_variant c0p_noncompact_soft025_cap010 \
  --editor-pull-strength 0.25 \
  --decoded-graph-aug-bound 0.10 \
  --extra_flag=--decoded_require_c0p_noncompact_endpoint

echo "[pull-endpoint-grid] finished $(date -Is)"
