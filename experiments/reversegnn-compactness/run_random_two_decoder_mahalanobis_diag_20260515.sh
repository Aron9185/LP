#!/usr/bin/env bash
set -euo pipefail

cd /home/retro/ARON

PY=/home/retro/anaconda3/envs/pyg/bin/python
RUN=experiments/reversegnn-compactness/run_heart_pair_scorer_editor.py

STAMP="${STAMP:-20260515}"
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
  --decoded-remove-ratio 0.0
  --compactness-radius-metric mahalanobis
  --extra_flag=--split_mode
  --extra_flag=random
  --extra_flag=--skip_oom_epoch
)

if [[ -n "${WAIT_FOR_SESSION}" ]]; then
  echo "[mahalanobis-diag] waiting for tmux session '${WAIT_FOR_SESSION}' to finish"
  while tmux has-session -t "${WAIT_FOR_SESSION}" 2>/dev/null; do
    sleep 60
  done
fi

run_variant() {
  local name="$1"
  shift
  local prefix="random_two_decoder_mahalanobis_${name}_${STAMP}"

  echo
  date
  echo "[mahalanobis-diag] prefix=${prefix} $*"
  "${PY}" "${RUN}" \
    --prefix "${prefix}" \
    "${COMMON[@]}" \
    "$@"
}

echo "[mahalanobis-diag] started $(date -Is)"
echo "[mahalanobis-diag] seeds=${SEEDS[*]} datasets=${DATASETS[*]}"

run_variant current \
  --decoded-add-ratio 0.01 \
  --extra_flag=--compactness_weight \
  --extra_flag=0.2

run_variant add000_compact000 \
  --decoded-add-ratio 0.0 \
  --extra_flag=--compactness_weight \
  --extra_flag=0.0

echo "[mahalanobis-diag] finished $(date -Is)"
