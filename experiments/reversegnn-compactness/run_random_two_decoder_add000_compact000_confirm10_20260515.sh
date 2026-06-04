#!/usr/bin/env bash
set -euo pipefail

cd /home/retro/ARON

PY=/home/retro/anaconda3/envs/pyg/bin/python
RUN=experiments/reversegnn-compactness/run_heart_pair_scorer_editor.py

STAMP="${STAMP:-20260515}"
SEEDS=(${SEEDS:-0 1 2 3 4 5 6 7 8 9})
DATASETS=(${DATASETS:-cora citeseer})
MAX_WORKERS="${MAX_WORKERS:-1}"
MLP_PAIR_MAX_ROWS="${MLP_PAIR_MAX_ROWS:-4}"
EPOCHS="${EPOCHS:-700}"
WAIT_FOR_SESSION="${WAIT_FOR_SESSION:-}"

PREFIX="random_two_decoder_add000_compact000_confirm10_${STAMP}"

COMMON=(
  --datasets "${DATASETS[@]}"
  --seeds "${SEEDS[@]}"
  --epochs "${EPOCHS}"
  --max-workers "${MAX_WORKERS}"
  --mlp-pair-max-rows "${MLP_PAIR_MAX_ROWS}"
  --configs two_decoder_pred
  --decoded-add-ratio 0.0
  --decoded-remove-ratio 0.0
  --extra_flag=--split_mode
  --extra_flag=random
  --extra_flag=--compactness_weight
  --extra_flag=0.0
  --extra_flag=--skip_oom_epoch
)

if [[ -n "${WAIT_FOR_SESSION}" ]]; then
  echo "[add000-compact000-confirm10] waiting for tmux session '${WAIT_FOR_SESSION}' to finish"
  while tmux has-session -t "${WAIT_FOR_SESSION}" 2>/dev/null; do
    sleep 60
  done
fi

echo "[add000-compact000-confirm10] started $(date -Is)"
echo "[add000-compact000-confirm10] prefix=${PREFIX}"
echo "[add000-compact000-confirm10] seeds=${SEEDS[*]} datasets=${DATASETS[*]}"

"${PY}" "${RUN}" \
  --prefix "${PREFIX}" \
  "${COMMON[@]}"

echo "[add000-compact000-confirm10] finished $(date -Is)"
