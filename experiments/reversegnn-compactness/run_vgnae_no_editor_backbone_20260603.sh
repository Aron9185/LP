#!/usr/bin/env bash
set -euo pipefail

cd /home/retro/ARON

PY="${PY:-/home/retro/anaconda3/envs/pyg/bin/python}"
RUN="${RUN:-experiments/reversegnn-compactness/run_no_editor_backbone.py}"

STAMP="${STAMP:-20260603}"
SPLIT_MODE="${SPLIT_MODE:-random}"
PREFIX="${PREFIX:-vgnae_no_editor_${SPLIT_MODE}_${STAMP}}"
DATASETS=(${DATASETS:-cora citeseer})
SEEDS=(${SEEDS:-0 1 2 3 4 5 6 7 8 9})
CONFIGS=(${CONFIGS:-dot pred})
EPOCHS="${EPOCHS:-700}"
MAX_WORKERS="${MAX_WORKERS:-1}"
EVAL_LOG_EVERY="${EVAL_LOG_EVERY:-5}"
FEAT_MASK_RATIO="${FEAT_MASK_RATIO:-0.1}"

echo "[vgnae-no-editor] started $(date -Is)"
echo "[vgnae-no-editor] prefix=${PREFIX}"
echo "[vgnae-no-editor] split=${SPLIT_MODE} datasets=${DATASETS[*]} seeds=${SEEDS[*]} configs=${CONFIGS[*]}"
echo "[vgnae-no-editor] epochs=${EPOCHS} workers=${MAX_WORKERS} eval_every=${EVAL_LOG_EVERY}"

"${PY}" "${RUN}" \
  --prefix "${PREFIX}" \
  --split-mode "${SPLIT_MODE}" \
  --backbones vgnae \
  --configs "${CONFIGS[@]}" \
  --datasets "${DATASETS[@]}" \
  --seeds "${SEEDS[@]}" \
  --epochs "${EPOCHS}" \
  --max-workers "${MAX_WORKERS}" \
  --eval-log-every "${EVAL_LOG_EVERY}" \
  --feat-mask-ratio "${FEAT_MASK_RATIO}"

echo "[vgnae-no-editor] finished $(date -Is)"

