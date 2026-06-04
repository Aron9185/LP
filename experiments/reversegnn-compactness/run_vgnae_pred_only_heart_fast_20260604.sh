#!/usr/bin/env bash
set -euo pipefail

cd /home/retro/ARON

PY="${PY:-/home/retro/anaconda3/envs/pyg/bin/python}"
RUN="${RUN:-experiments/reversegnn-compactness/run_no_editor_backbone.py}"

STAMP="${STAMP:-20260604}"
PREFIX="${PREFIX:-vgnae_pred_only_heart_fast_${STAMP}}"
DATASETS=(${DATASETS:-cora citeseer})
SEEDS=(${SEEDS:-0 1 2})
EPOCHS="${EPOCHS:-700}"
MAX_WORKERS="${MAX_WORKERS:-1}"
MLP_PAIR_MAX_ROWS="${MLP_PAIR_MAX_ROWS:-64}"
EVAL_LOG_EVERY="${EVAL_LOG_EVERY:-20}"
TRAIN_EVAL_EVERY="${TRAIN_EVAL_EVERY:-20}"
HEART_EVAL_EVERY="${HEART_EVAL_EVERY:-20}"
HEART_VAL_FRAC="${HEART_VAL_FRAC:-0.10}"
DECODER_DIAG_EVERY="${DECODER_DIAG_EVERY:-0}"
FEAT_MASK_RATIO="${FEAT_MASK_RATIO:-0.1}"

echo "[vgnae-pred-only-heart-fast] started $(date -Is)"
echo "[vgnae-pred-only-heart-fast] prefix=${PREFIX}"
echo "[vgnae-pred-only-heart-fast] datasets=${DATASETS[*]} seeds=${SEEDS[*]} epochs=${EPOCHS}"
echo "[vgnae-pred-only-heart-fast] eval_log=${EVAL_LOG_EVERY} train_eval=${TRAIN_EVAL_EVERY} heart_eval=${HEART_EVAL_EVERY} heart_val_frac=${HEART_VAL_FRAC} mlp_pair_max_rows=${MLP_PAIR_MAX_ROWS}"

"${PY}" "${RUN}" \
  --prefix "${PREFIX}" \
  --split-mode heart \
  --backbones vgnae \
  --configs pred \
  --datasets "${DATASETS[@]}" \
  --seeds "${SEEDS[@]}" \
  --epochs "${EPOCHS}" \
  --max-workers "${MAX_WORKERS}" \
  --eval-log-every "${EVAL_LOG_EVERY}" \
  --train-eval-every "${TRAIN_EVAL_EVERY}" \
  --heart-eval-every "${HEART_EVAL_EVERY}" \
  --heart-val-frac "${HEART_VAL_FRAC}" \
  --decoder-diag-every "${DECODER_DIAG_EVERY}" \
  --skip-train-acc \
  --feat-mask-ratio "${FEAT_MASK_RATIO}" \
  --extra_flag=--skip_oom_epoch \
  --extra_flag=--mlp_pair_max_rows \
  --extra_flag="${MLP_PAIR_MAX_ROWS}"

echo "[vgnae-pred-only-heart-fast] finished $(date -Is)"
