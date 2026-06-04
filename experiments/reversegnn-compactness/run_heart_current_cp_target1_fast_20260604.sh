#!/usr/bin/env bash
set -euo pipefail

cd /home/retro/ARON

PY="${PY:-/home/retro/anaconda3/envs/pyg/bin/python}"
RUN="${RUN:-experiments/reversegnn-compactness/run_heart_pair_scorer_editor.py}"

STAMP="${STAMP:-20260604}"
PREFIX="${PREFIX:-heart_current_cp_target1_fast_${STAMP}}"
SEEDS=(${SEEDS:-0 1 2})
DATASETS=(${DATASETS:-cora citeseer})
MAX_WORKERS="${MAX_WORKERS:-1}"
MLP_PAIR_MAX_ROWS="${MLP_PAIR_MAX_ROWS:-64}"
EPOCHS="${EPOCHS:-700}"
FEAT_MASK_RATIO="${FEAT_MASK_RATIO:-0.1}"
EVAL_LOG_EVERY="${EVAL_LOG_EVERY:-20}"
TRAIN_EVAL_EVERY="${TRAIN_EVAL_EVERY:-20}"
HEART_EVAL_EVERY="${HEART_EVAL_EVERY:-20}"
HEART_VAL_FRAC="${HEART_VAL_FRAC:-0.10}"
DECODER_DIAG_EVERY="${DECODER_DIAG_EVERY:-0}"
EDIT_METRIC_EVERY="${EDIT_METRIC_EVERY:-20}"
DECODED_REWRITE_EVERY="${DECODED_REWRITE_EVERY:-5}"

echo "[heart-current-cp-target1-fast] started $(date -Is)"
echo "[heart-current-cp-target1-fast] prefix=${PREFIX}"
echo "[heart-current-cp-target1-fast] datasets=${DATASETS[*]} seeds=${SEEDS[*]} epochs=${EPOCHS}"
echo "[heart-current-cp-target1-fast] workers=${MAX_WORKERS} mlp_pair_max_rows=${MLP_PAIR_MAX_ROWS}"
echo "[heart-current-cp-target1-fast] eval_log=${EVAL_LOG_EVERY} train_eval=${TRAIN_EVAL_EVERY} heart_eval=${HEART_EVAL_EVERY} heart_val_frac=${HEART_VAL_FRAC} decoder_diag=${DECODER_DIAG_EVERY} edit_metric=${EDIT_METRIC_EVERY} rewrite_every=${DECODED_REWRITE_EVERY}"

"${PY}" "${RUN}" \
  --prefix "${PREFIX}" \
  --split-mode heart \
  --datasets "${DATASETS[@]}" \
  --seeds "${SEEDS[@]}" \
  --epochs "${EPOCHS}" \
  --max-workers "${MAX_WORKERS}" \
  --mlp-pair-max-rows "${MLP_PAIR_MAX_ROWS}" \
  --decoded-rewrite-every "${DECODED_REWRITE_EVERY}" \
  --eval-log-every "${EVAL_LOG_EVERY}" \
  --train-eval-every "${TRAIN_EVAL_EVERY}" \
  --heart-eval-every "${HEART_EVAL_EVERY}" \
  --heart-val-frac "${HEART_VAL_FRAC}" \
  --decoder-diag-every "${DECODER_DIAG_EVERY}" \
  --edit-metric-every "${EDIT_METRIC_EVERY}" \
  --skip-train-acc \
  --configs two_decoder_pred \
  --decoded-remove-ratio 0.0 \
  --editor-pull-strength 0.25 \
  --compactness-mask-scope cp \
  --rewrite-endpoint-scope c0p \
  --decoded-add-ratio 0.20 \
  --decoded-graph-aug-bound 0.10 \
  --extra_flag=--skip_oom_epoch \
  --extra_flag=--feat_mask_ratio \
  --extra_flag="${FEAT_MASK_RATIO}" \
  --extra_flag=--decoded_require_c0p_noncompact_endpoint \
  --extra_flag=--decoded_add_degree_target \
  --extra_flag=1 \
  --extra_flag=--decoded_add_degree_target_scope \
  --extra_flag=intra_cluster \
  --extra_flag=--decoded_add_degree_target_nodes \
  --extra_flag=cp \
  --extra_flag=--decoded_guarantee_degree_target

echo "[heart-current-cp-target1-fast] finished $(date -Is)"
