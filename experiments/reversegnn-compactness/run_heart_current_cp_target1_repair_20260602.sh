#!/usr/bin/env bash
set -euo pipefail

cd /home/retro/ARON

PY=/home/retro/anaconda3/envs/pyg/bin/python
RUN=experiments/reversegnn-compactness/run_heart_pair_scorer_editor.py

STAMP="${STAMP:-20260602}"
PREFIX="${PREFIX:-heart_current_cp_target1_repair_${STAMP}}"
SEEDS=(${SEEDS:-0 1 2 3 4 5 6 7 8 9})
DATASETS=(${DATASETS:-cora citeseer})
MAX_WORKERS="${MAX_WORKERS:-1}"
MLP_PAIR_MAX_ROWS="${MLP_PAIR_MAX_ROWS:-16}"
EPOCHS="${EPOCHS:-700}"
FEAT_MASK_RATIO="${FEAT_MASK_RATIO:-0.1}"

echo "[heart-current-cp-target1] started $(date -Is)"
echo "[heart-current-cp-target1] prefix=${PREFIX}"
echo "[heart-current-cp-target1] datasets=${DATASETS[*]} seeds=${SEEDS[*]} epochs=${EPOCHS}"
echo "[heart-current-cp-target1] workers=${MAX_WORKERS} mlp_pair_max_rows=${MLP_PAIR_MAX_ROWS}"
echo "[heart-current-cp-target1] split=heart checkpoint=hit10 removal=off"

"${PY}" "${RUN}" \
  --prefix "${PREFIX}" \
  --split-mode heart \
  --datasets "${DATASETS[@]}" \
  --seeds "${SEEDS[@]}" \
  --epochs "${EPOCHS}" \
  --max-workers "${MAX_WORKERS}" \
  --mlp-pair-max-rows "${MLP_PAIR_MAX_ROWS}" \
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

echo "[heart-current-cp-target1] finished $(date -Is)"
