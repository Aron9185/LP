#!/usr/bin/env bash
set -euo pipefail

cd /home/retro/ARON

PY="${PY:-/home/retro/anaconda3/envs/pyg/bin/python}"
RUN="${RUN:-experiments/reversegnn-compactness/run_heart_pair_scorer_editor.py}"

STAMP="${STAMP:-20260607}"
SEEDS=(${SEEDS:-0 3})
DATASETS=(${DATASETS:-cora})
MAX_WORKERS="${MAX_WORKERS:-1}"
MLP_PAIR_MAX_ROWS="${MLP_PAIR_MAX_ROWS:-16}"
EPOCHS="${EPOCHS:-700}"
FEAT_MASK_RATIO="${FEAT_MASK_RATIO:-0.1}"

COMMON=(
  --split-mode random
  --random-checkpoint-metric hit10
  --datasets "${DATASETS[@]}"
  --seeds "${SEEDS[@]}"
  --epochs "${EPOCHS}"
  --max-workers "${MAX_WORKERS}"
  --mlp-pair-max-rows "${MLP_PAIR_MAX_ROWS}"
  --configs two_decoder_ncnc_pred
  --decoded-remove-ratio 0.0
  --editor-pull-strength 0.25
  --decoded-add-ratio 0.20
  --decoded-graph-aug-bound 0.10
  --eval-log-every 50
  --decoder-diag-every 50
  --edit-metric-every 50
  --decoded-audit-every 50
  --decoded-audit-max-edges 4096
  --skip-train-acc
  --extra_flag=--skip_oom_epoch
  --extra_flag=--feat_mask_ratio
  --extra_flag="${FEAT_MASK_RATIO}"
  --extra_flag=--decoded_require_c0p_noncompact_endpoint
  --extra_flag=--decoded_add_degree_target
  --extra_flag=1
  --extra_flag=--decoded_add_degree_target_scope
  --extra_flag=intra_cluster
  --extra_flag=--decoded_add_degree_target_nodes
  --extra_flag=cp
  --extra_flag=--decoded_guarantee_degree_target
)

run_variant() {
  local tag="$1"
  shift
  local prefix="random_rewrite_quality_${tag}_${STAMP}"
  echo
  date
  echo "[rewrite-quality] prefix=${prefix} extra=$*"
  "${PY}" "${RUN}" \
    --prefix "${prefix}" \
    "${COMMON[@]}" \
    "$@"
}

echo "[rewrite-quality] started $(date -Is)"
echo "[rewrite-quality] datasets=${DATASETS[*]} seeds=${SEEDS[*]} epochs=${EPOCHS}"
echo "[rewrite-quality] workers=${MAX_WORKERS} mlp_pair_max_rows=${MLP_PAIR_MAX_ROWS}"

run_variant current
run_variant aq002 --extra_flag=--decoded_add_quantile --extra_flag=0.02
run_variant aq005 --extra_flag=--decoded_add_quantile --extra_flag=0.05

echo "[rewrite-quality] finished $(date -Is)"
