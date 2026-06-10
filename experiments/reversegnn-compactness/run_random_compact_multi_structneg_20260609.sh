#!/usr/bin/env bash
set -euo pipefail

cd /home/retro/ARON

PY="${PY:-/home/retro/anaconda3/envs/pyg/bin/python}"
RUN="${RUN:-experiments/reversegnn-compactness/run_heart_pair_scorer_editor.py}"

STAMP="${STAMP:-20260609}"
SEEDS=(${SEEDS:-0 1 2 4 5})
DATASETS=(${DATASETS:-cora citeseer})
MAX_WORKERS="${MAX_WORKERS:-1}"
MLP_PAIR_MAX_ROWS="${MLP_PAIR_MAX_ROWS:-16}"
EPOCHS="${EPOCHS:-700}"
COMPACTNESS_WEIGHT="${COMPACTNESS_WEIGHT:-0.05}"
PRED_STRUCT_FRAC="${PRED_STRUCT_FRAC:-0.75}"
FEAT_MASK_RATIO="${FEAT_MASK_RATIO:-0.1}"

COMMON=(
  --split-mode random
  --random-checkpoint-metric hit10
  --datasets "${DATASETS[@]}"
  --seeds "${SEEDS[@]}"
  --epochs "${EPOCHS}"
  --max-workers "${MAX_WORKERS}"
  --mlp-pair-max-rows "${MLP_PAIR_MAX_ROWS}"
  --configs two_decoder_compact_multi_pred
  --decoded-remove-ratio 0.0
  --editor-pull-strength 0.25
  --decoded-add-ratio 0.20
  --decoded-graph-aug-bound 0.10
  --prediction-rank-neg-strategy struct
  --prediction-rank-struct-frac "${PRED_STRUCT_FRAC}"
  --eval-log-every 50
  --decoder-diag-every 100
  --edit-metric-every 100
  --decoded-audit-every 100
  --decoded-audit-max-edges 4096
  --skip-train-acc
  --extra_flag=--skip_oom_epoch
  --extra_flag=--compactness_weight
  --extra_flag="${COMPACTNESS_WEIGHT}"
  --extra_flag=--decoded_require_c0p_noncompact_endpoint
  --extra_flag=--decoded_add_degree_target
  --extra_flag=1
  --extra_flag=--decoded_add_degree_target_scope
  --extra_flag=intra_cluster
  --extra_flag=--decoded_add_degree_target_nodes
  --extra_flag=cp
  --extra_flag=--decoded_guarantee_degree_target
  --extra_flag=--hidden1
  --extra_flag=512
  --extra_flag=--hidden2
  --extra_flag=128
  --extra_flag=--dropout
  --extra_flag=0.4
  --extra_flag=--lr
  --extra_flag=0.001
  --extra_flag=--beta
  --extra_flag=1.0
  --extra_flag=--feat_mask_ratio
  --extra_flag="${FEAT_MASK_RATIO}"
)

run_variant() {
  local tag="$1"
  shift
  local prefix="random_compact_multi_structneg_${tag}_${STAMP}"
  echo
  date
  echo "[compact-multi-structneg] prefix=${prefix} datasets=${DATASETS[*]} seeds=${SEEDS[*]} extra=$*"
  "${PY}" "${RUN}" \
    --prefix "${prefix}" \
    "${COMMON[@]}" \
    "$@"
}

echo "[compact-multi-structneg] started $(date -Is)"
echo "[compact-multi-structneg] datasets=${DATASETS[*]} seeds=${SEEDS[*]} epochs=${EPOCHS}"
echo "[compact-multi-structneg] compactness_weight=${COMPACTNESS_WEIGHT} pred_struct_frac=${PRED_STRUCT_FRAC}"

run_variant same_aq002 \
  --extra_flag=--decoded_add_quantile \
  --extra_flag=0.02

run_variant cross_aq002 \
  --extra_flag=--decoded_allow_cross_cluster \
  --extra_flag=--decoded_add_quantile \
  --extra_flag=0.02

echo "[compact-multi-structneg] finished $(date -Is)"
