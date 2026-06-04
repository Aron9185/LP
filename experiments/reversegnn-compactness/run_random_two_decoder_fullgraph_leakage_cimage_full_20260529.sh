#!/usr/bin/env bash
set -euo pipefail

cd /home/retro/ARON

PY=/home/retro/anaconda3/envs/pyg/bin/python
RUN=experiments/reversegnn-compactness/run_heart_pair_scorer_editor.py

STAMP="${STAMP:-20260529}"
SEEDS=(${SEEDS:-0 1 2})
DATASETS=(${DATASETS:-cora citeseer})
MAX_WORKERS="${MAX_WORKERS:-1}"
MLP_PAIR_MAX_ROWS="${MLP_PAIR_MAX_ROWS:-16}"
EPOCHS="${EPOCHS:-700}"
FEAT_MASK_RATIO="${FEAT_MASK_RATIO:-0.1}"
CIMAGE_EDGE_MASK_RATE="${CIMAGE_EDGE_MASK_RATE:-0.3}"
CIMAGE_FACTOR_WEIGHT="${CIMAGE_FACTOR_WEIGHT:-0.1}"
CIMAGE_CLUSTER_WEIGHT="${CIMAGE_CLUSTER_WEIGHT:-0.1}"
CIMAGE_NUM_FACTORS="${CIMAGE_NUM_FACTORS:-8}"
CIMAGE_NUM_CLUSTERS="${CIMAGE_NUM_CLUSTERS:-16}"
CIMAGE_PSEUDO_LABEL_THRESHOLD="${CIMAGE_PSEUDO_LABEL_THRESHOLD:-0.90}"
CIMAGE_FACTOR_SELECT_RATIO="${CIMAGE_FACTOR_SELECT_RATIO:-0.50}"
CIMAGE_MRMR_REDUNDANCY_WEIGHT="${CIMAGE_MRMR_REDUNDANCY_WEIGHT:-0.20}"
CIMAGE_CLUSTER_BALANCE_WEIGHT="${CIMAGE_CLUSTER_BALANCE_WEIGHT:-0.05}"
CIMAGE_SCE_POWER="${CIMAGE_SCE_POWER:-2.0}"
FORCE="${FORCE:-0}"

COMMON=(
  --datasets "${DATASETS[@]}"
  --seeds "${SEEDS[@]}"
  --epochs "${EPOCHS}"
  --max-workers "${MAX_WORKERS}"
  --mlp-pair-max-rows "${MLP_PAIR_MAX_ROWS}"
  --configs two_decoder_pred
  --decoded-remove-ratio 0.0
  --editor-pull-strength 0.25
  --extra_flag=--skip_oom_epoch
  --extra_flag=--feat_mask_ratio
  --extra_flag="${FEAT_MASK_RATIO}"
  --extra_flag=--lp_train_graph
  --extra_flag=full
  --extra_flag=--decoded_require_c0p_noncompact_endpoint
  --extra_flag=--decoded_add_degree_target
  --extra_flag=1
  --extra_flag=--decoded_add_degree_target_scope
  --extra_flag=intra_cluster
  --extra_flag=--decoded_add_degree_target_nodes
  --extra_flag=cp
  --extra_flag=--decoded_guarantee_degree_target
  --extra_flag=--ae_backbone
  --extra_flag=cimage_full
  --extra_flag=--maskgae_mask_rate
  --extra_flag="${CIMAGE_EDGE_MASK_RATE}"
  --extra_flag=--maskgae_feature_weight
  --extra_flag=0.0
  --extra_flag=--cimage_factor_weight
  --extra_flag="${CIMAGE_FACTOR_WEIGHT}"
  --extra_flag=--cimage_cluster_weight
  --extra_flag="${CIMAGE_CLUSTER_WEIGHT}"
  --extra_flag=--cimage_num_factors
  --extra_flag="${CIMAGE_NUM_FACTORS}"
  --extra_flag=--cimage_num_clusters
  --extra_flag="${CIMAGE_NUM_CLUSTERS}"
  --extra_flag=--cimage_pseudo_label_threshold
  --extra_flag="${CIMAGE_PSEUDO_LABEL_THRESHOLD}"
  --extra_flag=--cimage_factor_select_ratio
  --extra_flag="${CIMAGE_FACTOR_SELECT_RATIO}"
  --extra_flag=--cimage_mrmr_redundancy_weight
  --extra_flag="${CIMAGE_MRMR_REDUNDANCY_WEIGHT}"
  --extra_flag=--cimage_cluster_balance_weight
  --extra_flag="${CIMAGE_CLUSTER_BALANCE_WEIGHT}"
  --extra_flag=--cimage_sce_power
  --extra_flag="${CIMAGE_SCE_POWER}"
)

FORCE_ARGS=()
if [[ "${FORCE}" == "1" ]]; then
  FORCE_ARGS=(--force)
fi

run_variant() {
  local name="$1"
  local add_ratio="$2"
  local cap="$3"
  local prefix="random_two_decoder_fullgraph_leakage_cimage_full_${name}_${STAMP}"

  echo
  date
  echo "[fullgraph-leakage-cimage-full] prefix=${prefix} lp_train_graph=full add_ratio=${add_ratio} cap=${cap} edge_mask=${CIMAGE_EDGE_MASK_RATE} factor_w=${CIMAGE_FACTOR_WEIGHT} cluster_w=${CIMAGE_CLUSTER_WEIGHT} pseudo_thr=${CIMAGE_PSEUDO_LABEL_THRESHOLD}"
  "${PY}" "${RUN}" \
    --prefix "${prefix}" \
    --split-mode random \
    "${COMMON[@]}" \
    "${FORCE_ARGS[@]}" \
    --decoded-add-ratio "${add_ratio}" \
    --decoded-graph-aug-bound "${cap}"
}

echo "[fullgraph-leakage-cimage-full] started $(date -Is)"
echo "[fullgraph-leakage-cimage-full] seeds=${SEEDS[*]} datasets=${DATASETS[*]}"

# Match the current CP target-1 repair setting, changing the AE backbone to
# CIMAGE-full and the LP training graph to the full observed graph.
run_variant c0p_noncompact_soft025_cap010_cp_dtarget1_guarantee_addr020 0.20 0.10

echo "[fullgraph-leakage-cimage-full] finished $(date -Is)"
