#!/usr/bin/env bash
set -euo pipefail

cd /home/retro/ARON

PY=/home/retro/anaconda3/envs/pyg/bin/python
RUN=experiments/reversegnn-compactness/run_heart_pair_scorer_editor.py

STAMP="${STAMP:-20260526}"
SEEDS=(${SEEDS:-0 1 2})
DATASETS=(${DATASETS:-cora citeseer})
MAX_WORKERS="${MAX_WORKERS:-1}"
MLP_PAIR_MAX_ROWS="${MLP_PAIR_MAX_ROWS:-16}"
EPOCHS="${EPOCHS:-700}"
FEAT_MASK_RATIO="${FEAT_MASK_RATIO:-0.1}"
MASKGAE_MASK_RATE="${MASKGAE_MASK_RATE:-0.3}"
MASKGAE_FEATURE_WEIGHT="${MASKGAE_FEATURE_WEIGHT:-0.0}"
CIMAGE_FACTOR_WEIGHT="${CIMAGE_FACTOR_WEIGHT:-0.1}"
CIMAGE_CLUSTER_WEIGHT="${CIMAGE_CLUSTER_WEIGHT:-0.1}"
CIMAGE_NUM_FACTORS="${CIMAGE_NUM_FACTORS:-8}"
CIMAGE_NUM_CLUSTERS="${CIMAGE_NUM_CLUSTERS:-16}"
CIMAGE_CLUSTER_ALPHA="${CIMAGE_CLUSTER_ALPHA:-1.0}"
WAIT_FOR_SESSION="${WAIT_FOR_SESSION:-}"
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
  --extra_flag=--decoded_require_c0p_noncompact_endpoint
  --extra_flag=--decoded_add_degree_target
  --extra_flag=1
  --extra_flag=--decoded_add_degree_target_scope
  --extra_flag=intra_cluster
  --extra_flag=--decoded_add_degree_target_nodes
  --extra_flag=cp
  --extra_flag=--decoded_guarantee_degree_target
  --extra_flag=--ae_backbone
  --extra_flag=cimage_lite
  --extra_flag=--maskgae_mask_rate
  --extra_flag="${MASKGAE_MASK_RATE}"
  --extra_flag=--maskgae_feature_weight
  --extra_flag="${MASKGAE_FEATURE_WEIGHT}"
  --extra_flag=--cimage_factor_weight
  --extra_flag="${CIMAGE_FACTOR_WEIGHT}"
  --extra_flag=--cimage_cluster_weight
  --extra_flag="${CIMAGE_CLUSTER_WEIGHT}"
  --extra_flag=--cimage_num_factors
  --extra_flag="${CIMAGE_NUM_FACTORS}"
  --extra_flag=--cimage_num_clusters
  --extra_flag="${CIMAGE_NUM_CLUSTERS}"
  --extra_flag=--cimage_cluster_alpha
  --extra_flag="${CIMAGE_CLUSTER_ALPHA}"
)

FORCE_ARGS=()
if [[ "${FORCE}" == "1" ]]; then
  FORCE_ARGS=(--force)
fi

if [[ -n "${WAIT_FOR_SESSION}" ]]; then
  echo "[cimage-lite-backbone] waiting for tmux session '${WAIT_FOR_SESSION}' to finish"
  while tmux has-session -t "${WAIT_FOR_SESSION}" 2>/dev/null; do
    sleep 60
  done
fi

run_variant() {
  local name="$1"
  local add_ratio="$2"
  local cap="$3"
  local prefix="random_two_decoder_cimage_lite_backbone_${name}_${STAMP}"

  echo
  date
  echo "[cimage-lite-backbone] prefix=${prefix} add_ratio=${add_ratio} cap=${cap} mask_rate=${MASKGAE_MASK_RATE} raw_feature_w=${MASKGAE_FEATURE_WEIGHT} factor_w=${CIMAGE_FACTOR_WEIGHT} cluster_w=${CIMAGE_CLUSTER_WEIGHT}"
  "${PY}" "${RUN}" \
    --prefix "${prefix}" \
    --split-mode random \
    "${COMMON[@]}" \
    "${FORCE_ARGS[@]}" \
    --decoded-add-ratio "${add_ratio}" \
    --decoded-graph-aug-bound "${cap}"
}

echo "[cimage-lite-backbone] started $(date -Is)"
echo "[cimage-lite-backbone] seeds=${SEEDS[*]} datasets=${DATASETS[*]}"

# Match the current CP target-1 repair setting, changing only the AE backbone.
run_variant c0p_noncompact_soft025_cap010_cp_dtarget1_guarantee_addr020 0.20 0.10

echo "[cimage-lite-backbone] finished $(date -Is)"
