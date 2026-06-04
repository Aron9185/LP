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
)

FORCE_ARGS=()
if [[ "${FORCE}" == "1" ]]; then
  FORCE_ARGS=(--force)
fi

run_variant() {
  local name="$1"
  local add_ratio="$2"
  local cap="$3"
  local prefix="random_two_decoder_fullgraph_leakage_current_${name}_${STAMP}"

  echo
  date
  echo "[fullgraph-leakage-current] prefix=${prefix} lp_train_graph=full add_ratio=${add_ratio} cap=${cap} feat_mask_ratio=${FEAT_MASK_RATIO}"
  "${PY}" "${RUN}" \
    --prefix "${prefix}" \
    --split-mode random \
    "${COMMON[@]}" \
    "${FORCE_ARGS[@]}" \
    --decoded-add-ratio "${add_ratio}" \
    --decoded-graph-aug-bound "${cap}"
}

echo "[fullgraph-leakage-current] started $(date -Is)"
echo "[fullgraph-leakage-current] seeds=${SEEDS[*]} datasets=${DATASETS[*]}"

# Match the current CP target-1 repair setting, changing only the LP training
# graph from no-leak train split to the full observed graph.
run_variant c0p_noncompact_soft025_cap010_cp_dtarget1_guarantee_addr020 0.20 0.10

echo "[fullgraph-leakage-current] finished $(date -Is)"
