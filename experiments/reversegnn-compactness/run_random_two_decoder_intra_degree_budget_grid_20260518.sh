#!/usr/bin/env bash
set -euo pipefail

cd /home/retro/ARON

PY=/home/retro/anaconda3/envs/pyg/bin/python
RUN=experiments/reversegnn-compactness/run_heart_pair_scorer_editor.py

STAMP="${STAMP:-20260518}"
SEEDS=(${SEEDS:-0 1 2 3 4})
DATASETS=(${DATASETS:-cora citeseer})
MAX_WORKERS="${MAX_WORKERS:-1}"
MLP_PAIR_MAX_ROWS="${MLP_PAIR_MAX_ROWS:-4}"
EPOCHS="${EPOCHS:-700}"
FEAT_MASK_RATIO="${FEAT_MASK_RATIO:-0.1}"
WAIT_FOR_SESSION="${WAIT_FOR_SESSION:-}"

COMMON=(
  --datasets "${DATASETS[@]}"
  --seeds "${SEEDS[@]}"
  --epochs "${EPOCHS}"
  --max-workers "${MAX_WORKERS}"
  --mlp-pair-max-rows "${MLP_PAIR_MAX_ROWS}"
  --configs two_decoder_pred
  --decoded-remove-ratio 0.0
  --editor-pull-strength 0.25
  --decoded-graph-aug-bound 0.10
  --extra_flag=--split_mode
  --extra_flag=random
  --extra_flag=--skip_oom_epoch
  --extra_flag=--feat_mask_ratio
  --extra_flag="${FEAT_MASK_RATIO}"
  --extra_flag=--decoded_require_c0p_noncompact_endpoint
  --extra_flag=--decoded_add_degree_target_scope
  --extra_flag=intra_cluster
)

if [[ -n "${WAIT_FOR_SESSION}" ]]; then
  echo "[intra-degree-budget-grid] waiting for tmux session '${WAIT_FOR_SESSION}' to finish"
  while tmux has-session -t "${WAIT_FOR_SESSION}" 2>/dev/null; do
    sleep 60
  done
fi

run_variant() {
  local name="$1"
  local target="$2"
  local add_ratio="$3"
  local prefix="random_two_decoder_intra_degree_budget_${name}_${STAMP}"

  echo
  date
  echo "[intra-degree-budget-grid] prefix=${prefix} target=${target} add_ratio=${add_ratio} feat_mask_ratio=${FEAT_MASK_RATIO}"
  "${PY}" "${RUN}" \
    --prefix "${prefix}" \
    "${COMMON[@]}" \
    --decoded-add-ratio "${add_ratio}" \
    --extra_flag=--decoded_add_degree_target \
    --extra_flag="${target}"
}

echo "[intra-degree-budget-grid] started $(date -Is)"
echo "[intra-degree-budget-grid] seeds=${SEEDS[*]} datasets=${DATASETS[*]}"

run_variant c0p_noncompact_soft025_cap010_intra_dtarget2_addr001 2 0.01
run_variant c0p_noncompact_soft025_cap010_intra_dtarget2_addr003 2 0.03
run_variant c0p_noncompact_soft025_cap010_intra_dtarget2_addr005 2 0.05

echo "[intra-degree-budget-grid] finished $(date -Is)"
