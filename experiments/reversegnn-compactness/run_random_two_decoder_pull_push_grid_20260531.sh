#!/usr/bin/env bash
set -euo pipefail

cd /home/retro/ARON

PY=/home/retro/anaconda3/envs/pyg/bin/python
RUN=experiments/reversegnn-compactness/run_heart_pair_scorer_editor.py

STAMP="${STAMP:-20260531}"
SEEDS=(${SEEDS:-0 1 2})
DATASETS=(${DATASETS:-cora citeseer})
MAX_WORKERS="${MAX_WORKERS:-1}"
MLP_PAIR_MAX_ROWS="${MLP_PAIR_MAX_ROWS:-16}"
EPOCHS="${EPOCHS:-700}"
FEAT_MASK_RATIO="${FEAT_MASK_RATIO:-0.1}"
WAIT_FOR_SESSION="${WAIT_FOR_SESSION:-}"

COMMON=(
  --datasets "${DATASETS[@]}"
  --seeds "${SEEDS[@]}"
  --epochs "${EPOCHS}"
  --split-mode random
  --max-workers "${MAX_WORKERS}"
  --mlp-pair-max-rows "${MLP_PAIR_MAX_ROWS}"
  --configs two_decoder_pred
  --decoded-remove-ratio 0.0
  --editor-pull-strength 0.25
  --compactness-mask-scope cp
  --rewrite-endpoint-scope c0p
  --decoded-add-ratio 0.20
  --decoded-graph-aug-bound 0.10
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

if [[ -n "${WAIT_FOR_SESSION}" ]]; then
  echo "[pull-push-grid] waiting for tmux session '${WAIT_FOR_SESSION}' to finish"
  while tmux has-session -t "${WAIT_FOR_SESSION}" 2>/dev/null; do
    sleep 60
  done
fi

run_variant() {
  local name="$1"
  local pull_scope="$2"
  local push_scope="$3"
  local noncompact_push="$4"
  local noise_push="$5"
  local prefix="random_two_decoder_pull_push_${name}_${STAMP}"

  echo
  date
  echo "[pull-push-grid] prefix=${prefix} pull_scope=${pull_scope} push_scope=${push_scope} noncompact_push=${noncompact_push} noise_push=${noise_push}"
  "${PY}" "${RUN}" \
    --prefix "${prefix}" \
    --pull-mask-scope "${pull_scope}" \
    "${COMMON[@]}" \
    --extra_flag=--editor_push_scope \
    --extra_flag="${push_scope}" \
    --extra_flag=--editor_noncompact_push_strength \
    --extra_flag="${noncompact_push}" \
    --extra_flag=--editor_noise_push_strength \
    --extra_flag="${noise_push}" \
    --extra_flag=--editor_push_preserve_norm
}

echo "[pull-push-grid] started $(date -Is)"
echo "[pull-push-grid] seeds=${SEEDS[*]} datasets=${DATASETS[*]} epochs=${EPOCHS}"

run_variant baseline_cp_pull cp none 0.00 0.00
run_variant c0p_pull_only c0p none 0.00 0.00
run_variant c0p_pull_push_weak c0p noncompact_cp_and_noise 0.05 0.02
run_variant c0p_pull_push_mid c0p noncompact_cp_and_noise 0.10 0.05

echo "[pull-push-grid] finished $(date -Is)"
