#!/usr/bin/env bash
set -euo pipefail

cd /home/retro/ARON

PY="${PY:-/home/retro/anaconda3/envs/pyg/bin/python}"
RUN="${RUN:-experiments/reversegnn-compactness/run_heart_pair_scorer_editor.py}"

SEEDS=(${SEEDS:-1 2 4})
MAX_WORKERS="${MAX_WORKERS:-1}"
MLP_PAIR_MAX_ROWS="${MLP_PAIR_MAX_ROWS:-16}"
EPOCHS="${EPOCHS:-700}"

COMMON=(
  --split-mode random
  --random-checkpoint-metric hit10
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
  --decoder-diag-every 100
  --edit-metric-every 100
  --skip-train-acc
  --extra_flag=--skip_oom_epoch
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
  --extra_flag=0.1
)

run_confirm() {
  local dataset="$1"
  local prefix="$2"
  shift 2

  echo
  date
  echo "[refined-vgnae-confirm] dataset=${dataset} prefix=${prefix} seeds=${SEEDS[*]} extra=$*"
  "${PY}" "${RUN}" \
    --prefix "${prefix}" \
    --datasets "${dataset}" \
    "${COMMON[@]}" \
    "$@"
}

echo "[refined-vgnae-confirm] started $(date -Is)"
echo "[refined-vgnae-confirm] seeds=${SEEDS[*]} epochs=${EPOCHS} workers=${MAX_WORKERS}"

run_confirm \
  cora \
  random_refined_vgnae_cora_aq002_h512_z128_do04_lr001_b1_fm01_20260608 \
  --extra_flag=--decoded_add_quantile \
  --extra_flag=0.02

run_confirm \
  citeseer \
  random_refined_vgnae_citeseer_current_h512_z128_do04_lr001_b1_fm01_20260608

echo "[refined-vgnae-confirm] finished $(date -Is)"
