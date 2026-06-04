#!/usr/bin/env bash
set -euo pipefail

cd /home/retro/ARON

PY=/home/retro/anaconda3/envs/pyg/bin/python
RUN=experiments/reversegnn-compactness/run_heart_pair_scorer_editor.py

STAMP="${STAMP:-20260510}"
SEEDS=(${SEEDS:-0 1 2 3 4})
DATASETS=(${DATASETS:-cora citeseer})
MAX_WORKERS="${MAX_WORKERS:-1}"
MLP_PAIR_MAX_ROWS="${MLP_PAIR_MAX_ROWS:-4}"
EPOCHS="${EPOCHS:-700}"

COMMON=(
  --datasets "${DATASETS[@]}"
  --seeds "${SEEDS[@]}"
  --epochs "${EPOCHS}"
  --max-workers "${MAX_WORKERS}"
  --mlp-pair-max-rows "${MLP_PAIR_MAX_ROWS}"
  --configs two_decoder_pred_remove
  --decoded-add-ratio 0.01
  --decoded-remove-ratio 0.05
  --extra_flag=--split_mode
  --extra_flag=random
  --extra_flag=--skip_oom_epoch
  --extra_flag=--decoded_degree_floor
  --extra_flag=0
  --extra_flag=--decoded_allow_cross_cluster
  --extra_flag=--decoded_no_c0p_endpoint
)

run_cap() {
  local cap="$1"
  local prefix="random_two_decoder_capped_remove_mr${cap}_${STAMP}"

  echo
  date
  echo "[capped-remove] prefix=${prefix} max_remove_per_round=${cap}"
  "${PY}" "${RUN}" \
    --prefix "${prefix}" \
    "${COMMON[@]}" \
    --extra_flag=--decoded_max_remove_per_round \
    --extra_flag="${cap}"
}

run_cap 1
run_cap 3
run_cap 5
run_cap 10
