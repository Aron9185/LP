#!/usr/bin/env bash
set -euo pipefail

cd /home/retro/ARON

PY=/home/retro/anaconda3/envs/pyg/bin/python
RUN=experiments/reversegnn-compactness/run_heart_pair_scorer_editor.py

STAMP="${STAMP:-20260512}"
SEEDS=(${SEEDS:-0 1 2 3 4})
DATASETS=(${DATASETS:-cora citeseer})
MAX_WORKERS="${MAX_WORKERS:-1}"
MLP_PAIR_MAX_ROWS="${MLP_PAIR_MAX_ROWS:-4}"
EPOCHS="${EPOCHS:-700}"

# Baseline for comparison already exists:
#   random_two_decoder_editor_20260501, config two_decoder_pred
# It uses decoder_type=pair_mlp_struct with the same prediction head and random split.
COMMON=(
  --datasets "${DATASETS[@]}"
  --seeds "${SEEDS[@]}"
  --epochs "${EPOCHS}"
  --max-workers "${MAX_WORKERS}"
  --mlp-pair-max-rows "${MLP_PAIR_MAX_ROWS}"
  --configs two_decoder_pred
  --decoded-add-ratio 0.01
  --decoded-remove-ratio 0.0
  --extra_flag=--split_mode
  --extra_flag=random
  --extra_flag=--skip_oom_epoch
)

run_decoder() {
  local decoder_type="$1"
  local prefix="random_two_decoder_decoder_ablate_${decoder_type}_${STAMP}"

  echo
  date
  echo "[decoder-ablation] prefix=${prefix} decoder_type=${decoder_type}"
  "${PY}" "${RUN}" \
    --prefix "${prefix}" \
    "${COMMON[@]}" \
    --extra_flag=--decoder_type \
    --extra_flag="${decoder_type}"
}

run_decoder mlp_pair
run_decoder bilinear
