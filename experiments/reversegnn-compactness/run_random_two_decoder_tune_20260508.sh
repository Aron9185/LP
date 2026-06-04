#!/usr/bin/env bash
set -euo pipefail

cd /home/retro/ARON

PY=/home/retro/anaconda3/envs/pyg/bin/python
RUN=experiments/reversegnn-compactness/run_heart_pair_scorer_editor.py

COMMON=(
  --datasets cora citeseer
  --seeds 0 1 2 3 4
  --epochs 700
  --max-workers 1
  --mlp-pair-max-rows 4
  --configs two_decoder_pred
  --extra_flag=--split_mode
  --extra_flag=random
  --extra_flag=--skip_oom_epoch
)

run_variant() {
  local prefix="$1"
  shift

  echo
  date
  echo "[variant] ${prefix} $*"
  "${PY}" "${RUN}" --prefix "${prefix}" "${COMMON[@]}" "$@"
}

run_variant random_two_decoder_tune_bce005_20260508 --prediction-bce-weight 0.05
run_variant random_two_decoder_tune_bce020_20260508 --prediction-bce-weight 0.2

run_variant random_two_decoder_tune_enc000_20260508 --prediction-encoder-weight 0.0
run_variant random_two_decoder_tune_enc002_20260508 --prediction-encoder-weight 0.02
run_variant random_two_decoder_tune_enc010_20260508 --prediction-encoder-weight 0.1

run_variant random_two_decoder_tune_add0005_20260508 --decoded-add-ratio 0.005
run_variant random_two_decoder_tune_add002_20260508 --decoded-add-ratio 0.02
