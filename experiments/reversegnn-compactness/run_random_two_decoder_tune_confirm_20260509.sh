#!/usr/bin/env bash
set -euo pipefail

cd /home/retro/ARON

PY=/home/retro/anaconda3/envs/pyg/bin/python
RUN=experiments/reversegnn-compactness/run_heart_pair_scorer_editor.py

COMMON=(
  --seeds 5 6 7 8 9
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
  echo "[confirm] ${prefix} $*"
  "${PY}" "${RUN}" --prefix "${prefix}" "${COMMON[@]}" "$@"
}

run_variant random_two_decoder_tune_bce005_20260508 \
  --datasets cora citeseer \
  --prediction-bce-weight 0.05

run_variant random_two_decoder_tune_enc010_20260508 \
  --datasets cora \
  --prediction-encoder-weight 0.1
