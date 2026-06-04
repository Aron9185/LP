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
# It uses pair_mlp_struct + pair_residual_struct, random split, add_ratio=0.01,
# compactness_weight=0.2, dynamic C0p targets, and no removal.
COMMON=(
  --datasets "${DATASETS[@]}"
  --seeds "${SEEDS[@]}"
  --epochs "${EPOCHS}"
  --max-workers "${MAX_WORKERS}"
  --mlp-pair-max-rows "${MLP_PAIR_MAX_ROWS}"
  --configs two_decoder_pred
  --decoded-remove-ratio 0.0
  --extra_flag=--split_mode
  --extra_flag=random
  --extra_flag=--skip_oom_epoch
)

run_variant() {
  local name="$1"
  shift
  local prefix="random_two_decoder_aug_compact_${name}_${STAMP}"

  echo
  date
  echo "[aug-compact-ablation] prefix=${prefix} $*"
  "${PY}" "${RUN}" \
    --prefix "${prefix}" \
    "${COMMON[@]}" \
    "$@"
}

run_variant add000 \
  --decoded-add-ratio 0.0

run_variant compact000 \
  --decoded-add-ratio 0.01 \
  --extra_flag=--compactness_weight \
  --extra_flag=0.0

run_variant compact100_frozen \
  --decoded-add-ratio 0.01 \
  --extra_flag=--compactness_weight \
  --extra_flag=1.0 \
  --extra_flag=--freeze_c0p_at_edit_start
