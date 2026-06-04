#!/usr/bin/env bash
set -euo pipefail

cd /home/retro/ARON

PY="${PY:-/home/retro/anaconda3/envs/pyg/bin/python}"
STAMP="${STAMP:-20260604}"
LOG="experiments/reversegnn-compactness/results/baseline_gap_queue_${STAMP}_tmux.log"

mkdir -p "$(dirname "${LOG}")"

{
  echo "[baseline-gap-queue] started $(date -Is)"
  echo "[baseline-gap-queue] GPU queue: HeaRT MaskGAE/CIMAGE dot, then ARON-random NCNC/BUDDY"

  "${PY}" experiments/reversegnn-compactness/run_no_editor_backbone.py \
    --prefix "heart_maskgae_cimage_dot_${STAMP}" \
    --split-mode heart \
    --datasets cora citeseer \
    --seeds 0 1 2 3 4 \
    --epochs 700 \
    --backbones maskgae cimage_full \
    --configs dot \
    --max-workers 1 \
    --eval-log-every 20 \
    --train-eval-every 20 \
    --heart-eval-every 20 \
    --heart-val-frac 1.0 \
    --heart-checkpoint-metric hit10 \
    --skip-train-acc \
    --decoder-diag-every 0

  "${PY}" experiments/reversegnn-compactness/run_random_ncnc_buddy_baselines.py \
    --prefix "random_ncnc_buddy_${STAMP}" \
    --datasets cora citeseer \
    --models ncnc buddy \
    --seeds 0 1 2 3 4

  echo "[baseline-gap-queue] finished $(date -Is)"
} 2>&1 | tee -a "${LOG}"
