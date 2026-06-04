#!/usr/bin/env bash
set -euo pipefail

cd /home/retro/ARON

echo "[fullgraph-leakage-backbones] started $(date -Is)"
echo "[fullgraph-leakage-backbones] phase=maskgae"
/home/retro/ARON/experiments/reversegnn-compactness/run_random_two_decoder_fullgraph_leakage_maskgae_20260529.sh

echo "[fullgraph-leakage-backbones] phase=cimage_full"
/home/retro/ARON/experiments/reversegnn-compactness/run_random_two_decoder_fullgraph_leakage_cimage_full_20260529.sh

echo "[fullgraph-leakage-backbones] finished $(date -Is)"
