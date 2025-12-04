#!/usr/bin/env bash
set -euo pipefail

# Stamp for this batch
DATE=$(date +%m%d)

# --- Training knobs ---
EPOCHS=1000
CLUSTER_METHOD="gmm"
CLUSTER_MODE="intra"        # safe default; labels still come from GMM

# α–γ for c0p gating (used by prune_c0p_only)
ALPHA=0.80
GAMMA=0.8

# (kept as-is so your pipeline stays consistent)
AUG_RATIO=10
AUG_BOUND=0.5
DEG_THR=0.9
TOPK_PER_NODE=64
GMM_K=32
GMM_TAU=0.60

# Datasets / scopes / seeds
DATASETS=("cora" "Cora_ML" "citeseer" "LastFMAsia")
SCOPES=("c0p_only" "cp_all" "cp_minus_c0p")   # maps to ver=prune_<scope>
SEEDS=(0)                              # 5 runs

for ds in "${DATASETS[@]}"; do
  for scope in "${SCOPES[@]}"; do
    VER="prune_${scope}"                       # <- online pruning family name
    for seed in "${SEEDS[@]}"; do
      idx=${seed}

      echo "===== RUN META ====="
      echo "date=${DATE} time=$(date '+%Y-%m-%d %H:%M:%S')"
      echo "dataset=${ds} ver=${VER} mode=${CLUSTER_MODE} method=${CLUSTER_METHOD}"
      echo "scope=${scope}"
      echo "seed=${seed} idx=${idx}"
      echo "alpha=${ALPHA} gamma=${GAMMA} (restricted=1 for c0p gating)"
      echo "aug_ratio=${AUG_RATIO} aug_bound=${AUG_BOUND} degree_thr=${DEG_THR}"
      echo "topk_per_node=${TOPK_PER_NODE} gmm_k=${GMM_K} gmm_tau=${GMM_TAU}"
      echo "===================="

      log_dir="logs/prune_runs/${ds}/${scope}"
      mkdir -p "${log_dir}"
      log_file="${log_dir}/${ds}_${VER}_seed${seed}_idx${idx}.log"

      # Suppress tqdm progress bars in logs
      TQDM_DISABLE=1 PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0 \
      python -u ./src/aron_main.py \
        --dataset "${ds}" \
        --epochs ${EPOCHS} \
        --ver "${VER}" \
        --idx "${idx}" \
        --seed ${seed} \
        --cluster_method "${CLUSTER_METHOD}" \
        --cluster_mode "${CLUSTER_MODE}" \
        --gmm_k ${GMM_K} \
        --gmm_tau ${GMM_TAU} \
        --aug_ratio ${AUG_RATIO} \
        --aug_bound ${AUG_BOUND} \
        --degree_threshold ${DEG_THR} \
        --topk_per_node ${TOPK_PER_NODE} \
        --restricted \
        --restrict_alpha ${ALPHA} \
        --restrict_gamma ${GAMMA} \
        2>&1 | tee "${log_file}"

    done
  done
done
