#!/usr/bin/env bash
set -e

# Stamp for this batch
DATE=$(date +%m%d)

EPOCHS=700
VER="remove_only_intra_cp"      # train-time setting (offline sweep uses raw train graph)
CLUSTER_METHOD="gmm"
CLUSTER_MODE="intra"

ALPHA=0.80
GAMMA=1.00

AUG_RATIO=10
AUG_BOUND=0.5
DEG_THR=0.9
TOPK_PER_NODE=64
GMM_K=32
GMM_TAU=0.60
C0P_PRUNE_FRAC=0.01

DATASETS=("cora" "Cora_ML" "citeseer" "LastFMAsia")
SCOPES=("cp_all" "c0p_only" "cp_minus_c0p")
SEEDS=(0)               # 5 runs per dataset per scope

for ds in "${DATASETS[@]}"; do
  for scope in "${SCOPES[@]}"; do
    for seed in "${SEEDS[@]}"; do
      idx=${seed}

      echo "===== RUN META ====="
      echo "date=${DATE} time=$(date '+%Y-%m-%d %H:%M:%S')"
      echo "dataset=${ds} ver=${VER} mode=${CLUSTER_MODE} method=${CLUSTER_METHOD}"
      echo "sweep_scope=${scope}"
      echo "seed=${seed} idx=${idx}"
      echo "alpha=${ALPHA} gamma=${GAMMA} restricted=1"
      echo "aug_ratio=${AUG_RATIO} aug_bound=${AUG_BOUND} degree_thr=${DEG_THR}"
      echo "topk_per_node=${TOPK_PER_NODE} gmm_k=${GMM_K} gmm_tau=${GMM_TAU}"
      echo "c0p_prune_frac=${C0P_PRUNE_FRAC}"
      echo "===================="

      log_dir="logs/cp_sweep_runs/${ds}/${scope}"
      mkdir -p "${log_dir}"
      log_file="${log_dir}/${ds}_${VER}_${scope}_seed${seed}_idx${idx}.log"

      CUDA_VISIBLE_DEVICES=0 python ./src/aron_main.py \
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
        --c0p_prune_frac ${C0P_PRUNE_FRAC} \
        --restricted \
        --restrict_alpha ${ALPHA} \
        --restrict_gamma ${GAMMA} \
        --sweep_scope "${scope}" \
        2>&1 | tee "${log_file}"

    done
  done
done
