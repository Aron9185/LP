#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Static C0p/CP prune sweep for aron_desc contrastive encoder
# - prune_frac: 0.00, 0.05, ..., 0.95
# - datasets: cora, Cora_ML, citeseer, LastFMAsia
# - scopes: c0p_only, cp_minus_c0p, cp_all
# - encoder ver: aron_desc (degree_aug_v6 + contrastive)
# - clusters: GMM on pretrained aron_desc encoder, on ORIGINAL train graph
# ============================================================

DATE=$(date +%m%d)

# --- Training knobs ---
EPOCHS=700
VER="aron_desc"

CLUSTER_METHOD="gmm"
CLUSTER_MODE="intra"        # labels from GMM on Z; "intra" is safe default

# α–γ for c0p gating (used when defining cores)
ALPHA=0.80
GAMMA=0.80

# Contrastive / aug knobs kept same as your previous pipeline
AUG_RATIO=10
AUG_BOUND=0.5
DEG_THR=0.9
TOPK_PER_NODE=64
GMM_K=32
GMM_TAU=0.60

# Datasets / scopes / seeds
#DATASETS=("cora" "Cora_ML" "citeseer")
DATASETS=("LastFMAsia")
# Internal scope names used by your code: cp_all, c0p_only, cp_minus_c0p
SCOPES=("c0p_only" "cp_minus_c0p" "cp_all")

# Prune fractions: 0%..95% step 5% (as decimals)
PRUNE_FRACS=("0.00" "0.05" "0.10" "0.15" "0.20" "0.25" "0.30" "0.35" "0.40" "0.45" \
             "0.50" "0.55" "0.60" "0.65" "0.70" "0.75" "0.80" "0.85" "0.90" "0.95")

SEEDS=(0 1 2 3 4)   # extend to (0 1 2 3 4) if you want 5 runs

for ds in "${DATASETS[@]}"; do
  for scope in "${SCOPES[@]}"; do
    for frac in "${PRUNE_FRACS[@]}"; do
      for seed in "${SEEDS[@]}"; do

        # Purely cosmetic index: encode frac*100 as integer (00, 05, 10, ...)
        frac_int=$(printf "%02d" "$(echo "${frac} * 100" | bc -l | xargs printf "%.0f")")
        idx="${frac_int}"

        echo "===== RUN META ====="
        echo "date=${DATE} time=$(date '+%Y-%m-%d %H:%M:%S')"
        echo "dataset=${ds} ver=${VER} mode=${CLUSTER_MODE} method=${CLUSTER_METHOD}"
        echo "scope=${scope} pre_prune_frac=${frac}"
        echo "seed=${seed} idx=${idx}"
        echo "alpha=${ALPHA} gamma=${GAMMA} (restricted=1 for c0p gating)"
        echo "aug_ratio=${AUG_RATIO} aug_bound=${AUG_BOUND} degree_thr=${DEG_THR}"
        echo "topk_per_node=${TOPK_PER_NODE} gmm_k=${GMM_K} gmm_tau=${GMM_TAU}"
        echo "===================="

        log_dir="logs/static_prune_aron_desc/${ds}/${scope}"
        mkdir -p "${log_dir}"
        log_file="${log_dir}/${ds}_aron_desc_${scope}_frac${frac_int}_seed${seed}.log"

        # Run: static pre-prune + aron_desc contrastive training
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
          --pre_prune_frac "${frac}" \
          --pre_prune_scope "${scope}" \
          2>&1 | tee "${log_file}"

      done
    done
  done
done
