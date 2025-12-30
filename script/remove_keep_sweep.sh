#!/usr/bin/env bash
set -euo pipefail

# ===== EDIT THESE =====
PY=python
MAIN=src/aron_main.py

# Datasets to run (edit "other two" here if needed)
#DATASETS=("cora" "Cora_ML" "citeseer" "LastFMAsia")
DATASETS=("Cora_ML" "LastFMAsia")

# Scope tokens MUST match your remove_state init parsing
SCOPES=(cp_all c0p_only cp_minus_c0p)

# Keep % sweep
#KEEPS=(100 97 94 91 88 85 82 79 76 73 70 50 25 5)
KEEPS=(100 97 94 91 88 85 82 79 76 73 70 50 25 5)

# Kind: intra/inter/both (you can set just "both" if you want)
KINDS=(intra)

# Seeds (you can do seed0 only first, then extend)
SEEDS=({5..30})
#SEEDS=(0)

# Global run options you typically use (edit as needed)
COMMON_ARGS=(
  --cluster_method gmm
  --gmm_k 32
  --gmm_tau 0.60
  --restricted
  --restrict_alpha 0.80
  --restrict_gamma 1.00
  --aug_ratio 1.00
  --aug_bound 0.50
  --topk_per_node 64
  --degree_thr 0.90
)

# Where logs go
LOG_ROOT="logs/remove_keep"

mkdir -p "$LOG_ROOT"

echo "===== REMOVE-KEEP SWEEP ====="
echo "Datasets: ${DATASETS[*]}"
echo "Scopes:   ${SCOPES[*]}"
echo "Keeps:    ${KEEPS[*]}"
echo "Kinds:    ${KINDS[*]}"
echo "Seeds:    ${SEEDS[*]}"
echo "Log root: $LOG_ROOT"
echo "============================="

for dataset in "${DATASETS[@]}"; do
  for scope in "${SCOPES[@]}"; do
    for keep in "${KEEPS[@]}"; do
      for kind in "${KINDS[@]}"; do
        for seed in "${SEEDS[@]}"; do

          # ver naming: remove_only_<kind>_<scope>_keepXX
          ver="remove_only_${kind}_${scope}_keep${keep}"

          outdir="${LOG_ROOT}/${dataset}/${scope}/keep${keep}"
          mkdir -p "$outdir"
          logfile="${outdir}/${dataset}_${ver}_seed${seed}.log"

          echo
          echo "[RUN] dataset=${dataset} scope=${scope} keep=${keep} kind=${kind} seed=${seed}"
          echo "[LOG] ${logfile}"

          # You may need your usual flags here (gpu id, etc.)
          # If you have a --seed flag, keep it. If not, remove it.
          $PY "$MAIN" \
            --dataset "$dataset" \
            --ver "$ver" \
            --seed "$seed" \
            "${COMMON_ARGS[@]}" \
            2>&1 | tee "$logfile"

        done
      done
    done
  done
done

echo
echo "DONE. Logs at: $LOG_ROOT"
