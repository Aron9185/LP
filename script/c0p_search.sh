#!/usr/bin/env bash
set -euo pipefail

# -------------------- knobs you might tweak --------------------
PRETRAIN_EPOCHS=100
AUG_RATIO=10              # global edit budget (× |E0|)
AUG_BOUND=0.5             # per-node cap fraction (relative to deg0)

# Cluster methods and constrained versions
CLUSTER_METHODS=("gmm")
VERSIONS=("aron_desc_inter" "aron_desc_intra")

# Datasets
DATASETS=("cora" "Cora_ML" "citeseer" "LastFMAsia")

# seeds
SEEDS=({0..9})

# per-dataset degree_threshold (incl self-loop)
declare -A DEG_THR
DEG_THR["cora"]=0.9
DEG_THR["citeseer"]=0.9
DEG_THR["Cora_ML"]=0.5
DEG_THR["LastFMAsia"]=0.8

# (Optional) clustering params
GMM_K=32
GMM_TAU=0.60
LOUVAIN_RESOLUTION=1.0

# (Optional) HDBSCAN params
DBSCAN_EPS=""
DBSCAN_MIN_SAMPLES=5
DBSCAN_METRIC="cosine"

# New augmentation knobs
TOPK_PER_NODE=64
AUG_RATIO_EPOCH=""            # e.g., "0.02" to throttle; empty disables

# ---- restricted (α,γ, d̂) augmentation ----
RESTRICTED=1                  # 1=on, 0=off

# Sweep (alpha,gamma) settings — all within (0,1]
# Format: "alpha,gamma"
#RSTR_GRID=("0.60,1.00" "0.70,1.00" "0.80,1.00")
RSTR_GRID=("0.80,1.00")

# NEW: C0p prune fraction (use the new setting)
C0P_PRUNE_FRAC=0.01

# Optional suffix like "loss_origin"
LOSS_TAG="${LOSS_TAG:-}"
# ---------------------------------------------------------------

fmt2() { printf '%.2f' "$1"; }   # 2-decimal string

DATESTR=$(date +%m%d)
LOGDIR="logs/${DATESTR}"
mkdir -p "$LOGDIR" runs

run_one () {
  local ver="$1" cm="$2" ds="$3" seed="$4" idx="$5" r_alpha="$6" r_gamma="$7"

  local degthr="${DEG_THR[$ds]}"

  # derive cluster_mode from version
  local mode="any"
  [[ "$ver" == *"_intra" ]] && mode="intra"
  [[ "$ver" == *"_inter" ]] && mode="inter"

  # score/ckpt files per dataset
  local scores="runs/${ds}_scores${idx}.pt"
  local ckpt="runs/${ds}_pre${idx}.pt"

  # pretty stamps
  local r_str;     r_str=$(fmt2 "${AUG_RATIO}")
  local fmr_str;   fmr_str=$(fmt2 "${AUG_BOUND}")
  local d_str;     d_str=$(fmt2 "${degthr}")
  local a_str;     a_str=$(fmt2 "${r_alpha}")
  local g_str;     g_str=$(fmt2 "${r_gamma}")
  local c0p_str;   c0p_str=$(fmt2 "${C0P_PRUNE_FRAC}")

  # unique run suffix
  local RUNID="$(date +%H%M%S)_$$_${RANDOM}"

  # logfile name — include alpha/gamma and c0p up front for parsers
  local logfile="${LOGDIR}/${ds}_${ver}_a${a_str}_g${g_str}_c0p${c0p_str}_seed${seed}_idx${idx}_r${r_str}_fmr${fmr_str}_d${d_str}_${cm}"
  [[ -n "${LOSS_TAG}" ]] && logfile="${logfile}_${LOSS_TAG}"
  logfile="${logfile}_ts${RUNID}.log"

  # META header into the log
  {
    echo "===== RUN META ====="
    echo "date=${DATESTR} time=$(date +%F' '%T) runid=${RUNID}"
    echo "dataset=${ds} ver=${ver} mode=${mode} method=${cm}"
    echo "seed=${seed} idx=${idx}"
    echo "alpha=${r_alpha} gamma=${r_gamma} restricted=${RESTRICTED}"
    echo "aug_ratio=${AUG_RATIO} aug_bound=${AUG_BOUND} degree_thr=${degthr}"
    echo "topk_per_node=${TOPK_PER_NODE} aug_ratio_epoch=${AUG_RATIO_EPOCH:-<none>}"
    echo "gmm_k=${GMM_K} gmm_tau=${GMM_TAU}"
    echo "c0p_prune_frac=${C0P_PRUNE_FRAC}"
    echo "===================="
  } | tee -a "${logfile}"

  echo ">>> RUN ver=${ver} cm=${cm} mode=${mode} ds=${ds} seed=${seed} idx=${idx} thr=${degthr} a=${r_alpha} g=${r_gamma} c0p=${C0P_PRUNE_FRAC} | log=${logfile}"

  # Build arg list
  args=(
    python src/aron_main.py
    --ver "${ver}"
    --dataset "${ds}"
    --seed "${seed}"
    --idx  "${idx}"
    --pretrain_epochs "${PRETRAIN_EPOCHS}"
    --frozen_scores "${scores}"
    --pretrained_ckpt "${ckpt}"
    --aug_ratio "${AUG_RATIO}"
    --aug_bound "${AUG_BOUND}"
    --degree_threshold "${degthr}"
    --cluster_method "${cm}"
    --cluster_mode "${mode}"
    --gmm_k "${GMM_K}"
    --gmm_tau "${GMM_TAU}"
    --topk_per_node "${TOPK_PER_NODE}"
    --c0p_prune_frac "${C0P_PRUNE_FRAC}"
    --run_tag "a${a_str}_g${g_str}_c0p${c0p_str}"   # also embed in python logs
    --logging
    --date "${DATESTR}"
  )

  [[ -n "${AUG_RATIO_EPOCH}" ]] && args+=( --aug_ratio_epoch "${AUG_RATIO_EPOCH}" )

  if [[ "${RESTRICTED}" -eq 1 ]]; then
    args+=( --restricted
            --restrict_alpha "${r_alpha}"
            --restrict_gamma "${r_gamma}" )
  fi

  if [[ "${cm}" == "hdbscan" ]]; then
    args+=( --dbscan_min_samples "${DBSCAN_MIN_SAMPLES}" --dbscan_metric "${DBSCAN_METRIC}" )
    [[ -n "${DBSCAN_EPS}" ]] && args+=( --dbscan_eps "${DBSCAN_EPS}" )
  fi

  echo "[CMD] ${args[*]}" | tee -a "${logfile}"
  "${args[@]}" 2>&1 | tee -a "${logfile}"
}

# sweep
for ver in "${VERSIONS[@]}"; do
  for cm in "${CLUSTER_METHODS[@]}"; do
    for ds in "${DATASETS[@]}"; do
      for seed in "${SEEDS[@]}"; do
        if [[ "${RESTRICTED}" -eq 1 ]]; then
          for pair in "${RSTR_GRID[@]}"; do
            IFS=',' read -r a g <<< "${pair}"
            run_one "${ver}" "${cm}" "${ds}" "${seed}" "${seed}" "${a}" "${g}"
          done
        else
          # Baseline (not passing restrict flags)
          run_one "${ver}" "${cm}" "${ds}" "${seed}" "${seed}" "0.80" "1.00"
        fi
      done
    done
  done
done
