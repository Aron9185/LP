#!/usr/bin/env bash
set -euo pipefail

# -------------------- knobs you might tweak --------------------
PRETRAIN_EPOCHS=100
AUG_RATIO=10            # global edit budget (× |E0|)
AUG_BOUND=0.5           # per-node cap fraction (relative to deg0)

# Cluster methods and constrained versions
CLUSTER_METHODS=("louvain" "gmm")
VERSIONS=("aron_desc_inter" "aron_desc_intra")

# Datasets
DATASETS=("cora" "Cora_ML" "citeseer" "LastFMAsia")

# 30 runs: seeds 0..29
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

# (Optional) suffix like "loss_origin" if you want it in filenames
LOSS_TAG="${LOSS_TAG:-}"   # set via env: LOSS_TAG=loss_origin ./run.sh
# ---------------------------------------------------------------

DATESTR=$(date +%m%d)
LOGDIR="logs/${DATESTR}"
mkdir -p "$LOGDIR"
mkdir -p runs

run_one () {
  local ver="$1"          # aron_desc_intra | aron_desc_inter
  local cm="$2"           # gmm | louvain
  local ds="$3"
  local seed="$4"
  local idx="$5"          # mirror seed unless you want separate indexing

  local degthr="${DEG_THR[$ds]}"

  # derive cluster_mode from version
  local mode="any"
  if [[ "$ver" == *"_intra" ]]; then
    mode="intra"
  elif [[ "$ver" == *"_inter" ]]; then
    mode="inter"
  fi

  # score/ckpt files per dataset (reuse across methods/versions if you like)
  local scores="runs/${ds}_scores${idx}.pt"
  local ckpt="runs/${ds}_pre${idx}.pt"

  # pretty stamps for floats
  local r_str;   r_str=$(printf '%.1f' "${AUG_RATIO}")
  local fmr_str; fmr_str=$(printf '%.1f' "${AUG_BOUND}")
  local d_str;   d_str=$(printf '%.1f' "${degthr}")

  # logfile name (dataset_version_rX_fmrY_dZ_method_seed_idx[_{LOSS_TAG}].log)
  local logfile="${LOGDIR}/${ds}_${ver}_r${r_str}_fmr${fmr_str}_d${d_str}_${cm}_seed${seed}_idx${idx}.log"
  if [[ -n "${LOSS_TAG}" ]]; then
    logfile="${logfile%.log}_${LOSS_TAG}.log"
  fi

  echo ">>> RUN ver=${ver} cm=${cm} mode=${mode} ds=${ds} seed=${seed} idx=${idx} thr=${degthr} | log=${logfile}"

  python src/aron_main.py \
    --ver "${ver}" \
    --dataset "${ds}" \
    --seed "${seed}" \
    --idx  "${idx}" \
    --pretrain_epochs "${PRETRAIN_EPOCHS}" \
    --frozen_scores "${scores}" \
    --pretrained_ckpt "${ckpt}" \
    --aug_ratio "${AUG_RATIO}" \
    --aug_bound "${AUG_BOUND}" \
    --degree_threshold "${degthr}" \
    --cluster_method "${cm}" \
    --cluster_mode "${mode}" \
    --gmm_k "${GMM_K}" \
    --gmm_tau "${GMM_TAU}" \
    --logging \
    --date "${DATESTR}" \
    2>&1 | tee "${logfile}"
}

# sweep
for ver in "${VERSIONS[@]}"; do
  for cm in "${CLUSTER_METHODS[@]}"; do
    for ds in "${DATASETS[@]}"; do
      for seed in "${SEEDS[@]}"; do
        run_one "${ver}" "${cm}" "${ds}" "${seed}" "${seed}"
      done
    done
  done
done
