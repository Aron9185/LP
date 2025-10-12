#!/usr/bin/env bash
set -euo pipefail

# -------------------- knobs you might tweak --------------------
PRETRAIN_EPOCHS=100
AUG_RATIO=10          # global edit budget (× |E0|)
#AUG_BOUND=0.1         # per-node cap fraction (relative to deg0)
AUG_BOUND=0.5
VERSIONS=("aron_desc" "aron_asc")
# DATASETS=("citeseer" "Cora_ML" "LastFMAsia")
DATASETS=("LastFMAsia")

# per-dataset degree_threshold (incl self-loop), tune if needed
declare -A DEG_THR
DEG_THR["citeseer"]=0.9
DEG_THR["Cora_ML"]=0.5
DEG_THR["LastFMAsia"]=0.8

# ---------------------------------------------------------------

DATESTR=$(date +%m%d)
LOGDIR="logs/grid_aron_${DATESTR}"
mkdir -p "$LOGDIR"

run_one () {
  local ver="$1"
  local ds="$2"
  local seed="$3"
  local idx="$4"   # we’ll just mirror seed here; change if you want separate idx

  local degthr="${DEG_THR[$ds]}"
  # score/ckpt files per dataset (created if missing by your pretrain path)
  local scores="runs/${ds}_scores${idx}.pt"
  local ckpt="runs/${ds}_pre${idx}.pt"

  local logfile="${LOGDIR}/${ds}_${ver}_seed${seed}_idx${idx}.log"

  echo ">>> RUN ver=${ver} ds=${ds} seed=${seed} idx=${idx} thr=${degthr} | log=${logfile}"

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
    --logging \
    --date "${DATESTR}" \
    2>&1 | tee "${logfile}"
}

# sweep
for ver in "${VERSIONS[@]}"; do
  for ds in "${DATASETS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      run_one "${ver}" "${ds}" "${seed}" "${seed}"
    done
  done
done
