#!/usr/bin/env bash
set -euo pipefail

############################################
# Select which version(s) to run
# Options: aron | v6 | both
RUN_VERSION="${RUN_VERSION:-both}"

# How many repeats
REPEATS="${REPEATS:-5}"

# Common settings
DATASET="${DATASET:-cora}"
DATE_TAG="${DATE_TAG:-1002}"

# Log directory (as requested)
OUTDIR="log/${DATE_TAG}"
mkdir -p "$OUTDIR"
echo "Logs will be written to: $OUTDIR"
############################################

run_aron() {
  echo ">>> Running ARON config ($REPEATS runs)"
  for run in $(seq 1 "$REPEATS"); do
    echo "---- ARON run ${run}/${REPEATS} ----"
    LOG="${OUTDIR}/aron_run_${DATASET}${run}.log"
    python src/aron_main.py \
      --ver aron \
      --aug_ratio 0.1 \
      --aug_bound 0.2 \
      --degree_threshold 0.5 \
      --dataset-str "$DATASET" \
      --date "$DATE_TAG" \
      --logging \
      2>&1 | tee "$LOG"
  done
}

run_v6() {
  echo ">>> Running V6 config ($REPEATS runs)"
  for run in $(seq 1 "$REPEATS"); do
    echo "---- V6 run ${run}/${REPEATS} (seed=${run}, idx=${run}) ----"
    LOG="${OUTDIR}/v6_run_${DATASET}${run}.log"
    python src/aron_main.py \
      --logging \
      --dataset-str "$DATASET" \
      --seed "$run" \
      --epochs 1000 \
      --hidden1 256 \
      --hidden2 64 \
      --lr 0.001 \
      --dropout 0.3 \
      --weight_decay 5e-4 \
      --aug_graph_weight 3.0 \
      --aug_ratio 0.1 \
      --aug_bound 0.1 \
      --alpha 1.0 \
      --beta 1.0 \
      --gamma 1.0 \
      --delta 1.0 \
      --temperature 1.0 \
      --date "$DATE_TAG" \
      --ver v6 \
      --idx "$run" \
      2>&1 | tee "$LOG"
  done
}

case "$RUN_VERSION" in
  aron) run_aron ;;
  v6)   run_v6 ;;
  both) run_aron; run_v6 ;;
  *)
    echo "ERROR: RUN_VERSION must be one of: aron | v6 | both"
    exit 1
    ;;
esac

echo "All done. Logs are in: $OUTDIR"
