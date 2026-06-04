#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/retro/ARON"
PY="/home/retro/anaconda3/envs/pyg/bin/python"
RUNNER="${ROOT}/experiments/reversegnn-compactness/run_heart_pair_scorer_editor.py"
LOG_DIR="${ROOT}/experiments/reversegnn-compactness/results"
STAMP="${STAMP:-20260502_cora_rescue}"
LOG_FILE="${LOG_DIR}/heart_two_decoder_hard_remove_cora_rescue_${STAMP}_tmux.log"

EPOCHS="${EPOCHS:-700}"
SEEDS=(${SEEDS:-0 1 2})
DATASETS=(${DATASETS:-cora})
MAX_WORKERS="${MAX_WORKERS:-1}"
MLP_PAIR_MAX_ROWS="${MLP_PAIR_MAX_ROWS:-2}"
CONFIGS=(${CONFIGS:-two_decoder_pred_remove})

PREDICTION_RANK_NEG_K="${PREDICTION_RANK_NEG_K:-8}"
PREDICTION_RANK_POOL_FACTOR="${PREDICTION_RANK_POOL_FACTOR:-4}"
PREDICTION_ENCODER_WEIGHT="${PREDICTION_ENCODER_WEIGHT:-0.0}"
DECODER_RANK_NEG_K="${DECODER_RANK_NEG_K:-4}"
DECODER_RANK_POOL_FACTOR="${DECODER_RANK_POOL_FACTOR:-2}"

mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

cd "${ROOT}"

echo "[two-decoder-hard-remove-cora-rescue] started $(date -Is)"
echo "[two-decoder-hard-remove-cora-rescue] log=${LOG_FILE}"
echo "[two-decoder-hard-remove-cora-rescue] epochs=${EPOCHS} datasets=${DATASETS[*]} seeds=${SEEDS[*]} workers=${MAX_WORKERS}"
echo "[two-decoder-hard-remove-cora-rescue] configs=${CONFIGS[*]} mlp_pair_max_rows=${MLP_PAIR_MAX_ROWS}"
echo "[two-decoder-hard-remove-cora-rescue] pred_neg_k=${PREDICTION_RANK_NEG_K} pred_pool=${PREDICTION_RANK_POOL_FACTOR} pred_encoder_w=${PREDICTION_ENCODER_WEIGHT}"
echo "[two-decoder-hard-remove-cora-rescue] decoder_neg_k=${DECODER_RANK_NEG_K} decoder_pool=${DECODER_RANK_POOL_FACTOR}"

extra_common=(
  --extra_flag=--skip_oom_epoch
  --extra_flag=--decoded_degree_floor
  --extra_flag=0
)

run_case() {
  local name="$1"
  local add_ratio="$2"
  local remove_ratio="$3"
  local prefix="heart_two_decoder_hard_remove_${name}_${STAMP}"

  echo
  echo "[case] ${name} add=${add_ratio} remove=${remove_ratio} prefix=${prefix}"
  "${PY}" "${RUNNER}" \
    --prefix "${prefix}" \
    --datasets "${DATASETS[@]}" \
    --seeds "${SEEDS[@]}" \
    --epochs "${EPOCHS}" \
    --max-workers "${MAX_WORKERS}" \
    --mlp-pair-max-rows "${MLP_PAIR_MAX_ROWS}" \
    --prediction-rank-neg-k "${PREDICTION_RANK_NEG_K}" \
    --prediction-rank-pool-factor "${PREDICTION_RANK_POOL_FACTOR}" \
    --prediction-encoder-weight "${PREDICTION_ENCODER_WEIGHT}" \
    --decoder-rank-neg-k "${DECODER_RANK_NEG_K}" \
    --decoder-rank-pool-factor "${DECODER_RANK_POOL_FACTOR}" \
    --configs "${CONFIGS[@]}" \
    --decoded-add-ratio "${add_ratio}" \
    --decoded-remove-ratio "${remove_ratio}" \
    "${extra_common[@]}"
}

run_case "floor0_add001_rm001" 0.01 0.01
run_case "floor0_add001_rm003" 0.01 0.03
run_case "floor0_add001_rm005" 0.01 0.05
run_case "floor0_add0005_rm005" 0.005 0.05

echo
echo "[two-decoder-hard-remove-cora-rescue] finished $(date -Is)"
