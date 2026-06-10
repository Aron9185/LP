#!/usr/bin/env bash
set -euo pipefail

cd /home/retro/ARON

PY="${PY:-/home/retro/anaconda3/envs/pyg/bin/python}"
RUN="${RUN:-experiments/reversegnn-compactness/run_heart_pair_scorer_editor.py}"

STAMP="${STAMP:-20260608}"
SEEDS=(${SEEDS:-0 3})
MAX_WORKERS="${MAX_WORKERS:-1}"
MLP_PAIR_MAX_ROWS="${MLP_PAIR_MAX_ROWS:-16}"
EPOCHS="${EPOCHS:-700}"

COMMON=(
  --split-mode random
  --random-checkpoint-metric hit10
  --seeds "${SEEDS[@]}"
  --epochs "${EPOCHS}"
  --max-workers "${MAX_WORKERS}"
  --mlp-pair-max-rows "${MLP_PAIR_MAX_ROWS}"
  --configs two_decoder_ncnc_pred
  --decoded-remove-ratio 0.0
  --editor-pull-strength 0.25
  --decoded-add-ratio 0.20
  --decoded-graph-aug-bound 0.10
  --eval-log-every 50
  --decoder-diag-every 100
  --edit-metric-every 100
  --skip-train-acc
  --extra_flag=--skip_oom_epoch
  --extra_flag=--decoded_require_c0p_noncompact_endpoint
  --extra_flag=--decoded_add_degree_target
  --extra_flag=1
  --extra_flag=--decoded_add_degree_target_scope
  --extra_flag=intra_cluster
  --extra_flag=--decoded_add_degree_target_nodes
  --extra_flag=cp
  --extra_flag=--decoded_guarantee_degree_target
)

run_tune() {
  local dataset="$1"
  local policy="$2"
  local tag="$3"
  shift 3

  local prefix="random_refined_vgnae_${dataset}_${policy}_${tag}_${STAMP}"
  local policy_flags=()
  if [[ "${policy}" == "aq002" ]]; then
    policy_flags=(--extra_flag=--decoded_add_quantile --extra_flag=0.02)
  fi

  echo
  date
  echo "[refined-vgnae] dataset=${dataset} policy=${policy} tag=${tag} prefix=${prefix}"
  echo "[refined-vgnae] extra=$*"
  "${PY}" "${RUN}" \
    --prefix "${prefix}" \
    --datasets "${dataset}" \
    "${COMMON[@]}" \
    "${policy_flags[@]}" \
    "$@"
}

run_dataset_grid() {
  local dataset="$1"
  local policy="$2"

  # Current default, included as a same-run anchor.
  run_tune "${dataset}" "${policy}" h256_z64_do03_lr001_b1_fm01 \
    --extra_flag=--hidden1 --extra_flag=256 \
    --extra_flag=--hidden2 --extra_flag=64 \
    --extra_flag=--dropout --extra_flag=0.3 \
    --extra_flag=--lr --extra_flag=0.001 \
    --extra_flag=--beta --extra_flag=1.0 \
    --extra_flag=--feat_mask_ratio --extra_flag=0.1

  # Refined-GAE-style wider, regularized encoder.
  run_tune "${dataset}" "${policy}" h512_z128_do04_lr001_b1_fm01 \
    --extra_flag=--hidden1 --extra_flag=512 \
    --extra_flag=--hidden2 --extra_flag=128 \
    --extra_flag=--dropout --extra_flag=0.4 \
    --extra_flag=--lr --extra_flag=0.001 \
    --extra_flag=--beta --extra_flag=1.0 \
    --extra_flag=--feat_mask_ratio --extra_flag=0.1

  # Large/dropout-heavy Cora-like Refined-GAE recipe, but with our editor.
  run_tune "${dataset}" "${policy}" h1024_z128_do06_lr003_b1_fm01 \
    --extra_flag=--hidden1 --extra_flag=1024 \
    --extra_flag=--hidden2 --extra_flag=128 \
    --extra_flag=--dropout --extra_flag=0.6 \
    --extra_flag=--lr --extra_flag=0.003 \
    --extra_flag=--beta --extra_flag=1.0 \
    --extra_flag=--feat_mask_ratio --extra_flag=0.1

  # GAE-like low-KL objective: Refined-GAE argues old GAE baselines are undertuned.
  run_tune "${dataset}" "${policy}" h512_z128_do04_lr001_b001_fm01 \
    --extra_flag=--hidden1 --extra_flag=512 \
    --extra_flag=--hidden2 --extra_flag=128 \
    --extra_flag=--dropout --extra_flag=0.4 \
    --extra_flag=--lr --extra_flag=0.001 \
    --extra_flag=--beta --extra_flag=0.01 \
    --extra_flag=--feat_mask_ratio --extra_flag=0.1

  # Stronger input masking, inspired by MaskGAE/Refined-GAE maskinput.
  run_tune "${dataset}" "${policy}" h512_z128_do04_lr001_b001_fm03 \
    --extra_flag=--hidden1 --extra_flag=512 \
    --extra_flag=--hidden2 --extra_flag=128 \
    --extra_flag=--dropout --extra_flag=0.4 \
    --extra_flag=--lr --extra_flag=0.001 \
    --extra_flag=--beta --extra_flag=0.01 \
    --extra_flag=--feat_mask_ratio --extra_flag=0.3

  # More hard negatives for the NCNC-style prediction decoder.
  run_tune "${dataset}" "${policy}" h512_z128_do04_lr001_b001_fm01_neg32 \
    --prediction-rank-neg-k 32 \
    --prediction-rank-pool-factor 8 \
    --extra_flag=--hidden1 --extra_flag=512 \
    --extra_flag=--hidden2 --extra_flag=128 \
    --extra_flag=--dropout --extra_flag=0.4 \
    --extra_flag=--lr --extra_flag=0.001 \
    --extra_flag=--beta --extra_flag=0.01 \
    --extra_flag=--feat_mask_ratio --extra_flag=0.1
}

echo "[refined-vgnae] started $(date -Is)"
echo "[refined-vgnae] seeds=${SEEDS[*]} epochs=${EPOCHS} workers=${MAX_WORKERS}"
echo "[refined-vgnae] Cora policy=aq002; Citeseer policy=current"

run_dataset_grid cora aq002
run_dataset_grid citeseer current

echo "[refined-vgnae] finished $(date -Is)"
