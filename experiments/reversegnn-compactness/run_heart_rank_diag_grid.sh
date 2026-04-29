#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/retro/ARON"
PY="/home/retro/anaconda3/envs/pyg/bin/python"
RUNNER="${ROOT}/experiments/reversegnn-compactness/run_heart_comparison.py"
LOG_DIR="${ROOT}/experiments/reversegnn-compactness/results"
LOG_FILE="${LOG_DIR}/heart_rank_diag_grid_tmux.log"
EPOCHS="${EPOCHS:-700}"
EDIT_START="${EDIT_START:-100}"
MAX_WORKERS="${MAX_WORKERS:-1}"

mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

cd "${ROOT}"

echo "[rank-grid] started $(date -Is)"
echo "[rank-grid] log=${LOG_FILE}"
echo "[rank-grid] epochs=${EPOCHS}"
echo "[rank-grid] edit_start=${EDIT_START}"
echo "[rank-grid] max_workers=${MAX_WORKERS}"
echo "[rank-grid] protocol=heart samples.npy, full validation, checkpoint=hit10"

SEEDS=(${SEEDS:-0 1 2 3 4 5 6 7 8 9})
FORCE="${FORCE:-0}"
force_args=()
if [[ "${FORCE}" == "1" ]]; then
  force_args=(--force)
fi

pulls=(${PULLS:-1.0})
adds=(${ADDS:-0.01 0.02})
rank_specs=(${RANK_SPECS:-16:8 32:8 32:16})
total=$(( ${#pulls[@]} * ${#adds[@]} * ${#rank_specs[@]} ))
combo=0

for spec in "${rank_specs[@]}"; do
  neg_k="${spec%%:*}"
  pool_factor="${spec##*:}"
  for pull in "${pulls[@]}"; do
    for add in "${adds[@]}"; do
      combo=$((combo + 1))
      ptag="${pull/./p}"
      atag="${add/./p}"
      prefix="heart_rank_es${EDIT_START}_k${neg_k}_pool${pool_factor}_p${ptag}_a${atag}"

      echo
      echo "[combo ${combo}/${total}] edit_start=${EDIT_START} neg_k=${neg_k} pool=${pool_factor} pull=${pull} add=${add} prefix=${prefix}"
      "${PY}" "${RUNNER}" \
        --prefix "${prefix}" \
        --datasets cora citeseer \
        --seeds "${SEEDS[@]}" \
        --epochs "${EPOCHS}" \
        --edit-start-epoch "${EDIT_START}" \
        --max-workers "${MAX_WORKERS}" \
        "${force_args[@]}" \
        --configs new_dynamic_bilinear_hybrid_radius_heart_like \
        --editor-pull-strength "${pull}" \
        --decoded-add-ratio "${add}" \
        --decoder-rank-neg-k "${neg_k}" \
        --decoder-rank-pool-factor "${pool_factor}" \
        --extra_flag=--split_mode \
        --extra_flag=heart \
        --extra_flag=--heart_data_dir \
        --extra_flag=dataset \
        --extra_flag=--heart_filename \
        --extra_flag=samples.npy \
        --extra_flag=--heart_eval_every \
        --extra_flag=5 \
        --extra_flag=--heart_val_frac \
        --extra_flag=1.0 \
        --extra_flag=--heart_checkpoint_metric \
        --extra_flag=hit10
    done
  done
done

echo
echo "[rank-grid] finished $(date -Is)"
