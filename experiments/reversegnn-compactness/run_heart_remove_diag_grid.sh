#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/retro/ARON"
PY="/home/retro/anaconda3/envs/pyg/bin/python"
RUNNER="${ROOT}/experiments/reversegnn-compactness/run_heart_comparison.py"
LOG_DIR="${ROOT}/experiments/reversegnn-compactness/results"
LOG_FILE="${LOG_DIR}/heart_remove_diag_grid_tmux.log"
EPOCHS="${EPOCHS:-700}"
MAX_WORKERS="${MAX_WORKERS:-1}"

mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

cd "${ROOT}"

echo "[remove-grid] started $(date -Is)"
echo "[remove-grid] log=${LOG_FILE}"
echo "[remove-grid] epochs=${EPOCHS}"
echo "[remove-grid] max_workers=${MAX_WORKERS}"
echo "[remove-grid] protocol=heart samples.npy, full validation, checkpoint=hit10"

SEEDS=(${SEEDS:-0 1 2 3 4 5 6 7 8 9})
pulls=(1.0)
adds=(0.01 0.02)
removes=(0.002 0.005 0.01)
total=$(( ${#pulls[@]} * ${#adds[@]} * ${#removes[@]} ))
combo=0

for pull in "${pulls[@]}"; do
  for add in "${adds[@]}"; do
    for remove in "${removes[@]}"; do
      combo=$((combo + 1))
      ptag="${pull/./p}"
      atag="${add/./p}"
      rtag="${remove/./p}"
      prefix="heart_hit10_remove_diag_p${ptag}_a${atag}_rm${rtag}"

      echo
      echo "[combo ${combo}/${total}] pull=${pull} add=${add} remove=${remove} prefix=${prefix}"
      "${PY}" "${RUNNER}" \
        --prefix "${prefix}" \
        --datasets cora citeseer \
        --seeds "${SEEDS[@]}" \
        --epochs "${EPOCHS}" \
        --max-workers "${MAX_WORKERS}" \
        --configs new_dynamic_bilinear_hybrid_radius_heart_like \
        --editor-pull-strength "${pull}" \
        --decoded-add-ratio "${add}" \
        --decoded-remove-ratio "${remove}" \
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
echo "[remove-grid] finished $(date -Is)"
