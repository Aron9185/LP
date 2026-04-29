#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/retro/ARON"
PY="/home/retro/anaconda3/envs/pyg/bin/python"
RUNNER="${ROOT}/experiments/reversegnn-compactness/run_heart_comparison.py"
LOG_DIR="${ROOT}/experiments/reversegnn-compactness/results"
LOG_FILE="${LOG_DIR}/heart_threshold_remove_diag_grid_tmux.log"
EPOCHS="${EPOCHS:-700}"
EDIT_START="${EDIT_START:-100}"
MAX_REMOVE_PER_ROUND="${MAX_REMOVE_PER_ROUND:-50}"
MAX_WORKERS="${MAX_WORKERS:-1}"

mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

cd "${ROOT}"

echo "[threshold-remove-grid] started $(date -Is)"
echo "[threshold-remove-grid] log=${LOG_FILE}"
echo "[threshold-remove-grid] epochs=${EPOCHS}"
echo "[threshold-remove-grid] edit_start=${EDIT_START}"
echo "[threshold-remove-grid] max_remove_per_round=${MAX_REMOVE_PER_ROUND}"
echo "[threshold-remove-grid] max_workers=${MAX_WORKERS}"
echo "[threshold-remove-grid] protocol=heart samples.npy, full validation, checkpoint=hit10"

SEEDS=(${SEEDS:-0 1 2 3 4 5 6 7 8 9})
FORCE="${FORCE:-0}"
force_args=()
if [[ "${FORCE}" == "1" ]]; then
  force_args=(--force)
fi

pulls=(${PULLS:-1.0})
adds=(${ADDS:-0.01 0.02})
remove_thresholds=(${REMOVE_THRESHOLDS:-0.05 0.10 0.20})
total=$(( ${#pulls[@]} * ${#adds[@]} * ${#remove_thresholds[@]} ))
combo=0

for remove_threshold in "${remove_thresholds[@]}"; do
  for pull in "${pulls[@]}"; do
    for add in "${adds[@]}"; do
      combo=$((combo + 1))
      ptag="${pull/./p}"
      atag="${add/./p}"
      rtag="${remove_threshold/./p}"
      prefix="heart_remove_thr_p${ptag}_a${atag}_rt${rtag}_mr${MAX_REMOVE_PER_ROUND}"

      echo
      echo "[combo ${combo}/${total}] pull=${pull} add=${add} remove_threshold=${remove_threshold} max_remove=${MAX_REMOVE_PER_ROUND} prefix=${prefix}"
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
        --decoded-remove-ratio 0.0 \
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
        --extra_flag=hit10 \
        --extra_flag=--decoded_remove_threshold \
        --extra_flag="${remove_threshold}" \
        --extra_flag=--decoded_max_remove_per_round \
        --extra_flag="${MAX_REMOVE_PER_ROUND}"
    done
  done
done

echo
echo "[threshold-remove-grid] finished $(date -Is)"
