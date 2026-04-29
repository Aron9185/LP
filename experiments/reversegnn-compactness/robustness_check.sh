#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

source /home/retro/anaconda3/etc/profile.d/conda.sh
conda activate pyg

EXPERIMENT_ROOT="${ARON_EXPERIMENT_ROOT:-${SCRIPT_DIR}}"
LOG_DIR="${EXPERIMENT_ROOT}/results/sweep_logs"
mkdir -p "${LOG_DIR}"
cd "${REPO_ROOT}"

BASE_ARGS="--dataset cora --epochs 700 --use_edited_decoder --decoder_objective recon --compactness_objective radius --use_decoded_graph_augment --pull_mask_scope cp --compactness_mask_scope cp --rewrite_endpoint_scope c0p --decoded_same_cluster_only --decoded_require_c0p_endpoint --decoded_temporary_view_only --freeze_c0p_at_edit_start --decoded_add_ratio 0.01 --editor_pull_strength 0.1 --decoded_graph_aug_bound -1.0 --ver no"

echo "=== Robustness Check: Pull=0.1, Add Ratio=0.01, Aug Bound=-1.0 ==="
python src/aron_main.py $BASE_ARGS --seed 0 > "${LOG_DIR}/cora_robust_s0.txt" &
python src/aron_main.py $BASE_ARGS --seed 1 > "${LOG_DIR}/cora_robust_s1.txt" &
python src/aron_main.py $BASE_ARGS --seed 2 > "${LOG_DIR}/cora_robust_s2.txt" &
wait
echo "Done."
