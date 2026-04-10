#!/usr/bin/env bash
set -euo pipefail

# =========================================================
# Reverse-GNN / Variant C sweep
# Purpose:
#   ver=no + decoded graph temporary-view refinement
#   keep global defaults for all datasets,
#   but allow per-dataset overrides through map-style settings
#
# NEW for cora:
#   keep encoder trainable in phase-2 and use task-main loss
# =========================================================

PYTHON_BIN=${PYTHON_BIN:-python}
ENTRY=${ENTRY:-src/aron_main.py}

# -------------------- global knobs --------------------
EPOCHS=${EPOCHS:-1000}
#SEEDS=(${SEEDS:-0 1 2 3 4 5 6 7 8 9})
SEEDS=(${SEEDS:-10 11 12 13 14 15 16 17 18 19})
#DATASETS=(${DATASETS:-cora})
DATASETS=(${DATASETS:-cora citeseer Cora_ML LastFMAsia})

# -------------------- per-dataset degree threshold --------------------
declare -A DEG_THR
DEG_THR["cora"]=${DEGREE_THRESHOLD_CORA:-0.5}
DEG_THR["citeseer"]=${DEGREE_THRESHOLD_CITESEER:-0.5}
DEG_THR["Cora_ML"]=${DEGREE_THRESHOLD_CORAML:-0.5}
DEG_THR["LastFMAsia"]=${DEGREE_THRESHOLD_LASTFM:-0.5}

# -------------------- model / training globals --------------------
HIDDEN1=${HIDDEN1:-256}
HIDDEN2=${HIDDEN2:-64}
LR=${LR:-0.001}
DROPOUT=${DROPOUT:-0.3}
WEIGHT_DECAY=${WEIGHT_DECAY:-5e-4}

DECODER_RECON_WEIGHT=${DECODER_RECON_WEIGHT:-1.0}
EDITOR_PULL_STRENGTH=${EDITOR_PULL_STRENGTH:-0.10}

CLUSTER_METHOD=${CLUSTER_METHOD:-gmm}
CLUSTER_MODE=${CLUSTER_MODE:-any}
GMM_K=${GMM_K:-16}
GMM_TAU=${GMM_TAU:-0.55}

EVAL_LOG_EVERY=${EVAL_LOG_EVERY:-5}

DATESTR=${DATESTR:-$(date +%m%d)}
LOGDIR=${LOGDIR:-logs/${DATESTR}/reverse_gnn_varC_taskmain_unfrozen}
mkdir -p "${LOGDIR}"

# =========================================================
# Per-dataset maps
# Other datasets keep your current defaults.
# cora gets its own specific behavior.
# =========================================================

# edit start
declare -A EDIT_START_MAP
EDIT_START_MAP["cora"]=${EDIT_START_CORA:-220}
EDIT_START_MAP["citeseer"]=${EDIT_START_CITESEER:-180}
EDIT_START_MAP["Cora_ML"]=${EDIT_START_CORAML:-220}
EDIT_START_MAP["LastFMAsia"]=${EDIT_START_LASTFM:-250}

# kept for logging / compatibility, but set cora to zero for the new design
# phase-1 decoder warm-up uses recon on unpulled latent
# phase-2 uses decoder mainly for inference-time rewrite
declare -A COMPACTNESS_MAP
COMPACTNESS_MAP["cora"]=${COMPACTNESS_CORA:-0.0}
COMPACTNESS_MAP["citeseer"]=${COMPACTNESS_CITESEER:-0.0}
COMPACTNESS_MAP["Cora_ML"]=${COMPACTNESS_CORAML:-0.0}
COMPACTNESS_MAP["LastFMAsia"]=${COMPACTNESS_LASTFM:-0.0}

declare -A PRESERVE_MAP
PRESERVE_MAP["cora"]=${PRESERVE_CORA:-0.0}
PRESERVE_MAP["citeseer"]=${PRESERVE_CITESEER:-0.0}
PRESERVE_MAP["Cora_ML"]=${PRESERVE_CORAML:-0.0}
PRESERVE_MAP["LastFMAsia"]=${PRESERVE_LASTFM:-0.0}

declare -A RETAIN_RECON_MAP
RETAIN_RECON_MAP["cora"]=${RETAIN_RECON_CORA:-0.50}
RETAIN_RECON_MAP["citeseer"]=${RETAIN_RECON_CITESEER:-0.50}
RETAIN_RECON_MAP["Cora_ML"]=${RETAIN_RECON_CORAML:-0.50}
RETAIN_RECON_MAP["LastFMAsia"]=${RETAIN_RECON_LASTFM:-0.50}

declare -A RETAIN_CL_MAP
RETAIN_CL_MAP["cora"]=${RETAIN_CL_CORA:-0.20}
RETAIN_CL_MAP["citeseer"]=${RETAIN_CL_CITESEER:-0.20}
RETAIN_CL_MAP["Cora_ML"]=${RETAIN_CL_CORAML:-0.20}
RETAIN_CL_MAP["LastFMAsia"]=${RETAIN_CL_LASTFM:-0.20}

# phase-2 encoder LR scale (relative to base LR); cora now tunes encoder too
declare -A PHASE2_LR_SCALE_MAP
PHASE2_LR_SCALE_MAP["cora"]=${PHASE2_LR_SCALE_CORA:-0.10}
PHASE2_LR_SCALE_MAP["citeseer"]=${PHASE2_LR_SCALE_CITESEER:-0.05}
PHASE2_LR_SCALE_MAP["Cora_ML"]=${PHASE2_LR_SCALE_CORAML:-0.05}
PHASE2_LR_SCALE_MAP["LastFMAsia"]=${PHASE2_LR_SCALE_LASTFM:-0.05}

# decoded graph rewrite
declare -A ADD_RATIO_MAP
ADD_RATIO_MAP["cora"]=${ADD_RATIO_CORA:-0.005}
ADD_RATIO_MAP["citeseer"]=${ADD_RATIO_CITESEER:-0.02}
ADD_RATIO_MAP["Cora_ML"]=${ADD_RATIO_CORAML:-0.02}
ADD_RATIO_MAP["LastFMAsia"]=${ADD_RATIO_LASTFM:-0.02}

declare -A REMOVE_RATIO_MAP
REMOVE_RATIO_MAP["cora"]=${REMOVE_RATIO_CORA:-0.00}
REMOVE_RATIO_MAP["citeseer"]=${REMOVE_RATIO_CITESEER:-0.00}
REMOVE_RATIO_MAP["Cora_ML"]=${REMOVE_RATIO_CORAML:-0.00}
REMOVE_RATIO_MAP["LastFMAsia"]=${REMOVE_RATIO_LASTFM:-0.00}

declare -A AUG_BOUND_MAP
AUG_BOUND_MAP["cora"]=${AUG_BOUND_CORA:-0.05}
AUG_BOUND_MAP["citeseer"]=${AUG_BOUND_CITESEER:--1.0}
AUG_BOUND_MAP["Cora_ML"]=${AUG_BOUND_CORAML:--1.0}
AUG_BOUND_MAP["LastFMAsia"]=${AUG_BOUND_LASTFM:--1.0}

# phase-2 mode flags
# these require your training-code patch to support:
#   --phase2_task_main_loss
#   --edit_phase_edit_weight
declare -A PHASE2_TASK_MAIN_MAP
PHASE2_TASK_MAIN_MAP["cora"]=${PHASE2_TASK_MAIN_CORA:-1}
PHASE2_TASK_MAIN_MAP["citeseer"]=${PHASE2_TASK_MAIN_CITESEER:-1}
PHASE2_TASK_MAIN_MAP["Cora_ML"]=${PHASE2_TASK_MAIN_CORAML:-1}
PHASE2_TASK_MAIN_MAP["LastFMAsia"]=${PHASE2_TASK_MAIN_LASTFM:-1}

declare -A EDIT_PHASE_EDIT_WEIGHT_MAP
EDIT_PHASE_EDIT_WEIGHT_MAP["cora"]=${EDIT_PHASE_EDIT_WEIGHT_CORA:-0.10}
EDIT_PHASE_EDIT_WEIGHT_MAP["citeseer"]=${EDIT_PHASE_EDIT_WEIGHT_CITESEER:-0.10}
EDIT_PHASE_EDIT_WEIGHT_MAP["Cora_ML"]=${EDIT_PHASE_EDIT_WEIGHT_CORAML:-0.10}
EDIT_PHASE_EDIT_WEIGHT_MAP["LastFMAsia"]=${EDIT_PHASE_EDIT_WEIGHT_LASTFM:-0.10}

declare -A REQUIRE_C0P_ENDPOINT_MAP
REQUIRE_C0P_ENDPOINT_MAP["cora"]=${REQUIRE_C0P_ENDPOINT_CORA:-1}
REQUIRE_C0P_ENDPOINT_MAP["citeseer"]=${REQUIRE_C0P_ENDPOINT_CITESEER:-1}
REQUIRE_C0P_ENDPOINT_MAP["Cora_ML"]=${REQUIRE_C0P_ENDPOINT_CORAML:-1}
REQUIRE_C0P_ENDPOINT_MAP["LastFMAsia"]=${REQUIRE_C0P_ENDPOINT_LASTFM:-1}

# NEW: whether to freeze encoder in phase-2
declare -A PHASE2_FREEZE_ENCODER_MAP
PHASE2_FREEZE_ENCODER_MAP["cora"]=${PHASE2_FREEZE_ENCODER_CORA:-0}
PHASE2_FREEZE_ENCODER_MAP["citeseer"]=${PHASE2_FREEZE_ENCODER_CITESEER:-0}
PHASE2_FREEZE_ENCODER_MAP["Cora_ML"]=${PHASE2_FREEZE_ENCODER_CORAML:-0}
PHASE2_FREEZE_ENCODER_MAP["LastFMAsia"]=${PHASE2_FREEZE_ENCODER_LASTFM:-0}

# whether decoder stays inference-only in phase-2
declare -A PHASE2_DECODER_INFER_ONLY_MAP
PHASE2_DECODER_INFER_ONLY_MAP["cora"]=${PHASE2_DECODER_INFER_ONLY_CORA:-0}
PHASE2_DECODER_INFER_ONLY_MAP["citeseer"]=${PHASE2_DECODER_INFER_ONLY_CITESEER:-0}
PHASE2_DECODER_INFER_ONLY_MAP["Cora_ML"]=${PHASE2_DECODER_INFER_ONLY_CORAML:-0}
PHASE2_DECODER_INFER_ONLY_MAP["LastFMAsia"]=${PHASE2_DECODER_INFER_ONLY_LASTFM:-0}

run_one () {
  local ds="$1"
  local seed="$2"
  local idx="$3"

  local degthr="${DEG_THR[$ds]}"
  local edit_start="${EDIT_START_MAP[$ds]}"
  local compactness_weight="${COMPACTNESS_MAP[$ds]}"
  local preserve_weight="${PRESERVE_MAP[$ds]}"
  local retain_recon_weight="${RETAIN_RECON_MAP[$ds]}"
  local retain_cl_weight="${RETAIN_CL_MAP[$ds]}"
  local phase2_encoder_lr_scale="${PHASE2_LR_SCALE_MAP[$ds]}"
  local decoded_add_ratio="${ADD_RATIO_MAP[$ds]}"
  local decoded_remove_ratio="${REMOVE_RATIO_MAP[$ds]}"
  local decoded_graph_aug_bound="${AUG_BOUND_MAP[$ds]}"
  local phase2_task_main="${PHASE2_TASK_MAIN_MAP[$ds]}"
  local edit_phase_edit_weight="${EDIT_PHASE_EDIT_WEIGHT_MAP[$ds]}"
  local require_c0p_endpoint="${REQUIRE_C0P_ENDPOINT_MAP[$ds]}"
  local phase2_freeze_encoder="${PHASE2_FREEZE_ENCODER_MAP[$ds]}"
  local phase2_decoder_infer_only="${PHASE2_DECODER_INFER_ONLY_MAP[$ds]}"

  local tag="revgnn_varC_es${edit_start}_add${decoded_add_ratio}_rm${decoded_remove_ratio}_cp${compactness_weight}_pv${preserve_weight}_rr${retain_recon_weight}_rc${retain_cl_weight}_frz${phase2_freeze_encoder}_lrp2${phase2_encoder_lr_scale}_k${GMM_K}_tau${GMM_TAU}"
  local logfile="${LOGDIR}/${ds}_${tag}_seed${seed}_idx${idx}.log"

  echo "===================================================="
  echo "RUN dataset=${ds} seed=${seed} idx=${idx}"
  echo "log=${logfile}"
  echo "----------------------------------------------------"
  echo "EFFECTIVE SETTINGS:"
  echo "  ver=no"
  echo "  edit_start_epoch=${edit_start}"
  echo "  decoder_recon_weight=${DECODER_RECON_WEIGHT}"
  echo "  compactness_weight=${compactness_weight}"
  echo "  preserve_weight=${preserve_weight}"
  echo "  retain_recon_weight=${retain_recon_weight}"
  echo "  retain_cl_weight=${retain_cl_weight}"
  echo "  editor_pull_strength=${EDITOR_PULL_STRENGTH}"
  echo "  phase2_freeze_encoder=${phase2_freeze_encoder}"
  echo "  phase2_decoder_inference_only=${phase2_decoder_infer_only}"
  echo "  phase2_encoder_lr_scale=${phase2_encoder_lr_scale}"
  echo "  decoded_add_ratio=${decoded_add_ratio}"
  echo "  decoded_remove_ratio=${decoded_remove_ratio}"
  echo "  decoded_graph_aug_bound=${decoded_graph_aug_bound}"
  echo "  cluster_method=${CLUSTER_METHOD}"
  echo "  cluster_mode=${CLUSTER_MODE}"
  echo "  gmm_k=${GMM_K}"
  echo "  gmm_tau=${GMM_TAU}"
  echo "  temporary_view_only=1"
  echo "  same_cluster_only=1"
  echo "  freeze_c0p_at_edit_start=1"
  echo "  phase2_task_main_loss=${phase2_task_main}"
  echo "  decoded_require_c0p_endpoint=${require_c0p_endpoint}"
  if [[ "${phase2_task_main}" == "1" ]]; then
    echo "  edit_phase_edit_weight=${edit_phase_edit_weight}"
  fi

  cmd=(
    ${PYTHON_BIN} ${ENTRY}
    --dataset "${ds}"
    --ver no
    --seed "${seed}"
    --idx "${idx}"
    --epochs "${EPOCHS}"
    --hidden1 "${HIDDEN1}"
    --hidden2 "${HIDDEN2}"
    --lr "${LR}"
    --dropout "${DROPOUT}"
    --weight_decay "${WEIGHT_DECAY}"
    --degree_threshold "${degthr}"
    --cluster_method "${CLUSTER_METHOD}"
    --cluster_mode "${CLUSTER_MODE}"
    --gmm_k "${GMM_K}"
    --gmm_tau "${GMM_TAU}"
    --use_edited_decoder
    --decoder_type bilinear
    --decoder_recon_weight "${DECODER_RECON_WEIGHT}"
    --compactness_weight "${compactness_weight}"
    --preserve_weight "${preserve_weight}"
    --separate_edit_training
    --edit_phase_retain_recon_weight "${retain_recon_weight}"
    --edit_phase_retain_cl_weight "${retain_cl_weight}"
    --editor_pull_strength "${EDITOR_PULL_STRENGTH}"
    --edit_start_epoch "${edit_start}"
    --freeze_c0p_at_edit_start
    --use_decoded_graph_augment
    --decoded_temporary_view_only
    --decoded_same_cluster_only
    --decoded_add_ratio "${decoded_add_ratio}"
    --decoded_remove_ratio "${decoded_remove_ratio}"
    --decoded_graph_aug_bound "${decoded_graph_aug_bound}"
    --eval_log_every "${EVAL_LOG_EVERY}"
    --sweep_mode
    --date "${DATESTR}"
  )

  # requires patched training/main code
  if [[ "${phase2_task_main}" == "1" ]]; then
    cmd+=(--phase2_task_main_loss)
    cmd+=(--edit_phase_edit_weight "${edit_phase_edit_weight}")
  fi

  if [[ "${require_c0p_endpoint}" == "1" ]]; then
    cmd+=(--decoded_require_c0p_endpoint)
  fi

  if [[ "${phase2_decoder_infer_only}" == "1" ]]; then
    cmd+=(--phase2_decoder_inference_only)
  else
    cmd+=(--no_phase2_decoder_inference_only)
  fi

  # phase-2 encoder handling
  if [[ "${phase2_freeze_encoder}" == "1" ]]; then
    cmd+=(--phase2_freeze_encoder)
  else
    cmd+=(--phase2_tune_encoder)
    cmd+=(--edit_phase_encoder_lr_scale "${phase2_encoder_lr_scale}")
  fi

  "${cmd[@]}" 2>&1 | tee "${logfile}"
}

for ds in "${DATASETS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    run_one "${ds}" "${seed}" "${seed}"
  done
done