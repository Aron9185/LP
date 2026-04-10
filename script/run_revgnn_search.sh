#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN=${PYTHON_BIN:-python}
ENTRY=${ENTRY:-src/aron_main.py}
DATESTR=${DATESTR:-$(date +%m%d)}
ROOT_LOGDIR=${ROOT_LOGDIR:-logs/${DATESTR}/search_revgnn}
mkdir -p "${ROOT_LOGDIR}"

EPOCHS=${EPOCHS:-700}
SEEDS=(${SEEDS:-0 1 2})
DATASETS=(${DATASETS:-cora Cora_ML LastFMAsia})

HIDDEN1=${HIDDEN1:-256}
HIDDEN2=${HIDDEN2:-64}
LR=${LR:-0.001}
DROPOUT=${DROPOUT:-0.3}
WEIGHT_DECAY=${WEIGHT_DECAY:-5e-4}

VER=no
CLUSTER_METHOD=gmm
CLUSTER_MODE=any
DECODER_TYPE=bilinear

declare -A DEG_THR
DEG_THR["cora"]=0.5
DEG_THR["Cora_ML"]=0.5
DEG_THR["LastFMAsia"]=0.5

CORA_EDIT_STARTS=(180 200 220)
CORA_COMPACT=(0.1 0.2)
CORA_PRESERVE=(0.2 0.5)
CORA_PULL=(0.05 0.10)
CORA_RET_RECON=(0.1)
CORA_RET_CL=(0.05)
CORA_GMM_K=(16)
CORA_GMM_TAU=(0.55)

CORAML_EDIT_STARTS=(200 250)
CORAML_COMPACT=(0.05 0.10)
CORAML_PRESERVE=(0.5 1.0)
CORAML_PULL=(0.03 0.05)
CORAML_RET_RECON=(0.1 0.2)
CORAML_RET_CL=(0.05 0.10)
CORAML_GMM_K=(16)
CORAML_GMM_TAU=(0.55)

LASTFM_EDIT_STARTS=(200 300)
LASTFM_COMPACT=(0.05 0.10)
LASTFM_PRESERVE=(0.2 0.5)
LASTFM_PULL=(0.05)
LASTFM_RET_RECON=(0.1)
LASTFM_RET_CL=(0.05)
LASTFM_GMM_K=(8 16 32)
LASTFM_GMM_TAU=(0.40 0.55 0.70)
LASTFM_FREEZE_TARGET=(1 0)

run_one () {
  local ds="$1"; local seed="$2"; local idx="$3"; local edit_start="$4"; local compact="$5"; local preserve="$6"; local pull="$7"; local rr="$8"; local rc="$9"; local gmmk="${10}"; local gmmtau="${11}"; local freeze_targets="${12}"
  local degthr="${DEG_THR[$ds]}"
  local run_tag="varC_tmp_es${edit_start}_cp${compact}_pv${preserve}_pull${pull}_rr${rr}_rc${rc}_k${gmmk}_tau${gmmtau}_fz${freeze_targets}"
  local logfile="${ROOT_LOGDIR}/${ds}_${run_tag}_seed${seed}_idx${idx}.log"

  echo "===================================================="
  echo "RUN dataset=${ds} seed=${seed} idx=${idx} tag=${run_tag}"
  echo "log=${logfile}"
  echo "===================================================="

  cmd=(
    ${PYTHON_BIN} ${ENTRY}
    --dataset "${ds}"
    --ver "${VER}"
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
    --gmm_k "${gmmk}"
    --gmm_tau "${gmmtau}"
    --use_edited_decoder
    --decoder_type "${DECODER_TYPE}"
    --decoder_recon_weight 1.0
    --compactness_weight "${compact}"
    --preserve_weight "${preserve}"
    --separate_edit_training
    --edit_phase_retain_recon_weight "${rr}"
    --edit_phase_retain_cl_weight "${rc}"
    --editor_pull_strength "${pull}"
    --edit_start_epoch "${edit_start}"
    --use_decoded_graph_augment
    --decoded_temporary_view_only
    --decoded_same_cluster_only
    --decoded_add_ratio 0.02
    --decoded_remove_ratio 0.0
    --phase2_tune_encoder
    --edit_phase_encoder_lr_scale 0.1
    --eval_log_every 5
    --run_tag "${run_tag}"
    --sweep_mode
    --date "${DATESTR}"
  )

  if [[ "${freeze_targets}" == "1" ]]; then
    cmd+=(--freeze_c0p_at_edit_start)
  else
    cmd+=(--dynamic_c0p_targets)
  fi

  "${cmd[@]}" 2>&1 | tee "${logfile}"
}

search_cora () {
  for seed in "${SEEDS[@]}"; do
    for es in "${CORA_EDIT_STARTS[@]}"; do
      for cp in "${CORA_COMPACT[@]}"; do
        for pv in "${CORA_PRESERVE[@]}"; do
          for pull in "${CORA_PULL[@]}"; do
            for rr in "${CORA_RET_RECON[@]}"; do
              for rc in "${CORA_RET_CL[@]}"; do
                for k in "${CORA_GMM_K[@]}"; do
                  for tau in "${CORA_GMM_TAU[@]}"; do
                    run_one "cora" "${seed}" "${seed}" "${es}" "${cp}" "${pv}" "${pull}" "${rr}" "${rc}" "${k}" "${tau}" "1"
                  done
                done
              done
            done
          done
        done
      done
    done
  done
}

search_coraml () {
  for seed in "${SEEDS[@]}"; do
    for es in "${CORAML_EDIT_STARTS[@]}"; do
      for cp in "${CORAML_COMPACT[@]}"; do
        for pv in "${CORAML_PRESERVE[@]}"; do
          for pull in "${CORAML_PULL[@]}"; do
            for rr in "${CORAML_RET_RECON[@]}"; do
              for rc in "${CORAML_RET_CL[@]}"; do
                for k in "${CORAML_GMM_K[@]}"; do
                  for tau in "${CORAML_GMM_TAU[@]}"; do
                    run_one "Cora_ML" "${seed}" "${seed}" "${es}" "${cp}" "${pv}" "${pull}" "${rr}" "${rc}" "${k}" "${tau}" "1"
                  done
                done
              done
            done
          done
        done
      done
    done
  done
}

search_lastfm () {
  for seed in "${SEEDS[@]}"; do
    for es in "${LASTFM_EDIT_STARTS[@]}"; do
      for cp in "${LASTFM_COMPACT[@]}"; do
        for pv in "${LASTFM_PRESERVE[@]}"; do
          for pull in "${LASTFM_PULL[@]}"; do
            for rr in "${LASTFM_RET_RECON[@]}"; do
              for rc in "${LASTFM_RET_CL[@]}"; do
                for k in "${LASTFM_GMM_K[@]}"; do
                  for tau in "${LASTFM_GMM_TAU[@]}"; do
                    for freeze_t in "${LASTFM_FREEZE_TARGET[@]}"; do
                      run_one "LastFMAsia" "${seed}" "${seed}" "${es}" "${cp}" "${pv}" "${pull}" "${rr}" "${rc}" "${k}" "${tau}" "${freeze_t}"
                    done
                  done
                done
              done
            done
          done
        done
      done
    done
  done
}

for ds in "${DATASETS[@]}"; do
  case "${ds}" in
    cora) search_cora ;;
    Cora_ML) search_coraml ;;
    LastFMAsia) search_lastfm ;;
    *) echo "Unknown dataset ${ds}, skip." ;;
  esac
done
