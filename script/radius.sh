#!/usr/bin/env bash
set -euo pipefail

# ===================== knobs you might tweak =====================
PRETRAIN_EPOCHS=100

# Global edit budget (× |E0|). In remove-only, this caps total REMOVALS.
AUG_RATIO=0.50
# Per-node cumulative cap relative to original degree
AUG_BOUND=0.50

# Datasets
DATASETS=("cora" "Cora_ML" "citeseer" "LastFMAsia")

# seeds
SEEDS=({0..29})

# Cluster method for remove-only (used only to get cluster labels)
CLUSTER_METHODS=("gmm")
GMM_K=32
GMM_TAU=0.60

# Versions to run (remove-only, with scopes). You can comment/uncomment.
BASE_VERSIONS=(
  "remove_only_intra_c0p"
  "remove_only_intra_noncore"
  #"remove_only_intra_cp"
  #"remove_only_inter"
  # "remove_only_both"
)

# Per-node fixed drop fraction hints; empty means auto (from quota)
# Examples: "" "0.02" "0.05" "0.10"
FRAC_HINTS=("")

# Per-dataset degree_threshold (incl self-loop)
declare -A DEG_THR
DEG_THR["cora"]=0.9
DEG_THR["citeseer"]=0.9
DEG_THR["Cora_ML"]=0.5
DEG_THR["LastFMAsia"]=0.8

# Other train args
TOPK_PER_NODE=64
EPOCHS=700

# Logging
DATESTR=$(date +%m%d)
LOGROOT="logs/${DATESTR}"
mkdir -p "${LOGROOT}" runs

# Where radius CSVs are written by the code (keep in sync with train code)
RADIUS_DIR="logs/radius"
SUMMARY_TSV="${LOGROOT}/remove_only_summary.tsv"
# ================================================================

fmt2() { printf '%.2f' "$1"; }

# Write header for summary file (once)
if [[ ! -f "${SUMMARY_TSV}" ]]; then
  echo -e "dataset\tver\tseed\tidx\tend_radius\tdelta_radius\ttotal_removed\tlast_added\tlast_removed\tlog_path\tradius_csv" > "${SUMMARY_TSV}"
fi

# Run one experiment
run_one () {
  local ver_no_frac="$1"     # e.g., remove_only_intra_c0p
  local frac_hint="$2"       # "" or "0.05"
  local cm="$3"              # gmm / louvain / hdbscan (we use gmm here)
  local ds="$4"
  local seed="$5"
  local idx="$6"

  local degthr="${DEG_THR[$ds]}"

  # Attach _fracX to ver if requested
  local ver="${ver_no_frac}"
  if [[ -n "${frac_hint}" ]]; then
    ver="${ver}_frac${frac_hint}"
  fi

  local scores="runs/${ds}_scores${idx}.pt"   # optional; not used by remove-only path
  local ckpt="runs/${ds}_pre${idx}.pt"        # optional; not used by remove-only path

  local r_str;   r_str=$(fmt2 "${AUG_RATIO}")
  local fmr_str; fmr_str=$(fmt2 "${AUG_BOUND}")
  local d_str;   d_str=$(fmt2 "${degthr}")

  local RUNID
  RUNID="$(date +%H%M%S)_$$_${RANDOM}"
  local logfile="${LOGROOT}/${ds}_${ver}_r${r_str}_fmr${fmr_str}_d${d_str}_${cm}_seed${seed}_idx${idx}_ts${RUNID}.log"

  {
    echo "===== RUN META ====="
    echo "date=${DATESTR} time=$(date +%F' '%T) runid=${RUNID}"
    echo "dataset=${ds} ver=${ver} method=${cm}"
    echo "seed=${seed} idx=${idx}"
    echo "aug_ratio=${AUG_RATIO} aug_bound=${AUG_BOUND} degree_thr=${degthr}"
    echo "topk_per_node=${TOPK_PER_NODE}"
    echo "gmm_k=${GMM_K} gmm_tau=${GMM_TAU}"
    echo "===================="
  } | tee -a "${logfile}"

  echo ">>> RUN ver=${ver} cm=${cm} ds=${ds} seed=${seed} idx=${idx} thr=${degthr} | log=${logfile}"

  args=(
    python src/aron_main.py
    --ver "${ver}"
    --dataset "${ds}"
    --seed "${seed}"
    --idx  "${idx}"
    --epochs "${EPOCHS}"
    --pretrain_epochs "${PRETRAIN_EPOCHS}"
    --frozen_scores "${scores}"           # harmless if not present for remove-only
    --pretrained_ckpt "${ckpt}"           # harmless if not present for remove-only
    --aug_ratio "${AUG_RATIO}"
    --aug_bound "${AUG_BOUND}"
    --degree_threshold "${degthr}"
    --cluster_method "${cm}"
    --gmm_k "${GMM_K}"
    --gmm_tau "${GMM_TAU}"
    --topk_per_node "${TOPK_PER_NODE}"
    --logging
    --date "${DATESTR}"
  )

  echo "[CMD] ${args[*]}" | tee -a "${logfile}"
  "${args[@]}" 2>&1 | tee -a "${logfile}"

  # ----------------- post-run: compactness vs edits row -----------------
  set +e
  local radius_csv="${RADIUS_DIR}/${ds}_${ver}_seed${seed}_idx${idx}.csv"
  local end_radius="NA"
  local start_radius="NA"
  local delta_radius="NA"
  local total_removed="NA"
  local last_added="NA"
  local last_removed="NA"

  if [[ -f "${radius_csv}" ]]; then
    # guard against 1-line CSV (header only)
    local nrows
    nrows=$(awk 'END{print NR}' "${radius_csv}")
    if (( nrows > 1 )); then
      # CSV columns: epoch,radius_mean,added,removed,mod_ratio
      start_radius=$(awk -F',' 'NR==2{print $2}' "${radius_csv}" || true)
      end_radius=$(awk -F',' 'NR>1{val=$2} END{if (val=="") print "NA"; else print val}' "${radius_csv}" || true)

      # totals & last epoch stats (robust to empty/missing)
      total_removed=$(awk -F',' 'NR>1{S+=$4} END{if (S=="") S=0; printf "%d", S}' "${radius_csv}" || true)
      last_added=$(awk -F',' 'NR>1{val=$3} END{if (val=="") val=0; print val}' "${radius_csv}" || true)
      last_removed=$(awk -F',' 'NR>1{val=$4} END{if (val=="") val=0; print val}' "${radius_csv}" || true)

      # compute delta only when both ends exist & are numeric
      if [[ -n "${start_radius}" && -n "${end_radius}" && "${start_radius}" != "NA" && "${end_radius}" != "NA" ]]; then
        delta_radius=$(awk -v a="${end_radius}" -v b="${start_radius}" 'BEGIN{printf "%+.6f", a-b}')
      else
        delta_radius="NA"
      fi
    else
      echo "[POST] radius CSV has no data rows: ${radius_csv}" | tee -a "${logfile}"
    fi
  else
    echo "[POST] radius CSV not found: ${radius_csv}" | tee -a "${logfile}"
  fi

  # Append one summary row (tab-separated)
  echo -e "${ds}\t${ver}\t${seed}\t${idx}\t${end_radius}\t${delta_radius}\t${total_removed}\t${last_added}\t${last_removed}\t${logfile}\t${radius_csv}" >> "${SUMMARY_TSV}"
  echo "[POST] summary row appended to ${SUMMARY_TSV}"
  set -e
}  # <-- CLOSES run_one()

# =============================== sweep ===============================
for cm in "${CLUSTER_METHODS[@]}"; do
  for ds in "${DATASETS[@]}"; do
    for seed in "${SEEDS[@]}"; do
      idx="${seed}"   # keep idx aligned with seed for easy matching
      for base_ver in "${BASE_VERSIONS[@]}"; do
        for fh in "${FRAC_HINTS[@]}"; do
          run_one "${base_ver}" "${fh}" "${cm}" "${ds}" "${seed}" "${idx}"
        done
      done
    done
  done
done

echo "All runs done."
echo "Compactness–edits summary saved to: ${SUMMARY_TSV}"
echo "Per-run radius series saved under: ${RADIUS_DIR}/"
