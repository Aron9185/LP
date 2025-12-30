#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# TSNE sweep runner for MULTIPLE CACHE_ROOTS (hard-coded list)
# - Each CACHE_ROOT can be:
#   A) parent dir containing many preprune_* subdirs
#   B) a single preprune_* dir
# - For each root, run the same TSNE sweeps and write outputs
#   under OUT_ROOT/<root_tag>/...
# ============================================================

# --------- EDIT THESE SETTINGS ---------

# List your cache roots here (as many as you want)
# Examples:
#  "/home/retro/ARON/artifacts/tsne_cache/cora/aron_desc/seed0/c0p_only"
#  "/home/retro/ARON/artifacts/tsne_cache/cora/aron_desc/seed0/cp_all/preprune_0.30"
CACHE_ROOTS=(
  "/home/retro/ARON/artifacts/tsne_cache/cora/aron_desc/seed0/preprune_0.00"
  "/home/retro/ARON/artifacts/tsne_cache/cora/aron_desc/seed0/preprune_0.30"
  "/home/retro/ARON/artifacts/tsne_cache/cora/aron_desc/seed0/preprune_0.60"
  "/home/retro/ARON/artifacts/tsne_cache/cora/aron_desc/seed0/preprune_0.90"
)

# Output root folder
OUT_ROOT="tsne_sweep_outputs/multi_roots"

# Embeddings to plot (if exists)
PREFIXES=("Z0" "final")


# TSNE sweeps (seed fixed to 0 as you requested)
PERPLEXITIES=(5 10 20 30 40 60 80 100 150 200)
LRS=(100 200 400)
SEED=0

# runtime knobs
N_ITER=1500
MAX_POINTS=4000   # <=0 means use all points (your python script supports this)

PLOT_SCRIPT="/home/retro/ARON/src/plot_tsne_from_cache.py"
# --------------------------------------


need_file() { [[ -f "$1" ]]; }

# build a safe tag name from a path (for output folder)
tagify() {
  local p="$1"
  # turn /a/b/c into a_b_c, remove weird chars
  echo "$p" | sed 's#^/##' | sed 's#/#_#g' | sed 's#[^A-Za-z0-9_.-]#_#g'
}

echo "[INFO] OUT_ROOT=$OUT_ROOT"
echo "[INFO] PREFIXES=${PREFIXES[*]}"
echo "[INFO] PERPLEXITIES=${PERPLEXITIES[*]}"
echo "[INFO] LRS=${LRS[*]}"
echo "[INFO] SEED=$SEED N_ITER=$N_ITER MAX_POINTS=$MAX_POINTS"
echo "[INFO] PLOT_SCRIPT=$PLOT_SCRIPT"
echo

if [[ ! -f "$PLOT_SCRIPT" ]]; then
  echo "[ERROR] plot script not found: $PLOT_SCRIPT"
  exit 1
fi

mkdir -p "$OUT_ROOT"

for CACHE_ROOT in "${CACHE_ROOTS[@]}"; do
  echo "============================================================"
  echo "[ROOT] CACHE_ROOT=$CACHE_ROOT"
  echo "============================================================"

  if [[ ! -d "$CACHE_ROOT" ]]; then
    echo "[SKIP] root not found: $CACHE_ROOT"
    continue
  fi

  root_tag="$(tagify "$CACHE_ROOT")"
  root_out="$OUT_ROOT/$root_tag"
  mkdir -p "$root_out"

  BASE="$(basename "$CACHE_ROOT")"
  PRE_DIRS=()

  if [[ "$BASE" == preprune_* ]]; then
    PRE_DIRS=("$CACHE_ROOT")
    echo "[MODE] single preprune dir"
  else
    mapfile -t PRE_DIRS < <(find "$CACHE_ROOT" -maxdepth 1 -type d -name "preprune_*" | sort)
    if [[ ${#PRE_DIRS[@]} -eq 0 ]]; then
      echo "[SKIP] no preprune_* dirs under: $CACHE_ROOT"
      continue
    fi
    echo "[MODE] scan preprune dirs (n=${#PRE_DIRS[@]})"
  fi

  for d in "${PRE_DIRS[@]}"; do
    frac="$(basename "$d")"   # preprune_0.30

    for prefix in "${PREFIXES[@]}"; do
      zpath="$d/${prefix}_Z.pt"
      if ! need_file "$zpath"; then
        echo "[SKIP] $d missing ${prefix}_Z.pt"
        continue
      fi

      for p in "${PERPLEXITIES[@]}"; do
        for lr in "${LRS[@]}"; do
          out_dir="$root_out/$frac/$prefix"
          mkdir -p "$out_dir"

          out_png="$out_dir/tsne_${prefix}_${frac}_p${p}_lr${lr}_seed${SEED}.png"

          echo "[RUN] root_tag=$root_tag | $frac | prefix=$prefix p=$p lr=$lr seed=$SEED"
          python "$PLOT_SCRIPT" \
            --cache_dir "$d" \
            --prefix "$prefix" \
            --perplexity "$p" \
            --lr "$lr" \
            --n_iter "$N_ITER" \
            --seed "$SEED" \
            --max_points "$MAX_POINTS" \
            --out "$out_png"
        done
      done
    done
  done

  echo "[DONE] root outputs -> $root_out"
done

echo
echo "[ALL DONE] outputs under: $OUT_ROOT"
echo "Summary PNG count:"
echo "  find \"$OUT_ROOT\" -name '*.png' | wc -l"
