"""
Stage A: Add-only screen across all four datasets.

Grid:
  pull_strength      = [0.0, 0.1, 0.2, 0.4, 0.6]
  decoded_add_ratio  = [0.005, 0.01, 0.02, 0.05]
  decoded_remove_ratio = 0.0  (fixed)
  decoded_graph_aug_bound = -1 (no per-node cap)
  seeds = [0, 1, 2]
  datasets = cora, citeseer, Cora_ML, LastFMAsia

Total: 5 x 4 x 3 x 4 = 240 runs.

Fixed rewrite policy:
  pull_mask_scope = cp
  compactness_mask_scope = cp
  rewrite_endpoint_scope = c0p
  decoded_same_cluster_only = true
  decoded_require_c0p_endpoint = true
  decoded_temporary_view_only = true
  freeze_c0p_at_edit_start = true
"""

import os
import re
import subprocess
import itertools
import concurrent.futures
import pandas as pd

from experiment_paths import artifact_path, sweep_log_dir

# ── Grid ────────────────────────────────────────────────────────────────────
DATASETS     = ["cora", "citeseer", "Cora_ML", "LastFMAsia"]
PULL_STRENGTHS = [0.0, 0.1, 0.2, 0.4, 0.6]
ADD_RATIOS   = [0.005, 0.01, 0.02, 0.05]
REMOVE_RATIO = 0.0
AUG_BOUND    = -1.0
SEEDS        = [0, 1, 2]
EPOCHS       = 700
MAX_WORKERS  = 4          # concurrent runs; tune to your CPU
LOG_DIR = sweep_log_dir()
RESULTS_CSV = artifact_path("stageA_results.csv")

# ── Fixed base args ──────────────────────────────────────────────────────────
BASE_ARGS = [
    "python", "src/aron_main.py",
    "--epochs",                   str(EPOCHS),
    "--use_edited_decoder",
    "--use_decoded_graph_augment",
    "--pull_mask_scope",          "cp",
    "--compactness_mask_scope",   "cp",
    "--rewrite_endpoint_scope",   "c0p",
    "--decoded_same_cluster_only",
    "--decoded_require_c0p_endpoint",
    "--decoded_temporary_view_only",
    "--freeze_c0p_at_edit_start",
    "--decoded_remove_ratio",     str(REMOVE_RATIO),
    "--decoded_graph_aug_bound",  str(AUG_BOUND),
    "--ver",                      "no",
]

# ── Helpers ──────────────────────────────────────────────────────────────────
def log_path(dataset, seed, pull, add):
    tag = f"{dataset}_s{seed}_p{pull}_r{add}"
    return LOG_DIR / f"stageA_{tag}.txt"

def extract_metrics(path):
    try:
        content = open(path).read()
    except FileNotFoundError:
        return {}

    m = re.search(
        r'\[SANITY SUMMARY\] best_val_epoch=(\d+) val_roc=([0-9.]+) '
        r'radius_before=([0-9.]+) radius_after=([0-9.]+)',
        content
    )
    hits   = re.findall(r'test_hit10=([0-9.]+)', content)
    # added edges: pick the first [EDIT-GRAPH] line that has add=N
    add_m  = re.search(r'\[EDIT-GRAPH\].*? add=(\d+)', content)
    # node concentration: how many unique nodes absorbed edges
    # logged as per-node line – if not present, skip
    total_added  = int(add_m.group(1))  if add_m  else 0

    return {
        'val_roc':      float(m.group(2))    if m    else float('nan'),
        'radius_after': float(m.group(4))    if m    else float('nan'),
        'test_hit10':   float(hits[-1])      if hits else float('nan'),
        'added_edges':  total_added,
    }

# ── Worker ───────────────────────────────────────────────────────────────────
def run(args):
    dataset, seed, pull, add = args
    out = log_path(dataset, seed, pull, add)

    # skip if already done
    if os.path.exists(out) and os.path.getsize(out) > 1000:
        print(f"  [skip] {out.name}")
        metrics = extract_metrics(out)
    else:
        cmd = BASE_ARGS + [
            "--dataset",              dataset,
            "--seed",                 str(seed),
            "--editor_pull_strength", str(pull),
            "--decoded_add_ratio",    str(add),
        ]
        bash = (
            "source /home/retro/anaconda3/etc/profile.d/conda.sh && "
            "conda activate pyg && " +
            " ".join(cmd)
        )
        print(f"  [run ] {out.name}")
        with open(out, "w") as f:
            subprocess.run(["bash", "-c", bash], stdout=f, stderr=subprocess.STDOUT)
        metrics = extract_metrics(out)

    metrics.update({"dataset": dataset, "seed": seed,
                    "pull_strength": pull, "add_ratio": add})
    print(f"  [done] {dataset} s{seed} p{pull} r{add} "
          f"-> hit10={metrics.get('test_hit10'):.4f}  "
          f"roc={metrics.get('val_roc'):.4f}  "
          f"rad={metrics.get('radius_after'):.4f}  "
          f"add={metrics.get('added_edges')}")
    return metrics

# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    grid = list(itertools.product(DATASETS, SEEDS, PULL_STRENGTHS, ADD_RATIOS))
    print(f"Stage A: {len(grid)} runs  (workers={MAX_WORKERS})\n")

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for res in ex.map(run, grid):
            results.append(res)

    df = pd.DataFrame(results)
    df.to_csv(RESULTS_CSV, index=False)

    print("\n=== Stage A Summary (mean over seeds) ===")
    summary = (
        df.groupby(["dataset", "pull_strength", "add_ratio"])
          [["added_edges", "radius_after", "val_roc", "test_hit10"]]
          .mean()
          .round(4)
    )
    print(summary.to_string())
    print(f"\nFull results saved to {RESULTS_CSV}")
