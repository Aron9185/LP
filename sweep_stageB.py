"""
Stage B: Removal study — local refinement around best add-only settings.

For each dataset, pick the top-2 pull_strengths and top-2 add_ratios from
stageA_results.csv, then sweep:
  decoded_remove_ratio = [0.0, 0.002, 0.005, 0.01]

Everything else is identical to Stage A (including aug_bound = -1).

Run this script AFTER sweep_stageA.py has produced stageA_results.csv.
"""

import os
import re
import subprocess
import itertools
import concurrent.futures
import pandas as pd

from experiment_paths import artifact_path, sweep_log_dir

REMOVE_RATIOS = [0.0, 0.002, 0.005, 0.01]
AUG_BOUND    = -1.0
SEEDS        = [0, 1, 2]
EPOCHS       = 700
MAX_WORKERS  = 2
TOP_K        = 2   # top-2 pulls and top-2 add_ratios per dataset
LOG_DIR = sweep_log_dir()
STAGE_A_RESULTS_CSV = artifact_path("stageA_results.csv")
STAGE_B_RESULTS_CSV = artifact_path("stageB_results.csv")

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
    "--decoded_graph_aug_bound",  str(AUG_BOUND),
    "--ver",                      "no",
]

def log_path(dataset, seed, pull, add, remove):
    tag = f"{dataset}_s{seed}_p{pull}_r{add}_rm{remove}"
    return LOG_DIR / f"stageB_{tag}.txt"

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
    hits  = re.findall(r'test_hit10=([0-9.]+)', content)
    add_m = re.search(r'\[EDIT-GRAPH\].*? add=(\d+)', content)
    rm_m  = re.search(r'\[EDIT-GRAPH\].*? remove=(\d+)', content)
    return {
        'val_roc':        float(m.group(2))     if m    else float('nan'),
        'radius_after':   float(m.group(4))     if m    else float('nan'),
        'test_hit10':     float(hits[-1])        if hits else float('nan'),
        'added_edges':    int(add_m.group(1))    if add_m else 0,
        'removed_edges':  int(rm_m.group(1))     if rm_m else 0,
    }

def run(args):
    dataset, seed, pull, add, remove = args
    out = log_path(dataset, seed, pull, add, remove)

    if os.path.exists(out) and os.path.getsize(out) > 1000:
        print(f"  [skip] {out.name}")
        metrics = extract_metrics(out)
    else:
        cmd = BASE_ARGS + [
            "--dataset",              dataset,
            "--seed",                 str(seed),
            "--editor_pull_strength", str(pull),
            "--decoded_add_ratio",    str(add),
            "--decoded_remove_ratio", str(remove),
        ]
        bash = (
            "source /home/retro/anaconda3/etc/profile.d/conda.sh && "
            "conda activate pyg && " + " ".join(cmd)
        )
        print(f"  [run ] {out.name}")
        with open(out, "w") as f:
            subprocess.run(["bash", "-c", bash], stdout=f, stderr=subprocess.STDOUT)
        metrics = extract_metrics(out)

    metrics.update({"dataset": dataset, "seed": seed,
                    "pull_strength": pull, "add_ratio": add,
                    "remove_ratio": remove})
    print(f"  [done] {dataset} s{seed} p{pull} r{add} rm{remove} "
          f"-> hit10={metrics.get('test_hit10'):.4f}  "
          f"roc={metrics.get('val_roc'):.4f}  "
          f"add={metrics.get('added_edges')}  "
          f"rm={metrics.get('removed_edges')}")
    return metrics

def pick_top_settings(csv_path, dataset, top_k=2):
    """Return top-k pull strengths and add_ratios for a dataset from Stage A."""
    df = pd.read_csv(csv_path)
    df = df[df["dataset"] == dataset]
    agg = df.groupby(["pull_strength", "add_ratio"])["test_hit10"].mean()
    # top-k unique pull strengths
    top_pulls = (
        agg.groupby(level="pull_strength").max()
           .nlargest(top_k).index.tolist()
    )
    # top-k unique add_ratios
    top_adds = (
        agg.groupby(level="add_ratio").max()
           .nlargest(top_k).index.tolist()
    )
    return top_pulls, top_adds

if __name__ == "__main__":
    import sys
    stage_a_csv = STAGE_A_RESULTS_CSV
    if not os.path.exists(stage_a_csv):
        print(f"ERROR: {stage_a_csv} not found. Run sweep_stageA.py first.")
        sys.exit(1)

    datasets = ["cora", "citeseer", "Cora_ML", "LastFMAsia"]
    grid = []
    for ds in datasets:
        pulls, adds = pick_top_settings(stage_a_csv, ds, top_k=TOP_K)
        print(f"{ds}: top_pulls={pulls}  top_adds={adds}")
        for pull, add, remove, seed in itertools.product(pulls, adds, REMOVE_RATIOS, SEEDS):
            grid.append((ds, seed, pull, add, remove))

    print(f"\nStage B: {len(grid)} runs  (workers={MAX_WORKERS})\n")

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for res in ex.map(run, grid):
            results.append(res)

    df = pd.DataFrame(results)
    df.to_csv(STAGE_B_RESULTS_CSV, index=False)

    print("\n=== Stage B Summary (mean over seeds) ===")
    summary = (
        df.groupby(["dataset", "pull_strength", "add_ratio", "remove_ratio"])
          [["added_edges", "removed_edges", "radius_after", "val_roc", "test_hit10"]]
          .mean()
          .round(4)
    )
    print(summary.to_string())
    print(f"\nFull results saved to {STAGE_B_RESULTS_CSV}")
