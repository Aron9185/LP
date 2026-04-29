"""
Rerun Stage A for LastFMAsia only, with MAX_WORKERS=1.

Root cause: CUDA OOM when 4 processes compete for 12 GB GPU.
Fix: run each LastFMAsia job serially so it gets the full GPU.

After this completes, regenerate stageA_results.csv by running:
  python rebuild_stageA_csv.py
"""

import os
import re
import subprocess
import itertools
import concurrent.futures
import pandas as pd

from experiment_paths import artifact_path, repo_root, sweep_log_dir

DATASET      = "LastFMAsia"
PULL_STRENGTHS = [0.0, 0.1, 0.2, 0.4, 0.6]
ADD_RATIOS   = [0.005, 0.01, 0.02, 0.05]
REMOVE_RATIO = 0.0
AUG_BOUND    = -1.0
SEEDS        = [0, 1, 2]
EPOCHS       = 700
MAX_WORKERS  = 1   # serial — full GPU per run to avoid OOM
REPO_ROOT = repo_root()
LOG_DIR = sweep_log_dir()
RESULTS_CSV = artifact_path("stageA_LastFMAsia_results.csv")

BASE_ARGS = [
    "python", "src/aron_main.py",
    "--epochs",                   str(EPOCHS),
    "--use_edited_decoder",
    "--decoder_objective",        "recon",
    "--compactness_objective",    "radius",
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

def log_path(seed, pull, add):
    tag = f"{DATASET}_s{seed}_p{pull}_r{add}"
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
    hits  = re.findall(r'test_hit10=([0-9.]+)', content)
    add_m = re.search(r'\[EDIT-GRAPH\].*? add=(\d+)', content)
    return {
        'val_roc':      float(m.group(2))   if m    else float('nan'),
        'radius_after': float(m.group(4))   if m    else float('nan'),
        'test_hit10':   float(hits[-1])     if hits else float('nan'),
        'added_edges':  int(add_m.group(1)) if add_m else 0,
    }

def run(args):
    seed, pull, add = args
    out = log_path(seed, pull, add)

    # Force rerun: skip only if log contains a successful SANITY SUMMARY
    if os.path.exists(out):
        content = open(out).read()
        if '[SANITY SUMMARY]' in content:
            print(f"  [skip] {out.name}  (already succeeded)")
            metrics = extract_metrics(out)
            metrics.update({"dataset": DATASET, "seed": seed,
                            "pull_strength": pull, "add_ratio": add})
            return metrics

    cmd = BASE_ARGS + [
        "--dataset",              DATASET,
        "--seed",                 str(seed),
        "--editor_pull_strength", str(pull),
        "--decoded_add_ratio",    str(add),
    ]
    bash = (
        f'cd "{REPO_ROOT}" && '
        "source /home/retro/anaconda3/etc/profile.d/conda.sh && "
        "conda activate pyg && " + " ".join(cmd)
    )
    print(f"  [run ] {out.name}")
    with open(out, "w") as f:
        subprocess.run(["bash", "-c", bash], stdout=f, stderr=subprocess.STDOUT)
    metrics = extract_metrics(out)

    hit  = metrics.get('test_hit10', float('nan'))
    roc  = metrics.get('val_roc',    float('nan'))
    rad  = metrics.get('radius_after', float('nan'))
    add_ = metrics.get('added_edges', 0)
    status = "OK" if not (hit != hit) else "FAIL"  # nan check
    print(f"  [{status}] {DATASET} s{seed} p{pull} r{add} "
          f"-> hit10={hit:.4f}  roc={roc:.4f}  rad={rad:.4f}  add={add_}")
    metrics.update({"dataset": DATASET, "seed": seed,
                    "pull_strength": pull, "add_ratio": add})
    return metrics

if __name__ == "__main__":
    grid = list(itertools.product(SEEDS, PULL_STRENGTHS, ADD_RATIOS))
    total = len(grid)
    print(f"LastFMAsia Stage A rerun: {total} runs  (MAX_WORKERS={MAX_WORKERS})\n")

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        for i, res in enumerate(ex.map(run, grid), 1):
            results.append(res)
            print(f"  Progress: {i}/{total}")

    df = pd.DataFrame(results)
    df.to_csv(RESULTS_CSV, index=False)

    print("\n=== LastFMAsia Stage A Summary (mean over seeds) ===")
    summary = (
        df.groupby(["pull_strength", "add_ratio"])
          [["added_edges", "radius_after", "val_roc", "test_hit10"]]
          .mean()
          .round(4)
    )
    print(summary.to_string())
    print(f"\nSaved to {RESULTS_CSV}")
    print("Now run: python rebuild_stageA_csv.py  to merge into stageA_results.csv")
