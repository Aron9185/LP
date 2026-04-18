"""
Emergency Noon Deadline Sweep (Extreme Focus)
Grid:
  editor_pull_strength = [0.0, 0.4, 1.0]
  compactness_weight   = 0.2
  add_ratio            = [0.1, 0.5, 1.0]
Seeds: 5
Estimated completion: 3.5 hours (approx 11:45 AM)
"""
import os
import subprocess
import itertools
import concurrent.futures

from experiment_paths import sweep_log_dir

DATASETS_STD  = ["cora", "citeseer", "Cora_ML"]
PULLS         = [0.0, 0.4, 1.0]  # Reduced for speed
ADDS          = [0.1, 0.5, 1.0]  # Large ratios focus
EPOCHS        = 700
COMPACT_WEIGHT = 0.2
LOG_DIR = sweep_log_dir()

def run_job(ds, pull, add, seed):
    tag = f"finalA_{ds}_s{seed}_p{pull}_r{add}"
    logfile = LOG_DIR / f"{tag}.txt"
    
    if os.path.exists(logfile) and os.path.getsize(logfile) > 1000:
        return
    
    cmd = [
        "python", "src/aron_main.py",
        "--dataset", ds,
        "--seed", str(seed),
        "--epochs", str(EPOCHS),
        "--editor_pull_strength", str(pull),
        "--compactness_weight", str(COMPACT_WEIGHT),
        "--decoded_add_ratio", str(add),
        "--use_edited_decoder",
        "--use_decoded_graph_augment",
        "--pull_mask_scope", "cp",
        "--compactness_mask_scope", "cp",
        "--rewrite_endpoint_scope", "c0p",
        "--decoded_same_cluster_only",
        "--decoded_require_c0p_endpoint",
        "--decoded_temporary_view_only",
        "--freeze_c0p_at_edit_start",
        "--ver", "no",
        "--sweep_mode"
    ]
    
    bash_cmd = (
        "source /home/retro/anaconda3/etc/profile.d/conda.sh && "
        "conda activate pyg && " +
        " ".join(cmd)
    )
    
    with open(logfile, "w") as f:
        subprocess.run(["bash", "-c", bash_cmd], stdout=f, stderr=subprocess.STDOUT)
    print(f"  [done] {tag}")

if __name__ == "__main__":
    # Standard Datasets (5 seeds)
    std_grid = list(itertools.product(DATASETS_STD, PULLS, ADDS, range(5)))
    print(f"Launching Emergency Deadline Grid: {len(std_grid)} jobs (4 threads)...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(run_job, ds, p, a, s) for ds, p, a, s in std_grid]
        
    print("\n--- EMERGENCY SWEEP COMPLETE ---")
