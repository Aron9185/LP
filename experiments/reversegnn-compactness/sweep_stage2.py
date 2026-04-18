import os
import subprocess
import itertools
import pandas as pd
import re

from experiment_paths import artifact_path, repo_root, sweep_log_dir

SEEDS = [0, 1, 2]
PULL_STRENGTHS = [0.0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0]

# Adjust this if full 700 epochs is too slow; typically pilot runs can use fewer epochs, but we use default 700 unless changed.
EPOCHS = 700 
DATASET = "cora"
REPO_ROOT = repo_root()
LOG_DIR = sweep_log_dir()
RESULTS_CSV = artifact_path("stage2_pull_sweep_results.csv")

base_cmd = [
    "python", "src/aron_main.py",
    "--dataset", DATASET,
    "--epochs", str(EPOCHS),
    "--use_edited_decoder",
    "--use_decoded_graph_augment",
    "--pull_mask_scope", "cp",
    "--compactness_mask_scope", "cp",
    "--rewrite_endpoint_scope", "c0p",
    "--decoded_same_cluster_only",
    "--decoded_require_c0p_endpoint",
    "--decoded_temporary_view_only",
    "--freeze_c0p_at_edit_start",
    "--decoded_add_ratio", "0.01",
    "--decoded_graph_aug_bound", "0.1",
    "--ver", "no"
]

results = []

def extract_metric(filepath):
    """Parses output files for final essential metrics."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    # We want Hit@1, Hit@3, Hit@10 out of the history, or from the final evaluation block.
    # We look for [SANITY SUMMARY] or similar to get the best validation run metrics
    # e.g., [SANITY SUMMARY] best_val_epoch=43 val_roc=0.896587 radius_before=0.660343 radius_after=0.659227 delta=-0.001116 ... edit_recon=0.618503 edit_compact=0.903558
    sanity_summary = re.search(r'\[SANITY SUMMARY\] best_val_epoch=(\d+) val_roc=([0-9.]+) radius_before=([0-9.-]+) radius_after=([0-9.-]+).*?edit_compact=([0-9.-]+)', content)
    
    # Also want final test hit10
    # Usually logged as: test_hit10=0.1234
    test_hit_matches = re.findall(r'test_hit10=([0-9.]+)', content)
    test_hit10 = test_hit_matches[-1] if test_hit_matches else None
    
    # We'll just grab the final radius reports
    final_radius_core = re.search(r'\[FINAL-RADIUS\] core.*?mean=([0-9.]+).*?max=([0-9.]+)', content)
    
    # Grab rewrite applied count or add count
    add_match = re.search(r'\[EDIT-GRAPH\]\[.*?\] .*?add=(\d+)', content)
    added_edges = add_match.group(1) if add_match else 0
    
    ret = {}
    if sanity_summary:
        ret['best_val_epoch'] = sanity_summary.group(1)
        ret['val_roc'] = sanity_summary.group(2)
        ret['radius_before'] = sanity_summary.group(3)
        ret['radius_after'] = sanity_summary.group(4)
        
    if test_hit10:
        ret['test_hit10'] = test_hit10
        
    if final_radius_core:
        ret['final_core_radius_mean'] = final_radius_core.group(1)
        
    ret['added_edges'] = added_edges
    return ret

import concurrent.futures

def run_experiment(arg_tuple):
    seed, pull = arg_tuple
    print(f"Running Seed {seed}, Pull {pull}...")
    log_file = LOG_DIR / f"cora_s{seed}_p{pull}.txt"
    
    cmd = base_cmd + ["--seed", str(seed), "--editor_pull_strength", str(pull)]
    bash_str = f'cd "{REPO_ROOT}" && source /home/retro/anaconda3/etc/profile.d/conda.sh && conda activate pyg && ' + " ".join(cmd)
    
    with open(log_file, "w") as out:
        subprocess.run(["bash", "-c", bash_str], stdout=out, stderr=subprocess.STDOUT)
        
    metrics = extract_metric(log_file)
    metrics['seed'] = seed
    metrics['pull_strength'] = pull
    
    print(f"  Finished S{seed} P{pull} -> Val ROC: {metrics.get('val_roc')}, Test Hit@10: {metrics.get('test_hit10')}, CP Radius After: {metrics.get('radius_after')}")
    return metrics

if __name__ == "__main__":
    print("Starting Stage 2 Sweep...")
    
    args = list(itertools.product(SEEDS, PULL_STRENGTHS))
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        for res in executor.map(run_experiment, args):
            results.append(res)

    df = pd.DataFrame(results)
    df.to_csv(RESULTS_CSV, index=False)
    print(f"\nSweep Complete! Results saved to {RESULTS_CSV}")
    print(df.groupby('pull_strength')[['test_hit10', 'val_roc', 'radius_after']].mean())
