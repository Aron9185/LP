import os
import subprocess
import itertools
import pandas as pd
import re
import concurrent.futures

from experiment_paths import artifact_path, repo_root, sweep_log_dir

SEEDS = [0, 1, 2]
AUG_BOUNDS = [0.05, 0.1, -1.0]

# Best from Stage 3A
BEST_ADD_RATIO = 0.01
PULL_STRENGTH = 0.2
EPOCHS = 700 
DATASET = "cora"
REPO_ROOT = repo_root()
LOG_DIR = sweep_log_dir()
RESULTS_CSV = artifact_path("stage3b_augbound_sweep.csv")

base_cmd = [
    "python", "src/aron_main.py",
    "--dataset", DATASET,
    "--epochs", str(EPOCHS),
    "--use_edited_decoder",
    "--decoder_objective", "recon",
    "--compactness_objective", "radius",
    "--use_decoded_graph_augment",
    "--pull_mask_scope", "cp",
    "--compactness_mask_scope", "cp",
    "--rewrite_endpoint_scope", "c0p",
    "--decoded_same_cluster_only",
    "--decoded_require_c0p_endpoint",
    "--decoded_temporary_view_only",
    "--freeze_c0p_at_edit_start",
    "--decoded_add_ratio", str(BEST_ADD_RATIO),
    "--editor_pull_strength", str(PULL_STRENGTH),
    "--ver", "no"
]

def extract_metric(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
    
    sanity_summary = re.search(r'\[SANITY SUMMARY\] best_val_epoch=(\d+) val_roc=([0-9.]+) radius_before=([0-9.-]+) radius_after=([0-9.-]+).*?edit_compact=([0-9.-]+)', content)
    test_hit_matches = re.findall(r'test_hit10=([0-9.]+)', content)
    test_hit10 = test_hit_matches[-1] if test_hit_matches else float('nan')
    
    add_match = re.search(r'\[EDIT-GRAPH\]\[.*?\] .*?add=(\d+)', content)
    added_edges = float(add_match.group(1)) if add_match else 0.0
    
    ret = {}
    if sanity_summary:
        ret['val_roc'] = float(sanity_summary.group(2))
        ret['radius_after'] = float(sanity_summary.group(4))
    else:
        ret['val_roc'] = float('nan')
        ret['radius_after'] = float('nan')
        
    ret['test_hit10'] = float(test_hit10)
    ret['added_edges'] = added_edges
    return ret

def run_experiment(arg_tuple):
    seed, bound = arg_tuple
    print(f"Running Seed {seed}, Aug Bound {bound}...")
    log_file = LOG_DIR / f"cora_s{seed}_b{bound}.txt"
    
    cmd = base_cmd + ["--seed", str(seed), "--decoded_graph_aug_bound", str(bound)]
    bash_str = f'cd "{REPO_ROOT}" && source /home/retro/anaconda3/etc/profile.d/conda.sh && conda activate pyg && ' + " ".join(cmd)
    
    with open(log_file, "w") as out:
        subprocess.run(["bash", "-c", bash_str], stdout=out, stderr=subprocess.STDOUT)
        
    metrics = extract_metric(log_file)
    metrics['seed'] = seed
    metrics['aug_bound'] = bound
    
    print(f"  Finished S{seed} Bound {bound} -> Val ROC: {metrics.get('val_roc')}, Test Hit@10: {metrics.get('test_hit10')}, CP Radius: {metrics.get('radius_after')}, Added: {metrics.get('added_edges')}")
    return metrics

if __name__ == "__main__":
    print("Starting Stage 3B Sweep (Aug Bound)...")
    
    args = list(itertools.product(SEEDS, AUG_BOUNDS))
    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        for res in executor.map(run_experiment, args):
            results.append(res)

    df = pd.DataFrame(results)
    df.to_csv(RESULTS_CSV, index=False)
    print(f"\nSweep Complete! Results saved to {RESULTS_CSV}")
    print(df.groupby('aug_bound')[['added_edges', 'radius_after', 'val_roc', 'test_hit10']].mean())
