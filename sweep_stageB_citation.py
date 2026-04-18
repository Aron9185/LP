import subprocess
import os

from experiment_paths import sweep_log_dir

# --- Configurations ---
DATASETS = ["cora", "citeseer", "Cora_ML"]
ANCHORS = {
    "cora":     {"pull": 1.0, "add": 0.5},
    "citeseer": {"pull": 1.0, "add": 0.02},
    "Cora_ML":  {"pull": 1.0, "add": 0.5}
}
REMOVES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
SEEDS   = [0, 1, 2, 3, 4]
MAX_WORKERS = 4
PYTHON_EXE = "/home/retro/anaconda3/envs/pyg/bin/python3"

log_dir = sweep_log_dir()

def run_cmd(args):
    cmd, out_path = args
    with open(out_path, "w") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    return out_path

commands = []
for ds in DATASETS:
    p = ANCHORS[ds]["pull"]
    a = ANCHORS[ds]["add"]
    for r in REMOVES:
        for s in SEEDS:
            out_file = log_dir / f"finalB_{ds}_s{s}_p{p}_a{a}_r{r}.txt"
            if os.path.exists(out_file): continue
            cmd = [PYTHON_EXE, "src/aron_main.py", "--dataset", ds, "--seed", str(s), "--editor_pull_strength", str(p), "--decoded_add_ratio", str(a), "--decoded_remove_ratio", str(r), "--decoded_graph_aug_bound", "-1.0"]
            commands.append((cmd, out_file))

if commands:
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for _ in executor.map(run_cmd, commands): pass

print("--- STAGE B CITATION SWEEP COMPLETE ---")
