import subprocess
import os
import re

from experiment_paths import sweep_log_dir

# --- STAGE A GRID ---
PULLS = [0.0, 0.4, 1.0]
ADDS  = [0.01, 0.02, 0.1, 0.5, 1.0]
SEEDS = [0, 1, 2, 3, 4]
MAX_WORKERS = 4
PYTHON_EXE = "/home/retro/anaconda3/envs/pyg/bin/python3"
log_dir = sweep_log_dir()

def run_cmd(args):
    cmd, out_path = args
    with open(out_path, "w") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    return out_path

# 1. Run COMPLETE Addition Sweep for LastFMAsia
from concurrent.futures import ProcessPoolExecutor

cmds_a = []
for p in PULLS:
    for a in ADDS:
        for s in SEEDS:
            out_file = log_dir / f"finalA_LastFMAsia_s{s}_p{p}_r{a}.txt"
            if os.path.exists(out_file): continue
            cmd = [PYTHON_EXE, "src/aron_main.py", "--dataset", "LastFMAsia", "--seed", str(s), "--editor_pull_strength", str(p), "--decoded_add_ratio", str(a)]
            cmds_a.append((cmd, out_file))

if cmds_a:
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for _ in executor.map(run_cmd, cmds_a): pass

# 2. Identify Winner
records = []
for s in SEEDS:
    for p in PULLS:
        for a in ADDS:
            log_path = log_dir / f"finalA_LastFMAsia_s{s}_p{p}_r{a}.txt"
            try:
                with open(log_path, 'r') as f:
                    content = f.read()
                    if "SANITY SUMMARY" in content:
                        h10_match = re.search(r'\[FINAL TEST\] Hit@K: 1=[0-9.]+, 3=[0-9.]+, 10=([0-9.]+)', content)
                        if h10_match:
                            h10 = float(h10_match.group(1))
                            records.append({'p': p, 'r': a, 'h10': h10})
            except: pass

import pandas as pd
if not records:
    print("WARNING: No Stage A results for LastFMAsia yet.")
    exit(1)

df = pd.DataFrame(records)
best = df.groupby(['p', 'r']).agg({'h10': 'mean'}).reset_index().sort_values('h10', ascending=False).iloc[0]
best_p, best_a = best['p'], best['r']

# 3. Run Stage B (Removal)
REMOVES = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
cmds_b = []
for r in REMOVES:
    for s in SEEDS:
        out_file = log_dir / f"finalB_LastFMAsia_s{s}_p{best_p}_a{best_a}_r{r}.txt"
        if os.path.exists(out_file): continue
        cmd = [PYTHON_EXE, "src/aron_main.py", "--dataset", "LastFMAsia", "--seed", str(s), "--editor_pull_strength", str(best_p), "--decoded_add_ratio", str(best_a), "--decoded_remove_ratio", str(r)]
        cmds_b.append((cmd, out_file))

if cmds_b:
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for _ in executor.map(run_cmd, cmds_b): pass

print("--- LASTFMASIA FULL BENCHMARK COMPLETE ---")
