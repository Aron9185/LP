"""
Rebuild stageA_results.csv by merging:
  - Existing results for cora, citeseer, Cora_ML  (already good)
  - Fresh LastFMAsia results from stageA_LastFMAsia_results.csv

Run this AFTER rerun_stageA_LastFMAsia.py completes.
"""

import pandas as pd
import sys
import os

old_csv      = "stageA_results.csv"
lastfm_csv   = "stageA_LastFMAsia_results.csv"
out_csv      = "stageA_results.csv"

if not os.path.exists(lastfm_csv):
    print(f"ERROR: {lastfm_csv} not found. Run rerun_stageA_LastFMAsia.py first.")
    sys.exit(1)

old = pd.read_csv(old_csv)
old_no_lastfm = old[old["dataset"] != "LastFMAsia"]

new_lastfm = pd.read_csv(lastfm_csv)
print(f"Old non-LastFMAsia rows : {len(old_no_lastfm)}")
print(f"New LastFMAsia rows     : {len(new_lastfm)}")

merged = pd.concat([old_no_lastfm, new_lastfm], ignore_index=True)
merged.to_csv(out_csv, index=False)
print(f"\nRebuilt {out_csv}  ({len(merged)} rows total)")

print("\n=== Summary (mean over seeds) ===")
summary = (
    merged.groupby(["dataset", "pull_strength", "add_ratio"])
          [["test_hit10", "val_roc", "radius_after"]]
          .mean()
          .round(4)
)
print(summary.to_string())
