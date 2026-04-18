import pandas as pd
import numpy as np
import os

def analyze():
    a_path = "stageA_results.csv"
    b_path = "stageB_results.csv"

    if not os.path.exists(a_path) or not os.path.exists(b_path):
        print("ERROR: Result CSVs not found.")
        return

    df_a = pd.read_csv(a_path)
    df_b = pd.read_csv(b_path)

    # Ensure remove_ratio exists in A (it's 0.0)
    if 'remove_ratio' not in df_a.columns:
        df_a['remove_ratio'] = 0.0
    
    # Merge
    df = pd.concat([df_a, df_b], ignore_index=True)

    print(f"Total merged runs: {len(df)}")

    # 1. Best config per dataset
    agg = df.groupby(["dataset", "pull_strength", "add_ratio", "remove_ratio"])["test_hit10"].mean().reset_index()
    best_idx = agg.groupby("dataset")["test_hit10"].idxmax()
    best_configs = agg.loc[best_idx].sort_values("test_hit10", ascending=False)

    print("\n=== TOP CONFIGURATIONS PER DATASET ===")
    print(best_configs.to_string(index=False))

    # 2. Compactness vs Performance Correlation
    print("\n=== COMPACTNESS CORRELATION (radius_after vs test_hit10) ===")
    for ds in df['dataset'].unique():
        sub = df[df['dataset'] == ds].dropna(subset=['radius_after', 'test_hit10'])
        if len(sub) > 1:
            corr = sub['radius_after'].corr(sub['test_hit10'])
            print(f"  - {ds:11}: {corr: .4f}")

    # 3. Create Markdown Report
    with open("sweeps_summary.md", "w") as f:
        f.write("# ARON Sweep Analysis Results\n\n")
        f.write("## Overview\n")
        f.write(f"Analyzed {len(df)} total runs across {df['dataset'].nunique()} datasets.\n\n")
        
        f.write("## Best Configurations Found\n")
        f.write(best_configs.to_string(index=False))
        f.write("\n\n")
        
        f.write("## Hypothesis Validation: Compactness Correlation\n")
        f.write("Checking if smaller radii (more compactness) correlate with higher Hit@10.\n")
        f.write("> Negative correlation means smaller radius = higher performance (Hypothesis is VALID if negative)\n\n")
        f.write("| Dataset | Radius vs Hit@10 Correlation | Verdict |\n")
        f.write("| :--- | :--- | :--- |\n")
        for ds in df['dataset'].unique():
            sub = df[df['dataset'] == ds].dropna(subset=['radius_after', 'test_hit10'])
            corr = sub['radius_after'].corr(sub['test_hit10']) if len(sub) > 1 else float('nan')
            if corr < -0.2:
                verdict = "✅ Strong Match"
            elif corr < 0:
                verdict = "〰️ Weak Match"
            else:
                verdict = "❌ Inverse Correlation"
            f.write(f"| **{ds}** | {corr:.4f} | {verdict} |\n")

    print("\nFull report saved to sweeps_summary.md")

if __name__ == "__main__":
    analyze()
