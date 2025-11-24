#!/usr/bin/env python
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt

def main(csv_path):
    print(f"[PLOT] loading {csv_path}")
    if not os.path.exists(csv_path):
        print(f"[PLOT] ERROR: file not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    print("[PLOT] columns:", list(df.columns))

    # sanity checks
    required = ["epoch", "radius_mean", "val_hit@10"]
    for col in required:
        if col not in df.columns:
            print(f"[PLOT] ERROR: column '{col}' not found in CSV.")
            return

    epochs = df["epoch"].values
    radius = df["radius_mean"].values
    hit10  = df["val_hit@10"].values

    # -----------------------------
    # 1) Epoch vs Radius & Hit@10
    # -----------------------------
    fig, ax1 = plt.subplots(figsize=(8, 4.5))

    ax1.plot(epochs, radius, marker="o", label="Mean radius")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Mean radius")
    ax1.tick_params(axis="y")

    ax2 = ax1.twinx()
    ax2.plot(epochs, hit10, marker="x", linestyle="--", label="Val Hit@10")
    ax2.set_ylabel("Validation Hit@10")
    ax2.tick_params(axis="y")

    title = os.path.splitext(os.path.basename(csv_path))[0]
    plt.title(title)

    fig.tight_layout()
    out_path1 = os.path.splitext(csv_path)[0] + "_epoch_radius_hit10.png"
    plt.savefig(out_path1, bbox_inches="tight")
    print(f"[PLOT] saved {out_path1}")
    plt.close(fig)

    # -----------------------------
    # 2) Radius vs Hit@10 scatter
    # -----------------------------
    fig2, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(radius, hit10)
    ax.set_xlabel("Mean radius")
    ax.set_ylabel("Validation Hit@10")
    ax.set_title(title + " (radius vs Hit@10)")

    fig2.tight_layout()
    out_path2 = os.path.splitext(csv_path)[0] + "_scatter_radius_hit10.png"
    plt.savefig(out_path2, bbox_inches="tight")
    print(f"[PLOT] saved {out_path2}")
    plt.close(fig2)

    print("[PLOT] done.")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python plot_radius_hit.py path/to/radius_csv")
        sys.exit(1)
    main(sys.argv[1])
