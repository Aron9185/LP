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

    title = os.path.splitext(os.path.basename(csv_path))[0]

    # Figure out Hit@10 column name (optional)
    hit10_col = None
    if "val_hit@10" in df.columns:
        hit10_col = "val_hit@10"
    elif "val_hit10" in df.columns:
        hit10_col = "val_hit10"

    # --------------------------------------------
    # Mode 1: old epoch-based radius log
    #   columns: epoch, radius_mean, (val_hit@10 / val_hit10)
    #   -> plot epoch vs hit, epoch vs radius separately
    # --------------------------------------------
    if ("epoch" in df.columns) and ("radius_mean" in df.columns):
        print("[PLOT] detected mode: epoch-based radius log")

        epochs = df["epoch"].values
        radius = df["radius_c0p_a0.8_g1"].values

        # 1) Epoch vs Hit@10 (if available)
        if hit10_col is not None:
            hit10 = df[hit10_col].values
            fig_h, ax_h = plt.subplots(figsize=(8, 4.5))
            ax_h.plot(epochs, hit10, marker="x")
            ax_h.set_xlabel("Epoch")
            ax_h.set_ylabel("Validation Hit@10")
            ax_h.set_title(title + " (Epoch vs Hit@10)")
            fig_h.tight_layout()
            out_h = os.path.splitext(csv_path)[0] + "_epoch_hit10.png"
            plt.savefig(out_h, bbox_inches="tight")
            print(f"[PLOT] saved {out_h}")
            plt.close(fig_h)
        else:
            print("[PLOT] WARNING: no Hit@10 column found for epoch-based plot")

        # 2) Epoch vs radius_mean
        fig_r, ax_r = plt.subplots(figsize=(8, 4.5))
        ax_r.plot(epochs, radius, marker="o")
        ax_r.set_xlabel("Epoch")
        ax_r.set_ylabel("radius_mean")
        ax_r.set_title(title + " (Epoch vs radius_mean)")
        fig_r.tight_layout()
        out_r = os.path.splitext(csv_path)[0] + "_epoch_radius_mean.png"
        plt.savefig(out_r, bbox_inches="tight")
        print(f"[PLOT] saved {out_r}")
        plt.close(fig_r)

        print("[PLOT] done.")
        return

    # --------------------------------------------
    # Mode 2: prune sweep CSV
    #   columns: frac_removed, radius_cp_mean, (val_hit10 / val_hit@10)
    #   -> plot frac_removed vs hit, frac_removed vs radius_cp_mean
    # --------------------------------------------
    if ("frac_removed" in df.columns) and ("radius_cp_mean" in df.columns):
        print("[PLOT] detected mode: prune sweep log")

        frac = df["frac_removed"].values
        radius_cp_mean = df["radius_cp_mean"].values

        # 1) frac_removed vs Hit@10 (if available)
        if hit10_col is not None:
            hit10 = df[hit10_col].values
            fig_h, ax_h = plt.subplots(figsize=(8, 4.5))
            ax_h.plot(frac, hit10, marker="x")
            ax_h.set_xlabel("Fraction of removed edges")
            ax_h.set_ylabel("Validation Hit@10")
            ax_h.set_title(title + " (Remove fraction vs Hit@10)")
            fig_h.tight_layout()
            out_h = os.path.splitext(csv_path)[0] + "_frac_removed_hit10.png"
            plt.savefig(out_h, bbox_inches="tight")
            print(f"[PLOT] saved {out_h}")
            plt.close(fig_h)
        else:
            print("[PLOT] WARNING: no Hit@10 column found for sweep-based plot")

        # 2) frac_removed vs radius_cp_mean
        fig_r, ax_r = plt.subplots(figsize=(8, 4.5))
        ax_r.plot(frac, radius_cp_mean, marker="o")
        ax_r.set_xlabel("Fraction of removed edges")
        ax_r.set_ylabel("radius_cp_mean")
        ax_r.set_title(title + " (Remove fraction vs radius_cp_mean)")
        fig_r.tight_layout()
        out_r = os.path.splitext(csv_path)[0] + "_frac_removed_radius_cp_mean.png"
        plt.savefig(out_r, bbox_inches="tight")
        print(f"[PLOT] saved {out_r}")
        plt.close(fig_r)

        print("[PLOT] done.")
        return

    # --------------------------------------------
    # Fallback: unknown CSV format
    # --------------------------------------------
    print("[PLOT] ERROR: unrecognized CSV format.")
    print("  Expected either:")
    print("    - epoch-based:  columns include ['epoch', 'radius_mean', 'val_hit@10' or 'val_hit10'], or")
    print("    - sweep-based:  columns include ['frac_removed', 'radius_cp_mean', 'val_hit@10' or 'val_hit10'].")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python plot_radius_hit.py path/to/csv")
        sys.exit(1)
    main(sys.argv[1])
