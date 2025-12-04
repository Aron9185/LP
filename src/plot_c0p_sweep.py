#!/usr/bin/env python3
import os
import glob
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re


def guess_col(df, candidates, name):
    """Pick the first existing column from candidates; crash with info if none."""
    for c in candidates:
        if c in df.columns:
            print(f"[INFO] Using column '{c}' for {name}")
            return c
    raise RuntimeError(
        f"Could not find any {name} column. "
        f"Tried: {candidates}. Available columns: {list(df.columns)}"
    )


def load_and_stack(pattern):
    files = sorted(glob.glob(pattern))
    if not files:
        raise RuntimeError(f"No CSV files match pattern: {pattern}")
    print(f"[INFO] Found {len(files)} files:")
    for f in files:
        print("  -", f)

    dfs = [pd.read_csv(f) for f in files]
    print("[INFO] Columns in first file:", dfs[0].columns.tolist())
    return dfs, files


def aggregate_runs(dfs):
    """
    Aggregate multiple runs:
      - align them on the common frac_removed grid (intersection across runs)
      - compute mean/std for radius and Hit@K.
    Assumes columns (from your CSVs):
      frac_removed, radius_c0p_a0.8_g1.0, test_hit3, test_hit10
    """
    x_col = "frac_removed"
    radius_col = "radius_c0p_a0.8_g1.0"
    hit3_col = "test_hit3"
    hit10_col = "test_hit10"

    # 1) find common x grid (intersection of all frac_removed sets)
    x_sets = [set(df[x_col].values.tolist()) for df in dfs]
    x_common = sorted(set.intersection(*x_sets))
    print(f"[AGG] common frac_removed grid size = {len(x_common)}")

    x_common = np.array(x_common, dtype=np.float32)

    metrics = {
        "radius": radius_col,
        "hit3": hit3_col,
        "hit10": hit10_col,
    }

    agg = {"frac_removed": x_common}

    for key, col in metrics.items():
        runs_vals = []
        for df in dfs:
            # keep only rows whose frac_removed is in the common grid
            df_sub = df[df[x_col].isin(x_common)].copy()
            df_sub = df_sub.sort_values(x_col)
            vals = df_sub[col].values
            if len(vals) != len(x_common):
                raise RuntimeError(
                    f"[AGG] length mismatch for metric {key}: "
                    f"expected {len(x_common)}, got {len(vals)}"
                )
            runs_vals.append(vals)

        mat = np.stack(runs_vals, axis=0)  # [num_runs, num_points]
        agg[f"{key}_mean"] = mat.mean(axis=0)
        agg[f"{key}_std"] = mat.std(axis=0)

    df_agg = pd.DataFrame(agg)
    return df_agg

# --- change signature to pass which radius we want (C0p vs CP) ---
def aggregate_runs_binned(dfs, bin_width=0.01, min_runs_per_bin=1, x_max=1.0, want_c0p: bool = True):
    x_col = "frac_removed"
    # radius column choice
    if want_c0p:
        cands = [c for c in dfs[0].columns if c.startswith("radius_c0p_")]
        radius_col = cands[0] if cands else "radius_cp_mean"
    else:
        radius_col = "radius_cp_mean"

    hit3_col = "test_hit3"
    hit10_col = "test_hit10"

    bins = np.arange(0, x_max + bin_width + 1e-12, bin_width)
    centers = (bins[:-1] + bins[1:]) / 2.0

    metrics = {"radius": radius_col, "hit3": hit3_col, "hit10": hit10_col}
    per_metric_stack = {k: [] for k in metrics}

    for df in dfs:
        df = df[[x_col, radius_col, hit3_col, hit10_col]].copy().dropna(subset=[x_col])
        df["bin_center"] = pd.cut(df[x_col], bins=bins, labels=centers, include_lowest=True)
        for key, col in metrics.items():
            g = df.groupby("bin_center")[col].mean()
            series = g.reindex(centers)
            per_metric_stack[key].append(series.values.astype(float))

    agg = {"frac_removed": centers}
    valid_mask = None
    for key, mats in per_metric_stack.items():
        M = np.vstack(mats)
        counts = np.sum(~np.isnan(M), axis=0)
        mean = np.nanmean(M, axis=0)
        std  = np.nanstd(M, axis=0)
        keep = counts >= min_runs_per_bin
        valid_mask = keep if valid_mask is None else (valid_mask | keep)
        agg[f"{key}_mean"] = mean
        agg[f"{key}_std"]  = std
        agg[f"{key}_count"] = counts

    for k in list(agg.keys()):
        agg[k] = agg[k][valid_mask] if isinstance(agg[k], np.ndarray) else np.array(agg[k])[valid_mask]
    return pd.DataFrame(agg)


# --- update plot_aggregate to use dynamic labels & save paths ---
def plot_aggregate(df, *, save_dir, file_stub, title_prefix, x_label, radius_line_label, show=True):
    os.makedirs(save_dir, exist_ok=True)
    x = df["frac_removed"]

    # Radius
    plt.figure()
    m, s = df["radius_mean"], df["radius_std"]
    plt.plot(x, m, marker="o", label=radius_line_label)
    plt.fill_between(x, m - s, m + s, alpha=0.2)
    plt.xlabel(x_label)
    plt.ylabel("Cluster radius")
    plt.title(f"{title_prefix}: radius vs removed fraction")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    out = os.path.join(save_dir, f"{file_stub}_radius.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print("[INFO] Saved radius figure to", out)

    # Hit@K
    plt.figure()
    for key, label in [("hit3", "Hit@3 (mean)"), ("hit10", "Hit@10 (mean)")]:
        m, s = df[f"{key}_mean"], df[f"{key}_std"]
        plt.plot(x, m, marker="o", label=label)
        plt.fill_between(x, m - s, m + s, alpha=0.2)
    plt.xlabel(x_label)
    plt.ylabel("Test Hit@K")
    plt.title(f"{title_prefix}: Hit@K vs removed fraction")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    out = os.path.join(save_dir, f"{file_stub}_hit.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print("[INFO] Saved Hit@K figure to", out)

    if show:
        plt.show()
    else:
        plt.close("all")


# --- add this helper near the top ---
def infer_from_pattern(pattern: str):
    """
    Parse dataset, mode, and intra-scope variant from a sweep glob.
    Matches files like:
      cora_remove_only_intra_cp_c0p_only_seed0_idx0.csv
      cora_remove_only_intra_cp_cp_all_seed0_idx0.csv
      cora_remove_only_intra_cp_cp_minus_c0p_seed0_idx0.csv
    Returns a dict with labels and save paths.
    """
    dataset = os.path.basename(os.path.dirname(pattern.rstrip("/")))
    base = os.path.basename(pattern)

    # remove_only_<mode>_<scope>_(cp_all|c0p_only|cp_minus_c0p)
    m = re.search(
        r"remove_only_(intra|inter|both)_(cp|c0p)_(cp_all|c0p_only|cp_minus_c0p)",
        base
    )
    if not m:
        # fallback (older patterns)
        m = re.search(r"remove_only_(intra|inter|both)_(cp|c0p)", base)
    mode   = m.group(1) if m else "unknown"
    scope  = m.group(2) if m else "cp"  # default cp
    variant = m.group(3) if (m and m.lastindex and m.lastindex >= 3) else None

    # human for mode
    mode_phrase = {
        "intra": "intra-cluster",
        "inter": "inter-cluster",
        "both":  "all-cluster",
        "unknown": "cluster",
    }[mode]

    # scope core (for deciding radius series)
    scope_core = "C0p" if scope.lower().startswith("c0p") else "CP"

    # scope label shown to user (variant-aware)
    if variant == "cp_all":
        scope_label = "CP (all)"
    elif variant == "c0p_only":
        scope_label = "C0p (only)"
    elif variant == "cp_minus_c0p":
        scope_label = r"CP \ C0p"
    else:
        scope_label = scope_core

    # choose which radius column to plot
    want_c0p_radius = (variant == "c0p_only") or (scope_core == "C0p")
    radius_line_label = "C0p radius (mean)" if want_c0p_radius else "CP radius (mean)"

    # x-axis
    x_label = f"Fraction of {mode_phrase} {scope_label} edges removed"

    # save dir is the same directory as we glob from
    save_dir = os.path.dirname(pattern.rstrip("/"))

    # file stub clearly encodes the variant
    # e.g., cora_intra_c0p_only / cora_intra_cp_all / cora_intra_cp_minus_c0p
    stub_variant = variant if variant else scope.lower()
    file_stub = f"{dataset}_{mode}_{stub_variant}"

    # title
    title_prefix = f"{dataset} / {mode} {scope_label}"

    return {
        "dataset": dataset,
        "mode": mode,
        "variant": variant,
        "scope_core": scope_core,
        "want_c0p_radius": want_c0p_radius,
        "radius_line_label": radius_line_label,
        "x_label": x_label,
        "save_dir": save_dir,
        "file_stub": file_stub,
        "title_prefix": title_prefix,
    }



def build_save_prefix(args_save_prefix: str | None, dataset: str, mode: str, scope_short: str):
    """
    If user passed --save_prefix, append inferred suffix; else use directory next to pattern.
    """
    suffix = f"{dataset}_{mode}_{scope_short}"
    if args_save_prefix:
        # If user passed a directory path like ".../cora", append suffix.
        return f"{args_save_prefix}_{mode}_{scope_short}"
    else:
        # default next to dataset dir
        return os.path.join(os.path.dirname(os.path.dirname(args.pattern)), dataset, suffix)

# --- wire it in main() ---
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pattern", type=str,
        default="/home/retro/ARON/logs/c0p_sweep/cora/cora_remove_only_intra_cp_cp_all_seed*_idx*.csv")
    ap.add_argument("--no_show", action="store_true")
    ap.add_argument("--bin_width", type=float, default=0.01)
    ap.add_argument("--min_runs_per_bin", type=int, default=1)
    args = ap.parse_args()

    info = infer_from_pattern(args.pattern)

    dfs, _ = load_and_stack(args.pattern)
    df_agg = aggregate_runs_binned(
        dfs,
        bin_width=args.bin_width,
        min_runs_per_bin=args.min_runs_per_bin,
        want_c0p=info["want_c0p_radius"],
    ).rename(columns={"radius": "radius_mean", "radius_std": "radius_std"})

    plot_aggregate(
        df_agg,
        save_dir=info["save_dir"],
        file_stub=info["file_stub"],
        title_prefix=info["title_prefix"],
        x_label=info["x_label"],
        radius_line_label=info["radius_line_label"],
        show=not args.no_show,
    )

if __name__ == "__main__":
    main()