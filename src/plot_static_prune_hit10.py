#!/usr/bin/env python
import os
import re
import sys
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt


def parse_log(log_path):
    """
    Parse a single static-prune log.

    We look for:
      - Dataset: <name>
      - [static-preprune] frac=0.05, scope=c0p_only
      - [FINAL TEST] Hit@K: 1=..., 3=..., 10=...
      - [FINAL-RADIUS] core(c0p): mean=... std=... p50=... p90=... max=... n=...
        (also try to parse all_non_noise if present)
    """
    dataset = None
    frac = None
    scope = None
    hit3 = None
    hit10 = None

    # radius summaries (final)
    c0p_r_mean = None
    c0p_r_std = None
    all_r_mean = None
    all_r_std = None

    # patterns
    pat_dataset = re.compile(r"^Dataset:\s*([A-Za-z0-9_]+)")
    pat_static = re.compile(r"\[static-preprune\].*frac=([0-9.]+).*scope=([A-Za-z0-9_]+)")
    pat_hit = re.compile(r"\[FINAL TEST\] Hit@K:\s*(.*)$")

    # Example:
    # [FINAL-RADIUS] core(c0p): mean=0.123456 std=0.111111 p50=... p90=... max=... n=123
    pat_radius = re.compile(
        r"^\[FINAL-RADIUS\]\s*(?P<name>[^:]+):\s*mean=(?P<mean>[0-9.eE+-]+)\s+std=(?P<std>[0-9.eE+-]+)"
    )

    with open(log_path, "r") as f:
        for line in f:
            # Dataset line
            if dataset is None:
                m = pat_dataset.search(line)
                if m:
                    dataset = m.group(1)

            # Static prune meta
            if "[static-preprune]" in line and "frac=" in line:
                m = pat_static.search(line)
                if m:
                    frac = float(m.group(1))
                    scope = m.group(2)

            # FINAL TEST Hit@K
            if "[FINAL TEST] Hit@K:" in line:
                m = pat_hit.search(line)
                if m:
                    pairs = dict(re.findall(r"(\d+)=([0-9.]+)", m.group(1)))
                    if "3" in pairs:
                        hit3 = float(pairs["3"])
                    if "10" in pairs:
                        hit10 = float(pairs["10"])

            # FINAL-RADIUS summaries
            if line.startswith("[FINAL-RADIUS]"):
                m = pat_radius.search(line.strip())
                if m:
                    name = m.group("name").strip()
                    r_mean = float(m.group("mean"))
                    r_std = float(m.group("std"))

                    # match the exact names we printed
                    if name == "core(c0p)":
                        c0p_r_mean = r_mean
                        c0p_r_std = r_std
                    elif name == "all_non_noise":
                        all_r_mean = r_mean
                        all_r_std = r_std
                    # (optional) you could also store "noncore" here if you want later

    # Require at least hit10 so we can always plot @10
    if None in (dataset, frac, scope, hit10):
        print(
            f"[WARN] failed to fully parse {log_path}: "
            f"dataset={dataset}, frac={frac}, scope={scope}, "
            f"hit3={hit3}, hit10={hit10}, "
            f"c0p_r_mean={c0p_r_mean}, c0p_r_std={c0p_r_std}"
        )
        return None

    return {
        "dataset": dataset,
        "frac": frac,
        "scope": scope,
        "hit3": hit3,      # may be None
        "hit10": hit10,
        "c0p_r_mean": c0p_r_mean,  # may be None if run didn't print it
        "c0p_r_std": c0p_r_std,
        "all_r_mean": all_r_mean,
        "all_r_std": all_r_std,
        "path": log_path,
    }


def collect_results(base_dir):
    """
    Walk base_dir and collect all logs.

    results[(dataset, scope)][frac] = {
        "hit3": [...], "hit10": [...],
        "c0p_r_mean": [...], "c0p_r_std": [...],
        "all_r_mean": [...], "all_r_std": [...]
    }
    """
    results = defaultdict(lambda: defaultdict(lambda: {
        "hit3": [], "hit10": [],
        "c0p_r_mean": [], "c0p_r_std": [],
        "all_r_mean": [], "all_r_std": [],
    }))

    for root, _, files in os.walk(base_dir):
        for fname in files:
            if not fname.endswith(".log"):
                continue
            fpath = os.path.join(root, fname)
            info = parse_log(fpath)
            if info is None:
                continue

            key = (info["dataset"], info["scope"])
            bucket = results[key][info["frac"]]

            if info["hit3"] is not None:
                bucket["hit3"].append(info["hit3"])
            bucket["hit10"].append(info["hit10"])

            if info["c0p_r_mean"] is not None:
                bucket["c0p_r_mean"].append(info["c0p_r_mean"])
            if info["c0p_r_std"] is not None:
                bucket["c0p_r_std"].append(info["c0p_r_std"])

            if info["all_r_mean"] is not None:
                bucket["all_r_mean"].append(info["all_r_mean"])
            if info["all_r_std"] is not None:
                bucket["all_r_std"].append(info["all_r_std"])

            print(
                f"[PARSE] {fpath}: dataset={info['dataset']} "
                f"scope={info['scope']} frac={info['frac']} "
                f"hit3={info['hit3']} hit10={info['hit10']} "
                f"c0p_r_mean={info['c0p_r_mean']} all_r_mean={info['all_r_mean']}"
            )

    return results


def make_single_plot(fracs, means, dataset, scope, y_label,
                     out_dir, filename_suffix, ylim=None):
    """Helper to plot a single metric vs prune fraction."""
    means = np.array(means, dtype=float)

    if np.all(np.isnan(means)):
        print(f"[SKIP] {dataset} {scope} {y_label}: no data")
        return

    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(10, 4))
    plt.plot(fracs, means, marker="o", linestyle="-")

    plt.xlabel("Prune fraction")
    plt.ylabel(y_label)
    plt.title(f"{dataset} – {scope} (static prune)")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.xticks(fracs, [f"{f:.2f}" for f in fracs])

    if ylim is not None:
        plt.ylim(*ylim)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.18)

    out_path = os.path.join(out_dir, f"{dataset}_{scope}_{filename_suffix}.png")
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[PLOT] saved {out_path}")


def plot_results(results, base_dir):
    """
    For each (dataset, scope), output:
      - Hit@10 vs prune frac
      - Hit@3 vs prune frac
      - c0p radius(mean) vs prune frac   (if present)
    """
    out_dir10 = os.path.join(base_dir, "plots_hit10")
    out_dir3 = os.path.join(base_dir, "plots_hit3")
    out_dir_r = os.path.join(base_dir, "plots_radius_c0p")

    for (dataset, scope), frac_dict in sorted(results.items()):
        if not frac_dict:
            continue

        fracs = sorted(frac_dict.keys())

        hit3_means = []
        hit10_means = []
        c0p_r_means = []

        for f in fracs:
            vals3 = frac_dict[f]["hit3"]
            vals10 = frac_dict[f]["hit10"]
            vals_r = frac_dict[f]["c0p_r_mean"]

            hit3_means.append(np.nan if len(vals3) == 0 else float(np.mean(vals3)))
            hit10_means.append(np.nan if len(vals10) == 0 else float(np.mean(vals10)))
            c0p_r_means.append(np.nan if len(vals_r) == 0 else float(np.mean(vals_r)))

        print(f"[PLOT] dataset={dataset} scope={scope} | fracs={fracs}")

        # Hit@10
        make_single_plot(
            fracs, hit10_means, dataset, scope,
            y_label="Test Hit@10",
            out_dir=out_dir10,
            filename_suffix="hit10_vs_prune_frac",
            ylim=(0.0, 1.0),
        )

        # Hit@3
        make_single_plot(
            fracs, hit3_means, dataset, scope,
            y_label="Test Hit@3",
            out_dir=out_dir3,
            filename_suffix="hit3_vs_prune_frac",
            ylim=(0.0, 1.0),
        )

        # c0p radius
        make_single_plot(
            fracs, c0p_r_means, dataset, scope,
            y_label="Final c0p radius (mean)",
            out_dir=out_dir_r,
            filename_suffix="c0p_radius_vs_prune_frac",
            ylim=None,  # radius scale depends on metric
        )


def main():
    if len(sys.argv) != 2:
        print(
            "Usage: python plot_static_prune_hits.py "
            "path/to/static_prune_aron_desc"
        )
        sys.exit(1)

    base_dir = sys.argv[1]
    if not os.path.isdir(base_dir):
        print(f"[ERROR] base_dir not found: {base_dir}")
        sys.exit(1)

    print(f"[INFO] scanning logs under: {base_dir}")
    results = collect_results(base_dir)
    if not results:
        print("[INFO] no logs parsed; nothing to plot.")
        return

    plot_results(results, base_dir)


if __name__ == "__main__":
    main()
