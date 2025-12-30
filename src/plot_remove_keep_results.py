#!/usr/bin/env python3
import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


FLOAT_RE = r"([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)"


def find_logs(log_root: str) -> List[Path]:
    root = Path(log_root)
    return sorted([p for p in root.rglob("*.log") if p.is_file()])


def parse_filename_meta(p: Path) -> Dict[str, Optional[str]]:
    """
    Tries to parse dataset/scope/keep/seed from filename patterns like:
      citeseer_remove_only_intra_c0p_only_keep5_seed0.log
      cora_remove_only_both_cp_minus_c0p_keep90_seed3.log
    """
    name = p.name
    meta = {"dataset": None, "kind": None, "scope": None, "keep": None, "seed": None}

    m = re.search(
        r"^(?P<dataset>[^_]+)_remove_only_(?P<kind>intra|inter|both)_(?P<scope>.+?)_keep(?P<keep>\d+)_seed(?P<seed>\d+)",
        name,
    )
    if m:
        meta.update(m.groupdict())
    return meta


def parse_one_log(p: Path, hit_k: int, use_best: bool) -> Dict:
    meta = parse_filename_meta(p)

    dataset = meta["dataset"]
    kind = meta["kind"]
    scope = meta["scope"]
    keep = meta["keep"]
    seed = meta["seed"]

    with p.open("r", errors="ignore") as f:
        lines = f.readlines()

    for line in lines:
        if line.startswith("Dataset:"):
            dataset = line.split("Dataset:", 1)[1].strip()
            break

    for line in lines:
        m = re.search(r"\[REMOVE-ONLY-INIT\]\s*scope=([A-Za-z0-9_\-]+)\s+keep=(\d+)%", line)
        if m:
            scope = scope or m.group(1)
            keep = keep or m.group(2)
            break

    init_tot_re = re.compile(r"\[REMOVE-ONLY-INIT\]\s*E_scope0=(\d+)\s+target_remove_total=(\d+)")
    E_scope0 = None
    target_remove_total = None
    for line in lines:
        m = init_tot_re.search(line)
        if m:
            E_scope0 = int(m.group(1))
            target_remove_total = int(m.group(2))
            break

    removed_tot_re = re.compile(r"\bremoved_scope_so_far=(\d+)\s*/\s*(\d+)")
    removed_this_re = re.compile(r"\bremoved_this_epoch=(\d+)")
    removed_total_final = None
    target_total_seen = None
    removed_this_final = None

    for line in lines:
        m = removed_tot_re.search(line)
        if m:
            removed_total_final = int(m.group(1))
            target_total_seen = int(m.group(2))
        m2 = removed_this_re.search(line)
        if m2:
            removed_this_final = int(m2.group(1))

    kept_edges_final = None
    keep_pct_achieved = None
    remove_pct_achieved = None
    if E_scope0 is not None and removed_total_final is not None:
        kept_edges_final = int(E_scope0 - removed_total_final)
        keep_pct_achieved = 100.0 * kept_edges_final / float(max(1, E_scope0))
        remove_pct_achieved = 100.0 * removed_total_final / float(max(1, E_scope0))

    best_hit = None
    best_epoch = None

    hit_best_re = re.compile(
        rf"best\s+hit@{hit_k}\s+epoch\s*=\s*(\d+),\s*hit@{hit_k}\s*=\s*{FLOAT_RE}"
    )
    final_line_re = re.compile(r"\[FINAL TEST\]\s*Hit@K:\s*(.*)$")

    if use_best:
        for line in lines:
            m = hit_best_re.search(line)
            if m:
                best_epoch = int(m.group(1))
                best_hit = float(m.group(2))
        hit_value = best_hit
    else:
        hit_value = None
        for line in lines:
            m = final_line_re.search(line)
            if not m:
                continue
            payload = m.group(1)
            for kv in payload.split(","):
                kv = kv.strip()
                mm = re.match(rf"^{hit_k}\s*=\s*{FLOAT_RE}$", kv)
                if mm:
                    hit_value = float(mm.group(1))

    rad_re = re.compile(rf"\[RADIUS\]\s*epoch=(\d+)\s*mean={FLOAT_RE}")
    r_epochs = []
    r_means = []
    for line in lines:
        m = rad_re.search(line)
        if m:
            r_epochs.append(int(m.group(1)))
            r_means.append(float(m.group(2)))

    if len(r_means) > 0:
        radius_start = float(r_means[0])
        radius_end = float(r_means[-1])
        radius_best = float(np.min(r_means))
        radius_best_epoch = int(r_epochs[int(np.argmin(r_means))])
    else:
        radius_start = radius_end = radius_best = None
        radius_best_epoch = None

    keep_i = int(keep) if keep is not None else None
    seed_i = int(seed) if seed is not None else None

    return {
        "path": str(p),
        "dataset": dataset,
        "kind": kind,
        "scope": scope,
        "keep": keep_i,
        "seed": seed_i,
        "hit_k": hit_k,
        "hit": hit_value,
        "best_epoch": best_epoch,
        "radius_start": radius_start,
        "radius_end": radius_end,
        "radius_best": radius_best,
        "radius_best_epoch": radius_best_epoch,
        "E_scope0": E_scope0,
        "target_remove_total": target_remove_total,
        "target_total_seen": target_total_seen,
        "removed_total_final": removed_total_final,
        "removed_this_final": removed_this_final,
        "kept_edges_final": kept_edges_final,
        "keep_pct_achieved": keep_pct_achieved,
        "remove_pct_achieved": remove_pct_achieved,
    }


def agg_mean_std(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    g = df.groupby(["dataset", "scope", "keep"], dropna=False)[value_col]
    out = g.agg(["mean", "std", "count"]).reset_index()
    out = out.rename(columns={"mean": f"{value_col}_mean", "std": f"{value_col}_std", "count": "n"})
    return out


def agg_mean_std_by_x(df: pd.DataFrame, x_col: str, value_col: str) -> pd.DataFrame:
    g = df.groupby(["dataset", "scope", x_col], dropna=False)[value_col]
    out = g.agg(["mean", "std", "count"]).reset_index()
    out = out.rename(columns={"mean": f"{value_col}_mean", "std": f"{value_col}_std", "count": "n"})
    return out


def _plot_line_or_errorbar(x, y, yerr, label: str, show_std: bool):
    if show_std and yerr is not None:
        plt.errorbar(x, y, yerr=yerr, marker="o", label=label)
    else:
        plt.plot(x, y, marker="o", label=label)


def plot_keep_curve(
    agg: pd.DataFrame,
    y_mean: str,
    y_std: str,
    out_path: Path,
    title: str,
    ylabel: str,
    show_std: bool,
):
    for dataset, sub in agg.groupby("dataset"):
        plt.figure()
        for scope, ss in sub.groupby("scope"):
            ss = ss.sort_values("keep")
            x = ss["keep"].to_numpy()
            y = ss[y_mean].to_numpy()
            e = ss[y_std].to_numpy() if (show_std and y_std in ss.columns) else None
            _plot_line_or_errorbar(x, y, e, label=str(scope), show_std=show_std)
        plt.gca().invert_xaxis()
        plt.xlabel("Keep % (nominal; from ver)")
        plt.ylabel(ylabel)
        plt.title(f"{title} | {dataset}")
        plt.legend()
        plt.tight_layout()
        fp = out_path.parent / f"{out_path.stem}_{dataset}{out_path.suffix}"
        plt.savefig(fp, dpi=200)
        plt.close()


def plot_x_curve(
    agg: pd.DataFrame,
    x_col: str,
    y_mean: str,
    y_std: str,
    out_path: Path,
    title: str,
    xlabel: str,
    ylabel: str,
    show_std: bool,
    invert_x: bool = False,
):
    for dataset, sub in agg.groupby("dataset"):
        plt.figure()
        for scope, ss in sub.groupby("scope"):
            ss = ss.sort_values(x_col)
            x = ss[x_col].to_numpy()
            y = ss[y_mean].to_numpy()
            e = ss[y_std].to_numpy() if (show_std and y_std in ss.columns) else None
            _plot_line_or_errorbar(x, y, e, label=str(scope), show_std=show_std)
        if invert_x:
            plt.gca().invert_xaxis()
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(f"{title} | {dataset}")
        plt.legend()
        plt.tight_layout()
        fp = out_path.parent / f"{out_path.stem}_{dataset}{out_path.suffix}"
        plt.savefig(fp, dpi=200)
        plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_root", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--hit_k", type=int, default=10)
    ap.add_argument("--use_best", action="store_true", help="use 'best hit@K' lines instead of final test line")
    ap.add_argument("--show_std", action="store_true", help="show std-dev errorbars (otherwise plot mean only)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    logs = find_logs(args.log_root)
    if not logs:
        print(f"[ERROR] No .log files found under: {args.log_root}")
        return

    rows = []
    for p in logs:
        try:
            rows.append(parse_one_log(p, hit_k=args.hit_k, use_best=args.use_best))
        except Exception as e:
            print(f"[WARN] failed parsing {p}: {e}")

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["dataset", "scope", "keep"])

    csv_path = out_dir / "remove_keep_parsed.csv"
    df.to_csv(csv_path, index=False)
    print(f"[OK] wrote {csv_path}")

    # nominal keep%
    hit_agg = agg_mean_std(df.dropna(subset=["hit"]), "hit")
    rad_agg = agg_mean_std(df.dropna(subset=["radius_end"]), "radius_end")

    hit_agg.to_csv(out_dir / "remove_keep_hit_agg.csv", index=False)
    rad_agg.to_csv(out_dir / "remove_keep_radius_agg.csv", index=False)

    plot_keep_curve(
        hit_agg,
        y_mean="hit_mean",
        y_std="hit_std",
        out_path=out_dir / f"keep_vs_hit@{args.hit_k}.png",
        title=f"Nominal Keep% vs Hit@{args.hit_k}" + (" (best)" if args.use_best else " (final)"),
        ylabel=f"Hit@{args.hit_k}",
        show_std=args.show_std,
    )

    plot_keep_curve(
        rad_agg,
        y_mean="radius_end_mean",
        y_std="radius_end_std",
        out_path=out_dir / "keep_vs_radius_end.png",
        title="Nominal Keep% vs Radius (end of training)",
        ylabel="Radius (cosine-normalized)",
        show_std=args.show_std,
    )

    # achieved removals (count)
    df_rm = df.dropna(subset=["removed_total_final", "hit", "radius_end"])

    hit_agg_rm = agg_mean_std_by_x(df_rm, "removed_total_final", "hit")
    rad_agg_rm = agg_mean_std_by_x(df_rm, "removed_total_final", "radius_end")

    hit_agg_rm.to_csv(out_dir / "removed_count_hit_agg.csv", index=False)
    rad_agg_rm.to_csv(out_dir / "removed_count_radius_agg.csv", index=False)

    plot_x_curve(
        hit_agg_rm,
        x_col="removed_total_final",
        y_mean="hit_mean",
        y_std="hit_std",
        out_path=out_dir / f"removed_count_vs_hit@{args.hit_k}.png",
        title=f"Achieved Removed(scope) count vs Hit@{args.hit_k}" + (" (best)" if args.use_best else " (final)"),
        xlabel="Removed edges in scope (achieved, final)",
        ylabel=f"Hit@{args.hit_k}",
        show_std=args.show_std,
        invert_x=False,
    )

    plot_x_curve(
        rad_agg_rm,
        x_col="removed_total_final",
        y_mean="radius_end_mean",
        y_std="radius_end_std",
        out_path=out_dir / "removed_count_vs_radius_end.png",
        title="Achieved Removed(scope) count vs Radius (end of training)",
        xlabel="Removed edges in scope (achieved, final)",
        ylabel="Radius (cosine-normalized)",
        show_std=args.show_std,
        invert_x=False,
    )

    print("[DONE] plots saved under:", out_dir)


if __name__ == "__main__":
    main()
