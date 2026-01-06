#!/usr/bin/env python3
import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


FLOAT_RE = r"([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)"
COLOR_CYCLE = ["red", "green", "blue"]


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


def _trim_outliers_quantile(
    df: pd.DataFrame,
    group_cols: Sequence[str],
    value_col: str,
    outlier_frac: float,
    outlier_side: str,
    min_group_n: int = 4,
) -> Tuple[pd.DataFrame, int]:
    """
    Quantile-trim outliers within each group.
      - outlier_side='both' trims outlier_frac total (half low tail + half high tail)
      - 'high' trims only upper tail
      - 'low' trims only lower tail
    Returns (trimmed_df, num_removed).
    """
    if outlier_frac <= 0:
        return df, 0
    if not (0.0 <= outlier_frac < 0.5):
        raise ValueError("--outlier_frac must be in [0, 0.5).")
    if outlier_side not in ("both", "high", "low"):
        raise ValueError("--outlier_side must be one of: both, high, low.")
    if value_col not in df.columns:
        return df, 0

    work = df.copy()
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")

    keep_mask = pd.Series(True, index=work.index)

    g = work.groupby(list(group_cols), dropna=False)
    for _, idx in g.groups.items():
        idx = list(idx)
        s = work.loc[idx, value_col].dropna()
        if len(s) < min_group_n:
            continue

        if outlier_side == "both":
            q_lo = outlier_frac / 2.0
            q_hi = 1.0 - (outlier_frac / 2.0)
        elif outlier_side == "low":
            q_lo = outlier_frac
            q_hi = 1.0
        else:  # high
            q_lo = 0.0
            q_hi = 1.0 - outlier_frac

        lo = s.quantile(q_lo) if q_lo > 0 else -np.inf
        hi = s.quantile(q_hi) if q_hi < 1 else np.inf

        vals = work.loc[idx, value_col]
        group_keep = vals.isna() | ((vals >= lo) & (vals <= hi))
        keep_mask.loc[idx] = keep_mask.loc[idx] & group_keep

    trimmed = work[keep_mask].copy()
    removed = int((~keep_mask).sum())
    return trimmed, removed


def agg_mean_std(
    df: pd.DataFrame,
    value_col: str,
    *,
    outlier_frac: float = 0.0,
    outlier_side: str = "both",
) -> pd.DataFrame:
    group_cols = ["dataset", "scope", "keep"]
    df2, removed = _trim_outliers_quantile(df, group_cols, value_col, outlier_frac, outlier_side)
    if removed > 0:
        print(
            f"[OUTLIER] trimmed {removed} rows for value='{value_col}' within groups={group_cols} "
            f"(frac={outlier_frac}, side={outlier_side})"
        )

    g = df2.groupby(group_cols, dropna=False)[value_col]
    out = g.agg(["mean", "std", "count"]).reset_index()
    out = out.rename(columns={"mean": f"{value_col}_mean", "std": f"{value_col}_std", "count": "n"})
    return out


def agg_mean_std_by_x(
    df: pd.DataFrame,
    x_col: str,
    value_col: str,
    *,
    outlier_frac: float = 0.0,
    outlier_side: str = "both",
) -> pd.DataFrame:
    group_cols = ["dataset", "scope", x_col]
    df2, removed = _trim_outliers_quantile(df, group_cols, value_col, outlier_frac, outlier_side)
    if removed > 0:
        print(
            f"[OUTLIER] trimmed {removed} rows for value='{value_col}' within groups={group_cols} "
            f"(frac={outlier_frac}, side={outlier_side})"
        )

    g = df2.groupby(group_cols, dropna=False)[value_col]
    out = g.agg(["mean", "std", "count"]).reset_index()
    out = out.rename(columns={"mean": f"{value_col}_mean", "std": f"{value_col}_std", "count": "n"})
    return out


def _plot_line_or_errorbar(x, y, yerr, label: str, show_std: bool, color: str):
    if show_std and yerr is not None:
        plt.errorbar(
            x,
            y,
            yerr=yerr,
            marker="o",
            color=color,
            ecolor=color,
            capsize=3,
            label=label,
        )
    else:
        plt.plot(
            x,
            y,
            marker="o",
            color=color,
            label=label,
        )


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
        for i, (scope, ss) in enumerate(sub.groupby("scope")):
            color = COLOR_CYCLE[i % len(COLOR_CYCLE)]
            ss = ss.sort_values("keep")
            x = ss["keep"].to_numpy()
            y = ss[y_mean].to_numpy()
            e = ss[y_std].to_numpy() if (show_std and y_std in ss.columns) else None
            _plot_line_or_errorbar(x, y, e, label=str(scope), show_std=show_std, color=color)
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
        for i, (scope, ss) in enumerate(sub.groupby("scope")):
            color = COLOR_CYCLE[i % len(COLOR_CYCLE)]
            ss = ss.sort_values(x_col)
            x = ss[x_col].to_numpy()
            y = ss[y_mean].to_numpy()
            e = ss[y_std].to_numpy() if (show_std and y_std in ss.columns) else None
            _plot_line_or_errorbar(x, y, e, label=str(scope), show_std=show_std, color=color)
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


def plot_scatter_removed_vs_hit(
    df: pd.DataFrame,
    out_path: Path,
    title: str,
    hit_k: int,
):
    """
    Raw scatter: each point is one run (one log/seed).
    x = removed_total_final, y = hit
    """
    for dataset, sub in df.groupby("dataset"):
        plt.figure()
        for i, (scope, ss) in enumerate(sub.groupby("scope")):
            color = COLOR_CYCLE[i % len(COLOR_CYCLE)]
            plt.scatter(
                ss["removed_total_final"].to_numpy(),
                ss["hit"].to_numpy(),
                color=color,
                label=str(scope),
                alpha=0.8,
            )
        plt.xlabel("Removed edges in scope (achieved, final)")
        plt.ylabel(f"Hit@{hit_k}")
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

    # outlier trimming
    ap.add_argument(
        "--outlier_frac",
        type=float,
        default=0.0,
        help="portion to trim as outliers within each group BEFORE mean/std. "
        "If side=both, trims half from low tail and half from high tail. "
        "Example: 0.1 => drop 5% lowest + 5% highest. (range: [0, 0.5))",
    )
    ap.add_argument(
        "--outlier_side",
        type=str,
        default="both",
        choices=["both", "high", "low"],
        help="which tail(s) to trim when --outlier_frac > 0",
    )

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
    hit_agg = agg_mean_std(
        df.dropna(subset=["hit"]),
        "hit",
        outlier_frac=args.outlier_frac,
        outlier_side=args.outlier_side,
    )
    rad_agg = agg_mean_std(
        df.dropna(subset=["radius_end"]),
        "radius_end",
        outlier_frac=args.outlier_frac,
        outlier_side=args.outlier_side,
    )

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

    # achieved removals (count): keep only rows that have all needed fields
    df_rm = df.dropna(subset=["removed_total_final", "hit", "radius_end"])

    # aggregate curves (mean/std) against removed_total_final
    hit_agg_rm = agg_mean_std_by_x(
        df_rm,
        "removed_total_final",
        "hit",
        outlier_frac=args.outlier_frac,
        outlier_side=args.outlier_side,
    )
    rad_agg_rm = agg_mean_std_by_x(
        df_rm,
        "removed_total_final",
        "radius_end",
        outlier_frac=args.outlier_frac,
        outlier_side=args.outlier_side,
    )

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

    # NEW: raw scatter of hit vs removed edge count (each point = one run)
    df_rm_scatter, removed_sc = _trim_outliers_quantile(
        df_rm,
        group_cols=["dataset", "scope", "removed_total_final"],
        value_col="hit",
        outlier_frac=args.outlier_frac,
        outlier_side=args.outlier_side,
    )
    if removed_sc > 0:
        print(
            "[OUTLIER] trimmed "
            f"{removed_sc} rows for RAW scatter value='hit' within "
            "groups=['dataset','scope','removed_total_final'] "
            f"(frac={args.outlier_frac}, side={args.outlier_side})"
        )

    plot_scatter_removed_vs_hit(
        df_rm_scatter,
        out_path=out_dir / f"removed_count_vs_hit@{args.hit_k}_scatter.png",
        title=f"RAW: Removed(scope) count vs Hit@{args.hit_k}" + (" (best)" if args.use_best else " (final)"),
        hit_k=args.hit_k,
    )

    print("[DONE] plots saved under:", out_dir)


if __name__ == "__main__":
    main()
