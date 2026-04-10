#!/usr/bin/env python3
import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


FLOAT_RE = r"([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)"


def find_logs(log_root: str) -> List[Path]:
    root = Path(log_root)
    return sorted([p for p in root.rglob("*.log") if p.is_file()])


def parse_filename_meta(p: Path) -> Dict[str, Optional[str]]:
    """
    Tries to parse dataset/kind/scope/keep/seed from filename patterns like:
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

    # optional: dataset line inside log
    for line in lines:
        if line.startswith("Dataset:"):
            dataset = line.split("Dataset:", 1)[1].strip()
            break

    # authoritative keep/scope from init line
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
    removed_total_final = None
    target_total_seen = None
    for line in lines:
        m = removed_tot_re.search(line)
        if m:
            removed_total_final = int(m.group(1))
            target_total_seen = int(m.group(2))

    keep_pct_achieved = None
    remove_pct_achieved = None
    if E_scope0 is not None and removed_total_final is not None:
        kept_edges_final = int(E_scope0 - removed_total_final)
        keep_pct_achieved = 100.0 * kept_edges_final / float(max(1, E_scope0))
        remove_pct_achieved = 100.0 * removed_total_final / float(max(1, E_scope0))

    # hit parsing
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

    # radius end (optional)
    rad_re = re.compile(rf"\[RADIUS\]\s*epoch=(\d+)\s*mean={FLOAT_RE}")
    r_means = []
    for line in lines:
        m = rad_re.search(line)
        if m:
            r_means.append(float(m.group(2)))
    radius_end = float(r_means[-1]) if r_means else None

    return {
        "path": str(p),
        "dataset": dataset,
        "kind": kind,
        "scope": scope,
        "keep": int(keep) if keep is not None else None,
        "seed": int(seed) if seed is not None else None,
        "hit": hit_value,
        "radius_end": radius_end,
        "E_scope0": E_scope0,
        "removed_total_final": removed_total_final,
        "keep_pct_achieved": keep_pct_achieved,
        "remove_pct_achieved": remove_pct_achieved,
        "target_remove_total": target_remove_total,
        "target_total_seen": target_total_seen,
    }


def add_baseline_and_delta_hit(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds:
      - hit_base: mean hit at keep=100 for (dataset, kind, seed)
      - delta_hit: (hit - hit_base) / hit_base
    """
    base = (
        df[(df["keep"] == 100) & df["hit"].notna()]
        .groupby(["dataset", "kind", "seed"], dropna=False)["hit"]
        .mean()
        .reset_index()
        .rename(columns={"hit": "hit_base"})
    )
    df = df.merge(base, on=["dataset", "kind", "seed"], how="left")
    df["delta_hit"] = (df["hit"] - df["hit_base"]) / df["hit_base"]
    return df


def _compute_x(work: pd.DataFrame, x_mode: str) -> Tuple[pd.DataFrame, str]:
    w = work.copy()
    if x_mode == "remove_frac":
        w["x"] = w["removed_total_final"] / w["E_scope0"].replace(0, np.nan)
        xlabel = "Removed fraction (achieved)"
    elif x_mode == "remove_pct":
        w["x"] = w["remove_pct_achieved"]
        xlabel = "Removed % (achieved)"
    elif x_mode == "removed_count":
        w["x"] = w["removed_total_final"]
        xlabel = "Removed edges"
    else:
        raise ValueError(f"Unknown x_mode={x_mode}")
    return w, xlabel


def _trim_outliers_quantile(
    df: pd.DataFrame,
    group_cols: Sequence[str],
    value_col: str,
    outlier_frac: float,
    outlier_side: str,
    min_group_n: int = 4,
) -> Tuple[pd.DataFrame, int]:
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


def _legend_outside_right(fig, ax, ds2c: Dict[str, tuple], sc2m: Dict[str, str]):
    """
    Put Dataset and Scope legends OUTSIDE the axes, on the right side.
    """
    ds_handles = [
        Line2D([0], [0], marker="o", linestyle="None", markersize=7,
               markerfacecolor=ds2c[d], markeredgecolor=ds2c[d])
        for d in ds2c.keys()
    ]
    ds_labels = list(ds2c.keys())

    sc_handles = [
        Line2D([0], [0], marker=sc2m[s], linestyle="None", markersize=7,
               markerfacecolor="black", markeredgecolor="black")
        for s in sc2m.keys()
    ]
    sc_labels = list(sc2m.keys())

    leg1 = ax.legend(
        ds_handles, ds_labels, title="Dataset",
        loc="upper left", bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0, frameon=True
    )
    ax.add_artist(leg1)

    ax.legend(
        sc_handles, sc_labels, title="Scope",
        loc="upper left", bbox_to_anchor=(1.02, 0.52),
        borderaxespad=0.0, frameon=True
    )

    # reserve right margin for legends
    fig.tight_layout(rect=[0.0, 0.0, 0.80, 1.0])


def plot_scatter(
    df: pd.DataFrame,
    out_path: Path,
    hit_k: int,
    x_mode: str,
    y_col: str,      # "hit" or "delta_hit"
    title: str,
    outlier_frac: float = 0.0,
    outlier_side: str = "both",
):
    work, xlabel = _compute_x(df, x_mode)

    if y_col == "hit":
        ylabel = f"Hit@{hit_k} (absolute)"
    elif y_col == "delta_hit":
        ylabel = f"ΔHit@{hit_k} vs keep=100 (relative)"
    else:
        raise ValueError(f"Unknown y_col={y_col}")

    sub = work.dropna(subset=["dataset", "scope", "x", y_col]).copy()
    if len(sub) == 0:
        print(f"[WARN] no rows for {out_path.name}; skipping.")
        return

    # outlier trim (on y)
    if outlier_frac > 0:
        if x_mode in ("remove_frac", "removed_count"):
            group_cols = ["dataset", "scope", "removed_total_final"]
        else:
            group_cols = ["dataset", "scope", "remove_pct_achieved"]

        sub2, removed = _trim_outliers_quantile(
            sub,
            group_cols=group_cols,
            value_col=y_col,
            outlier_frac=outlier_frac,
            outlier_side=outlier_side,
            min_group_n=4,
        )
        if removed > 0:
            print(
                f"[OUTLIER] trimmed {removed} rows for scatter value='{y_col}' within groups={group_cols} "
                f"(frac={outlier_frac}, side={outlier_side})"
            )
        sub = sub2

    if len(sub) == 0:
        print(f"[WARN] no rows after outlier trim for {out_path.name}; skipping.")
        return

    # wider figure because legends sit outside on the right
    fig, ax = plt.subplots(figsize=(10.8, 5))
    cmap = plt.get_cmap("tab10")

    datasets = sorted(sub["dataset"].unique())
    scopes = sorted(sub["scope"].unique())

    ds2c = {d: cmap(i % 10) for i, d in enumerate(datasets)}
    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "<", ">"]
    sc2m = {s: markers[i % len(markers)] for i, s in enumerate(scopes)}

    for (d, s), g in sub.groupby(["dataset", "scope"], dropna=False):
        ax.scatter(
            g["x"],
            g[y_col],
            c=[ds2c.get(d, "gray")],
            marker=sc2m.get(s, "o"),
            alpha=0.8,
            edgecolors="none",
        )

    if y_col == "delta_hit":
        ax.axhline(0.0, linestyle="--", linewidth=1.0, color="black", alpha=0.35)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    _legend_outside_right(fig, ax, ds2c, sc2m)

    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[OK] wrote {out_path}")


def _plot_suite(
    df: pd.DataFrame,
    out_dir: Path,
    hit_k: int,
    x_mode: str,
    use_best: bool,
    outlier_frac: float,
    outlier_side: str,
    keep_window: Optional[Tuple[int, int]],
    prefix: str,
    title_suffix: str,
):
    """
    Generates:
      - full abs
      - full delta
      - (optional) keep-window abs
      - (optional) keep-window delta
    Filenames are prefixed with `prefix`.
    """
    plot_scatter(
        df=df,
        out_path=out_dir / f"{prefix}scatter_hit_full_abs.png",
        hit_k=hit_k,
        x_mode=x_mode,
        y_col="hit",
        title=f"Hit vs Removal (all keeps, absolute){title_suffix}",
        outlier_frac=outlier_frac,
        outlier_side=outlier_side,
    )
    plot_scatter(
        df=df,
        out_path=out_dir / f"{prefix}scatter_hit_full_delta.png",
        hit_k=hit_k,
        x_mode=x_mode,
        y_col="delta_hit",
        title=f"Hit vs Removal (all keeps, relative to keep=100){title_suffix}",
        outlier_frac=outlier_frac,
        outlier_side=outlier_side,
    )

    if keep_window is not None:
        lo, hi = keep_window
        dfw = df[(df["keep"] >= lo) & (df["keep"] <= hi)].copy()

        plot_scatter(
            df=dfw,
            out_path=out_dir / f"{prefix}scatter_hit_keep{hi}_{lo}_abs.png",
            hit_k=hit_k,
            x_mode=x_mode,
            y_col="hit",
            title=f"Hit vs Removal (keep {hi}–{lo}, absolute){title_suffix}",
            outlier_frac=outlier_frac,
            outlier_side=outlier_side,
        )
        plot_scatter(
            df=dfw,
            out_path=out_dir / f"{prefix}scatter_hit_keep{hi}_{lo}_delta.png",
            hit_k=hit_k,
            x_mode=x_mode,
            y_col="delta_hit",
            title=f"Hit vs Removal (keep {hi}–{lo}, relative to keep=100){title_suffix}",
            outlier_frac=outlier_frac,
            outlier_side=outlier_side,
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log_root", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--hit_k", type=int, default=10)
    ap.add_argument("--x_mode", default="remove_frac", choices=["remove_frac", "remove_pct", "removed_count"])

    ap.add_argument("--use_best", action="store_true", help="Use 'best hit@K' line instead of [FINAL TEST].")

    # keep-window plots (e.g., keep 100~80)
    ap.add_argument("--also_plot_keep_window", action="store_true")
    ap.add_argument("--keep_min", type=int, default=80)
    ap.add_argument("--keep_max", type=int, default=100)

    # outlier trimming
    ap.add_argument("--outlier_frac", type=float, default=0.0)
    ap.add_argument("--outlier_side", type=str, default="both", choices=["both", "high", "low"])

    # NEW: per-dataset / per-scope / per-(dataset,scope) suites
    ap.add_argument("--plot_per_dataset", action="store_true", help="Also plot a full suite for each dataset (1 at a time).")
    ap.add_argument("--plot_per_scope", action="store_true", help="Also plot a full suite for each scope (1 at a time).")
    ap.add_argument(
        "--plot_per_dataset_scope",
        action="store_true",
        help="Also plot a full suite for each (dataset, scope) pair (1 at a time).",
    )

    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for p in find_logs(args.log_root):
        try:
            rows.append(parse_one_log(p, args.hit_k, use_best=args.use_best))
        except Exception as e:
            print(f"[WARN] failed parsing {p}: {e}")

    df = pd.DataFrame(rows).dropna(subset=["dataset", "scope", "keep"])
    df = add_baseline_and_delta_hit(df)
    df.to_csv(out_dir / "parsed.csv", index=False)
    print(f"[OK] wrote {out_dir / 'parsed.csv'}")

    keep_window = None
    if args.also_plot_keep_window:
        lo, hi = sorted([args.keep_min, args.keep_max])
        keep_window = (lo, hi)

    # -----------------------
    # GLOBAL suite (all data together)
    # -----------------------
    _plot_suite(
        df=df,
        out_dir=out_dir,
        hit_k=args.hit_k,
        x_mode=args.x_mode,
        use_best=args.use_best,
        outlier_frac=args.outlier_frac,
        outlier_side=args.outlier_side,
        keep_window=keep_window,
        prefix="",
        title_suffix="",
    )

    # -----------------------
    # Per-dataset suites (one dataset at a time)
    # -----------------------
    if args.plot_per_dataset:
        for ds in sorted(df["dataset"].dropna().unique()):
            sub = df[df["dataset"] == ds].copy()
            prefix = f"ds_{ds}__"
            title_suffix = f" | dataset={ds}"
            _plot_suite(
                df=sub,
                out_dir=out_dir,
                hit_k=args.hit_k,
                x_mode=args.x_mode,
                use_best=args.use_best,
                outlier_frac=args.outlier_frac,
                outlier_side=args.outlier_side,
                keep_window=keep_window,
                prefix=prefix,
                title_suffix=title_suffix,
            )

    # -----------------------
    # Per-scope suites (one scope at a time)
    # -----------------------
    if args.plot_per_scope:
        for sc in sorted(df["scope"].dropna().unique()):
            sub = df[df["scope"] == sc].copy()
            prefix = f"sc_{sc}__"
            title_suffix = f" | scope={sc}"
            _plot_suite(
                df=sub,
                out_dir=out_dir,
                hit_k=args.hit_k,
                x_mode=args.x_mode,
                use_best=args.use_best,
                outlier_frac=args.outlier_frac,
                outlier_side=args.outlier_side,
                keep_window=keep_window,
                prefix=prefix,
                title_suffix=title_suffix,
            )

    # -----------------------
    # Per-(dataset, scope) suites (one pair at a time)
    # -----------------------
    if args.plot_per_dataset_scope:
        pairs = df[["dataset", "scope"]].dropna().drop_duplicates()
        for _, row in pairs.iterrows():
            ds = row["dataset"]
            sc = row["scope"]
            sub = df[(df["dataset"] == ds) & (df["scope"] == sc)].copy()
            prefix = f"ds_{ds}__sc_{sc}__"
            title_suffix = f" | dataset={ds}, scope={sc}"
            _plot_suite(
                df=sub,
                out_dir=out_dir,
                hit_k=args.hit_k,
                x_mode=args.x_mode,
                use_best=args.use_best,
                outlier_frac=args.outlier_frac,
                outlier_side=args.outlier_side,
                keep_window=keep_window,
                prefix=prefix,
                title_suffix=title_suffix,
            )

    print("[DONE] plots saved under:", out_dir)


if __name__ == "__main__":
    main()
