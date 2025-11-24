#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Plot per-epoch compactness (radius) and hit rates (Hit@3/Hit@10) for each run.

Example:
  python tools/plot_epoch_trends.py \
    --log-dirs logs/1108 logs/1109 \
    --radius-dir logs/radius \
    --out-dir plots/remove_only \
    --glob "*.log" \
    --include-splits VAL

Outputs per run (under out-dir/{dataset}/{scope}/):
  - {dataset}_{ver}_seed{seed}_idx{idx}_trend.csv
  - {dataset}_{ver}_seed{seed}_idx{idx}_radius.png
  - {dataset}_{ver}_seed{seed}_idx{idx}_hits.png
And an overall index:
  - {out-dir}/per_run_summary.csv
"""

import argparse, re, sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
import matplotlib.pyplot as plt

# ------------------ meta block in logs ------------------
META_START = re.compile(r"^=+ RUN META =+\s*$")
META_END   = re.compile(r"^=+\s*$")
RE_FIELD = {
    "dataset": re.compile(r"\bdataset\s*=\s*([A-Za-z0-9_.-]+)"),
    "ver":     re.compile(r"\bver\s*=\s*([A-Za-z0-9_.-]+)"),
    "seed":    re.compile(r"\bseed\s*=\s*([0-9]+)"),
    "idx":     re.compile(r"\bidx\s*=\s*([0-9]+)"),
}

# ------------------ filename fallback ------------------
RE_FNAME = re.compile(
    r"""^
    (?P<dataset>[A-Za-z0-9_]+)_
    (?P<ver>remove_only_[^_]+(?:_[^_]+)*)_
    .*?seed(?P<seed>\d+).*?_idx(?P<idx>\d+).*?\.log$
    """, re.X
)

# scope/frac from ver
RE_VER = re.compile(r"^remove_only_(intra|inter|both)(?:_(c0p|noncore|cp))?(?:_frac([0-9]*\.?[0-9]+))?$")

# ------------------ in-body patterns ------------------
RE_RADIUS = re.compile(r"\[RADIUS\]\s*epoch\s*=\s*(\d+)\s*mean\s*=\s*([0-9]*\.?[0-9]+)")
RE_HITK   = re.compile(r"\[(VAL|TEST|FINAL TEST)\][^\n]*Hit@K\s*:\s*(.+)", re.I)
RE_PAIR   = re.compile(r"(\d+)\s*=\s*([0-9]*\.?[0-9]+)")

# epoch trackers that appear in many formats
RE_EPOCH_INLINE = [
    re.compile(r"\bepoch\s*[:=]\s*(\d+)", re.I),
    re.compile(r"\bEpoch\s*[:# ]\s*(\d+)", re.I),
    re.compile(r"\[ep(?:och)?\s*=?\s*(\d+)\]", re.I),
]

def _read_text(p: Path) -> Optional[str]:
    try:
        return p.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        print(f"[READ-ERR] {p}: {e}", file=sys.stderr)
        return None

def _to_fraction(s: str) -> Optional[float]:
    try:
        v = float(s)
    except:
        return None
    if 1.0 < v <= 100.0:  # tolerate percents
        v /= 100.0
    return v

def _extract_meta_block(text: str) -> Dict[str, Any]:
    meta = {}
    inb, buf = False, []
    for ln in text.splitlines():
        if META_START.search(ln):
            inb, buf = True, []
            continue
        if inb and META_END.search(ln):
            joined = "\n".join(buf)
            for k, pat in RE_FIELD.items():
                m = pat.search(joined)
                if m: meta[k] = m.group(1)
            break
        if inb:
            buf.append(ln)
    for k in ("seed","idx"):
        if k in meta:
            try: meta[k] = int(meta[k])
            except: pass
    return meta

def _parse_from_fname(name: str) -> Dict[str, Any]:
    d = {}
    m = RE_FNAME.match(name)
    if m:
        d = m.groupdict()
        for k in ("seed","idx"):
            d[k] = int(d[k])
    return d

def _scope_from_ver(ver: Optional[str]) -> Optional[str]:
    if not ver: return None
    m = RE_VER.match(ver)
    if m: return m.group(2) or None
    for s in ("c0p","noncore","cp"):
        if f"_{s}" in ver: return s
    return None

def _expected_radius_csv(radius_dir: Path, dataset: str, ver: str, seed: int, idx: int) -> Path:
    return radius_dir / f"{dataset}_{ver}_seed{seed}_idx{idx}.csv"

def _load_radius_series(radius_csv: Path, log_text: str) -> pd.DataFrame:
    if radius_csv.exists():
        try:
            df = pd.read_csv(radius_csv)
            # require at least epoch & radius_mean
            if not {"epoch","radius_mean"}.issubset(df.columns):
                raise ValueError("missing epoch/radius_mean in radius CSV")
            for c in ("added","removed","mod_ratio"):
                if c not in df.columns: df[c] = pd.NA
            return df[["epoch","radius_mean","added","removed","mod_ratio"]].copy()
        except Exception as e:
            print(f"[WARN] bad radius CSV {radius_csv}: {e}")

    # fallback: parse from log
    rows = []
    for m in RE_RADIUS.finditer(log_text):
        rows.append({"epoch": int(m.group(1)), "radius_mean": float(m.group(2))})
    if not rows:
        return pd.DataFrame(columns=["epoch","radius_mean","added","removed","mod_ratio"])
    df = pd.DataFrame(rows).sort_values("epoch")
    df["added"] = pd.NA
    df["removed"] = pd.NA
    df["mod_ratio"] = pd.NA
    return df

def _parse_hits_with_epoch(text: str, include_splits: List[str]) -> pd.DataFrame:
    """
    Attach epoch to each Hit@K line. We keep a rolling 'current_epoch' updated
    whenever a line contains something like 'epoch=12', 'Epoch 12', or '[ep=12]'.
    """
    rows = []
    current_epoch: Optional[int] = None
    for ln in text.splitlines():
        # update epoch cursor if the line mentions an epoch
        for rx in RE_EPOCH_INLINE:
            me = rx.search(ln)
            if me:
                try:
                    current_epoch = int(me.group(1))
                except:
                    pass
                break

        mh = RE_HITK.search(ln)
        if not mh:
            continue
        split = mh.group(1).upper()
        if include_splits and split not in include_splits:
            continue

        payload = mh.group(2)
        pairs = dict((k, _to_fraction(v)) for k, v in RE_PAIR.findall(payload))
        rows.append({
            "epoch": current_epoch,  # may be None if your log never prints epochs with hits
            "split": split,
            "hit1": pairs.get("1"),
            "hit3": pairs.get("3"),
            "hit10": pairs.get("10"),
        })

    df = pd.DataFrame(rows)
    # If epochs are missing, try to backfill by order (0..N-1)
    if not df.empty and df["epoch"].isna().all():
        df = df.reset_index(drop=True)
        df["epoch"] = df.index  # best effort fallback
    return df

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _plot_radius(df_r: pd.DataFrame, title: str, out_png: Path):
    plt.figure()
    if not df_r.empty:
        plt.plot(df_r["epoch"], df_r["radius_mean"])
    plt.xlabel("epoch")
    plt.ylabel("mean within-cluster radius")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

def _plot_hits(df_h: pd.DataFrame, title: str, out_png: Path):
    plt.figure()
    if not df_h.empty:
        # draw lines for hit@3 and hit@10 (VAL/TEST merged if both included)
        # aggregate by epoch taking the last value seen per epoch
        g = df_h.groupby("epoch", as_index=False).agg({"hit3":"last","hit10":"last"})
        if not g.empty:
            plt.plot(g["epoch"], g["hit3"], label="Hit@3")
            plt.plot(g["epoch"], g["hit10"], label="Hit@10")
            plt.legend()
    plt.xlabel("epoch")
    plt.ylabel("Hit@K")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dirs", nargs="+", required=True)
    ap.add_argument("--glob", type=str, default="*.log")
    ap.add_argument("--radius-dir", type=str, default="logs/radius")
    ap.add_argument("--out-dir", type=str, default="plots/remove_only")
    ap.add_argument("--include-splits", nargs="+", default=["VAL"], help="Which splits to use for epoch curves (e.g., VAL TEST)")
    ap.add_argument("--save-merged-csv", action="store_true")
    args = ap.parse_args()

    radius_dir = Path(args.radius_dir)
    out_dir    = Path(args.out_dir)
    _ensure_dir(out_dir)

    summary_rows: List[Dict[str, Any]] = []
    total_logs = 0
    processed  = 0

    for d in args.log_dirs:
        base = Path(d)
        if not base.exists():
            print(f"[WARN] missing log dir: {d}")
            continue
        hits = list(base.rglob(args.glob))
        total_logs += len(hits)

        for p in sorted(hits):
            txt = _read_text(p)
            if txt is None:
                continue

            meta_blk = _extract_meta_block(txt)
            fmeta    = _parse_from_fname(p.name)

            dataset = meta_blk.get("dataset") or fmeta.get("dataset")
            ver     = meta_blk.get("ver") or fmeta.get("ver")
            seed    = meta_blk.get("seed") if "seed" in meta_blk else fmeta.get("seed")
            idx     = meta_blk.get("idx")  if "idx" in meta_blk  else fmeta.get("idx")
            if not dataset or not ver or seed is None or idx is None:
                continue

            scope = _scope_from_ver(ver) or "NA"
            run_tag = f"{dataset}_{ver}_seed{seed}_idx{idx}"
            run_dir = out_dir / dataset / scope
            _ensure_dir(run_dir)

            # radius series
            r_csv = _expected_radius_csv(radius_dir, dataset, ver, seed, idx)
            df_r = _load_radius_series(r_csv, txt).copy()
            # performance series (per-epoch)
            df_h = _parse_hits_with_epoch(txt, include_splits=[s.upper() for s in args.include_splits]).copy()

            # Merge by epoch (outer join), keep last seen hit per epoch
            if not df_h.empty:
                df_h_last = df_h.sort_values("epoch").groupby("epoch", as_index=False).last()
            else:
                df_h_last = pd.DataFrame(columns=["epoch","split","hit1","hit3","hit10"])

            merged = pd.merge(df_r, df_h_last[["epoch","hit3","hit10"]], on="epoch", how="outer").sort_values("epoch")

            # Save per-run merged CSV if requested
            if args.save_merged_csv:
                merged.to_csv(run_dir / f"{run_tag}_trend.csv", index=False)

            # Plots
            _plot_radius(merged.dropna(subset=["epoch"]), title=f"{run_tag} — radius", out_png=run_dir / f"{run_tag}_radius.png")
            _plot_hits(merged.dropna(subset=["epoch"]), title=f"{run_tag} — Hit@K", out_png=run_dir / f"{run_tag}_hits.png")

            # Summary row
            start_r = float(merged["radius_mean"].dropna().iloc[0]) if merged["radius_mean"].notna().any() else None
            end_r   = float(merged["radius_mean"].dropna().iloc[-1]) if merged["radius_mean"].notna().any() else None
            delta_r = (end_r - start_r) if (start_r is not None and end_r is not None) else None
            n_ep_r  = int(merged["radius_mean"].notna().sum())
            n_ep_h  = int((merged["hit3"].notna() | merged["hit10"].notna()).sum())

            total_removed = None
            if "removed" in merged.columns and merged["removed"].notna().any():
                try:
                    total_removed = int(pd.to_numeric(merged["removed"], errors="coerce").fillna(0).sum())
                except Exception:
                    total_removed = None

            summary_rows.append({
                "dataset": dataset,
                "scope": scope,
                "ver": ver,
                "seed": seed,
                "idx": idx,
                "log_file": str(p),
                "radius_csv": str(r_csv),
                "epochs_with_radius": n_ep_r,
                "epochs_with_hits": n_ep_h,
                "start_radius": start_r,
                "end_radius": end_r,
                "delta_radius": delta_r,
                "total_removed": total_removed,
            })
            processed += 1

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "per_run_summary.csv", index=False)
    print(f"[DONE] scanned={total_logs} | processed={processed} | summary={out_dir/'per_run_summary.csv'}")

if __name__ == "__main__":
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 60)
    main()
