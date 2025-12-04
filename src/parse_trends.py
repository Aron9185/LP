#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Parse ARON remove-only logs to extract per-epoch radius & performance trends,
plot them per run, and write per-run CSVs + an overall summary.

USAGE EXAMPLE:
  python tools/parse_trends.py \
    --log-dirs logs/1108 logs/1109 \
    --radius-dir logs/radius \
    --out-dir plots/remove_only \
    --glob "*.log" \
    --save-per-run-csv

Notes:
- Expected radius CSV path for a run:
    {radius_dir}/{dataset}_{ver}_seed{seed}_idx{idx}.csv
  with columns: epoch,radius_mean,added,removed,mod_ratio
- If the radius CSV is missing, we fallback to parsing lines like:
    "[RADIUS] epoch=123 mean=0.612345"
- Performance series are parsed from lines like:
    "[VAL] ... Hit@K: 1=..., 3=..., 10=..."
  Final and "best" are parsed from:
    "[FINAL TEST] Hit@K: ..."
    "best hit@3 epoch = 55, hit@3 = 0.5197"
"""

import argparse, re, sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
import matplotlib.pyplot as plt

# ----------------------------- meta block (optional) -----------------------------
META_START = re.compile(r"^=+ RUN META =+\s*$")
META_END   = re.compile(r"^=+\s*$")
RE_FIELD = {
    "dataset": re.compile(r"\bdataset\s*=\s*([A-Za-z0-9_.-]+)"),
    "ver":     re.compile(r"\bver\s*=\s*([A-Za-z0-9_.-]+)"),
    "mode":    re.compile(r"\bmode\s*=\s*([A-Za-z0-9_.-]+)"),
    "seed":    re.compile(r"\bseed\s*=\s*([0-9]+)"),
    "idx":     re.compile(r"\bidx\s*=\s*([0-9]+)"),
}

# ----------------------------- filename patterns (your style) -----------------------------
RE_FNAME_SIMPLE = re.compile(
    r"""^
    (?P<dataset>[A-Za-z0-9_]+)_
    (?P<ver>remove_only_[^_]+(?:_[^_]+)*)_
    .*?
    seed(?P<seed>\d+).*?
    _idx(?P<idx>\d+)
    .*?\.log$
    """,
    re.X
)

# Scope/frac parser
RE_VER = re.compile(r"^remove_only_(intra|inter|both)(?:_(c0p|noncore|cp))?(?:_frac([0-9]*\.?[0-9]+))?$")

# ----------------------------- in-body patterns -----------------------------
RE_RADIUS_LINE = re.compile(r"\[RADIUS\]\s*epoch\s*=\s*(\d+)\s*mean\s*=\s*([0-9]*\.?[0-9]+)")
RE_BUDGET_LINE = re.compile(r"\[BUDGET\].*add/remove this epoch\s*=\s*(\d+)\s*/\s*(\d+)", re.I)

RE_HIT_LINE    = re.compile(r"\[(VAL|TEST|FINAL TEST)\][^\n]*Hit@K\s*:\s*(.+)", re.I)
RE_PAIR        = re.compile(r"(\d+)\s*=\s*([0-9]*\.?[0-9]+)")
RE_BEST_H3     = re.compile(r"best\s+hit@?3\s+epoch\s*=\s*(\d+)\s*,\s*hit@?3\s*=\s*([0-9]*\.?[0-9]+)", re.I)
RE_BEST_H10    = re.compile(r"best\s+hit@?10\s+epoch\s*=\s*(\d+)\s*,\s*hit@?10\s*=\s*([0-9]*\.?[0-9]+)", re.I)

# ----------------------------- helpers -----------------------------
def _read_text(path: Path) -> Optional[str]:
    try:
        return path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        print(f"[READ-ERR] {path}: {e}", file=sys.stderr)
        return None

def _to_fraction(val_str: str) -> Optional[float]:
    try:
        v = float(val_str)
    except:
        return None
    if 1.0 < v <= 100.0:
        v = v / 100.0
    return v

def _extract_meta_block(text: str) -> Dict[str, Any]:
    meta: Dict[str, Any] = {}
    lines = text.splitlines()
    in_block, buf = False, []
    for ln in lines:
        if META_START.search(ln):
            in_block, buf = True, []
            continue
        if in_block and META_END.search(ln):
            joined = "\n".join(buf)
            for k, pat in RE_FIELD.items():
                m = pat.search(joined)
                if m:
                    meta[k] = m.group(1)
            break
        if in_block:
            buf.append(ln)
    # Cast
    for k in ("seed","idx"):
        if k in meta:
            try: meta[k] = int(meta[k])
            except: pass
    return meta

def _parse_from_fname(name: str) -> Dict[str, Any]:
    d: Dict[str, Any] = {}
    m = RE_FNAME_SIMPLE.match(name)
    if m:
        d.update({k: m.group(k) for k in m.groupdict().keys()})
        for k in ("seed","idx"):
            if k in d:
                try: d[k] = int(d[k])
                except: pass
    return d

def _parse_scope_frac(ver: Optional[str]) -> Tuple[Optional[str], Optional[float]]:
    if not ver:
        return None, None
    m = RE_VER.match(ver)
    if not m:
        # fallback: scan tokens
        scope = None
        for s in ("c0p","noncore","cp"):
            if f"_{s}" in ver:
                scope = s
                break
        frac = None
        mf = re.search(r"_frac([0-9]*\.?[0-9]+)", ver)
        if mf: frac = float(mf.group(1))
        return scope, frac
    scope = m.group(2) or None
    frac = float(m.group(3)) if m.group(3) else None
    return scope, frac

def _expected_radius_csv(radius_dir: Path, dataset: str, ver: str, seed: int, idx: int) -> Path:
    name = f"{dataset}_{ver}_seed{seed}_idx{idx}.csv"
    return radius_dir / name

def _parse_hit_series(text: str) -> pd.DataFrame:
    """
    Returns a dataframe with columns: step, split, hit1, hit3, hit10
    'step' is an increasing counter when epoch index cannot be recovered.
    """
    rows = []
    step = 0
    for m in RE_HIT_LINE.finditer(text):
        split = m.group(1).upper()  # VAL / TEST / FINAL TEST
        payload = m.group(2)
        pairs = dict((k, _to_fraction(v)) for k, v in RE_PAIR.findall(payload))
        rows.append({
            "step": step,
            "split": split,
            "hit1": pairs.get("1"),
            "hit3": pairs.get("3"),
            "hit10": pairs.get("10"),
        })
        step += 1
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["step","split","hit1","hit3","hit10"])

def _parse_best_markers(text: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    m3 = RE_BEST_H3.findall(text)
    if m3:
        out["best_hit3_epoch"] = int(m3[-1][0])
        out["best_hit3"] = _to_fraction(m3[-1][1])
    m10 = RE_BEST_H10.findall(text)
    if m10:
        out["best_hit10_epoch"] = int(m10[-1][0])
        out["best_hit10"] = _to_fraction(m10[-1][1])
    return out

def _parse_radius_from_log(text: str) -> pd.DataFrame:
    rows = []
    # try to also collect add/remove if available
    budget_rows: Dict[int, Tuple[int,int]] = {}
    # pre-pass for budget lines
    for b in RE_BUDGET_LINE.finditer(text):
        # These lines often appear once per epoch, but may not include epoch.
        # We'll attach them later if counts match rows length; otherwise leave NaN.
        pass

    for m in RE_RADIUS_LINE.finditer(text):
        epoch = int(m.group(1))
        r     = float(m.group(2))
        rows.append({"epoch": epoch, "radius_mean": r})

    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["epoch","radius_mean","added","removed","mod_ratio"])

    # added/removed per-epoch are only guaranteed in CSV; leave NaN if unknown
    df["added"] = pd.NA
    df["removed"] = pd.NA
    df["mod_ratio"] = pd.NA
    return df

def _load_radius_series(radius_csv: Path, log_text: str) -> pd.DataFrame:
    if radius_csv.exists():
        try:
            df = pd.read_csv(radius_csv)
            need_cols = {"epoch","radius_mean"}
            if not need_cols.issubset(set(df.columns)):
                raise ValueError("radius CSV missing required columns")
            # Make sure added/removed/mod_ratio exist
            for c in ("added","removed","mod_ratio"):
                if c not in df.columns:
                    df[c] = pd.NA
            return df
        except Exception as e:
            print(f"[WARN] bad radius CSV {radius_csv}: {e}")
    # fallback to parse from log
    return _parse_radius_from_log(log_text)

def _ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

# ----------------------------- plotting helpers -----------------------------
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

def _plot_perf(df_p: pd.DataFrame, title: str, out_png: Path):
    plt.figure()
    if not df_p.empty:
        # If we have VAL steps, draw VAL hit@3 and hit@10 as lines by step
        df_val = df_p[df_p["split"]=="VAL"]
        if not df_val.empty:
            plt.plot(df_val["step"], df_val["hit3"], label="VAL Hit@3")
            plt.plot(df_val["step"], df_val["hit10"], label="VAL Hit@10")
        # Always mark FINAL TEST if present
        df_fin = df_p[df_p["split"]=="FINAL TEST"]
        if not df_fin.empty:
            # mark the last final test hit@3/10 as horizontal lines
            h3 = df_fin["hit3"].dropna().iloc[-1] if df_fin["hit3"].notna().any() else None
            h10 = df_fin["hit10"].dropna().iloc[-1] if df_fin["hit10"].notna().any() else None
            if h3 is not None:
                plt.axhline(h3, linestyle="--", label=f"FINAL Hit@3={h3:.3f}")
            if h10 is not None:
                plt.axhline(h10, linestyle="--", label=f"FINAL Hit@10={h10:.3f}")
        # Legend only if something was drawn
        if not df_val.empty or not df_fin.empty:
            plt.legend()
    plt.xlabel("step (VAL eval order)")
    plt.ylabel("Hit@K")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()

# ----------------------------- main -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dirs", nargs="+", required=True)
    ap.add_argument("--glob", type=str, default="*.log")
    ap.add_argument("--radius-dir", type=str, default="logs/radius")
    ap.add_argument("--out-dir", type=str, default="plots/remove_only")
    ap.add_argument("--save-per-run-csv", action="store_true")
    args = ap.parse_args()

    radius_dir = Path(args.radius_dir)
    out_dir    = Path(args.out_dir)
    _ensure_dir(out_dir)

    summary_rows: List[Dict[str, Any]] = []

    total_logs = 0
    processed = 0

    for d in args.log-dirs if hasattr(args, "log-dirs") else args.log_dirs:  # argparse hyphen fix
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
            fname    = _parse_from_fname(p.name)

            dataset = meta_blk.get("dataset") or fname.get("dataset")
            ver     = meta_blk.get("ver") or fname.get("ver")
            seed    = meta_blk.get("seed") if "seed" in meta_blk else fname.get("seed")
            idx     = meta_blk.get("idx")  if "idx" in meta_blk  else fname.get("idx")

            if not dataset or not ver or seed is None or idx is None:
                # Not a remove-only run (or malformed)
                continue

            scope, frac_hint = _parse_scope_frac(ver)
            run_tag = f"{dataset}_{ver}_seed{seed}_idx{idx}"
            run_dir = out_dir / dataset / (scope or "NA")
            _ensure_dir(run_dir)

            # radius series
            r_csv = _expected_radius_csv(radius_dir, dataset, ver, seed, idx)
            df_r = _load_radius_series(r_csv, txt)

            # performance series
            df_p = _parse_hit_series(txt)
            best_marks = _parse_best_markers(txt)

            # save per-run merged csv if requested
            if args.save_per_run_csv:
                merged_csv = run_dir / f"{run_tag}_trend.csv"
                # Merge on nearest step/epoch is undefined; we keep separate tabs by writing two CSVs
                df_r.to_csv(run_dir / f"{run_tag}_radius.csv", index=False)
                df_p.to_csv(run_dir / f"{run_tag}_perf.csv", index=False)

            # plots
            _plot_radius(df_r, title=f"{run_tag} — radius", out_png=run_dir / f"{run_tag}_radius.png")
            _plot_perf(df_p, title=f"{run_tag} — performance", out_png=run_dir / f"{run_tag}_perf.png")

            # summary stats for this run
            start_r = float(df_r["radius_mean"].iloc[0]) if not df_r.empty else None
            end_r   = float(df_r["radius_mean"].iloc[-1]) if not df_r.empty else None
            delta_r = (end_r - start_r) if (start_r is not None and end_r is not None) else None

            total_removed = None
            if "removed" in df_r.columns and df_r["removed"].notna().any():
                try:
                    total_removed = int(pd.to_numeric(df_r["removed"], errors="coerce").fillna(0).sum())
                except Exception:
                    total_removed = None

            final_row = df_p[df_p["split"]=="FINAL TEST"].tail(1)
            final_h3 = float(final_row["hit3"].iloc[0]) if not final_row.empty and final_row["hit3"].notna().any() else None
            final_h10 = float(final_row["hit10"].iloc[0]) if not final_row.empty and final_row["hit10"].notna().any() else None

            rec = {
                "dataset": dataset,
                "ver": ver,
                "scope": scope,
                "frac_hint": frac_hint,
                "seed": seed,
                "idx": idx,
                "log_file": str(p),
                "radius_csv": str(r_csv),
                "start_radius": start_r,
                "end_radius": end_r,
                "delta_radius": delta_r,
                "total_removed": total_removed,
                "final_hit3": final_h3,
                "final_hit10": final_h10,
                "best_hit3": best_marks.get("best_hit3"),
                "best_hit3_epoch": best_marks.get("best_hit3_epoch"),
                "best_hit10": best_marks.get("best_hit10"),
                "best_hit10_epoch": best_marks.get("best_hit10_epoch"),
            }
            summary_rows.append(rec)
            processed += 1

    summary = pd.DataFrame(summary_rows)
    summ_path = out_dir / "per_run_summary.csv"
    summary.to_csv(summ_path, index=False)
    print(f"[DONE] scanned={total_logs} | processed={processed} | summary={summ_path}")

if __name__ == "__main__":
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 60)
    main()
