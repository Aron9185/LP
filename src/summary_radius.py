#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, sys, re, math
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np

# ---------- optional: p-values if scipy available ----------
try:
    from scipy.stats import pearsonr
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

# ----------------------------- meta block (optional in logs) -----------------------------
META_START = re.compile(r"^=+ RUN META =+\s*$")
META_END   = re.compile(r"^=+\s*$")
RE_FIELD = {
    "dataset": re.compile(r"\bdataset\s*=\s*([A-Za-z0-9_.-]+)"),
    "ver":     re.compile(r"\bver\s*=\s*([A-Za-z0-9_.-]+)"),
    "mode":    re.compile(r"\bmode\s*=\s*([A-Za-z0-9_.-]+)"),
    "seed":    re.compile(r"\bseed\s*=\s*([0-9]+)"),
    "idx":     re.compile(r"\bidx\s*=\s*([0-9]+)"),
    "alpha":   re.compile(r"\balpha\s*=\s*([0-9]*\.?[0-9]+)"),
    "gamma":   re.compile(r"\bgamma\s*=\s*([0-9]*\.?[0-9]+)"),
    "run_tag": re.compile(r"\brun_tag\s*=\s*([A-Za-z0-9_.-]+)"),
}

# ----------------------------- filename patterns (your style) -----------------------------
RE_FNAME_FULL = re.compile(
    r"""^
    (?P<dataset>[A-Za-z0-9_]+)_
    (?P<ver>remove_only_[^_]+(?:_[^_]+)*)_
    a(?P<alpha>\d+\.\d+)_g(?P<gamma>\d+\.\d+)_c0p(?P<c0p_prune>[\d.]+)_
    seed(?P<seed>\d+)_r(?P<aug_ratio>[\d.]+)_fmr(?P<aug_bound>[\d.]+)_d(?P<deg_thr>[\d.]+)_
    (?P<cm>[A-Za-z0-9]+)_idx(?P<idx>\d+)
    (?:_[A-Za-z0-9]+)*
    \.log$
    """,
    re.X
)
RE_FNAME_SIMPLE = re.compile(
    r"""^
    (?P<dataset>[A-Za-z0-9_]+)_
    (?P<ver>remove_only_[^_]+(?:_[^_]+)*)_.*?
    seed(?P<seed>\d+).*?_idx(?P<idx>\d+).*?\.log$
    """,
    re.X
)
RE_TOK_A   = re.compile(r"_a(?P<alpha>[\d.]+)")
RE_TOK_G   = re.compile(r"_g(?P<gamma>[\d.]+)")
RE_TOK_C0P = re.compile(r"_c0p(?P<c0p_prune>[\d.]+)")
RE_TOK_R   = re.compile(r"_r(?P<aug_ratio>[\d.]+)")
RE_TOK_FMR = re.compile(r"_fmr(?P<aug_bound>[\d.]+)")
RE_TOK_D   = re.compile(r"_d(?P<deg_thr>[\d.]+)")
RE_TOK_CM  = re.compile(r"_(?P<cm>gmm|hdbscan|louvain)_")

# ----------------------------- Hit@K patterns -----------------------------
RE_FINAL_HITK_LINE = re.compile(r"\[FINAL TEST\][^\n]*Hit@K\s*:\s*(.+)", re.I)
RE_PAIR = re.compile(r"(\d+)\s*=\s*([0-9]*\.?[0-9]+)")
RE_BEST_HIT3 = re.compile(r"best\s+hit@?3\s+epoch\s*=\s*\d+\s*,\s*hit@?3\s*=\s*([0-9]*\.?[0-9]+)", re.I)
RE_BEST_HIT10 = re.compile(r"best\s+hit@?10\s+epoch\s*=\s*\d+\s*,\s*hit@?10\s*=\s*([0-9]*\.?[0-9]+)", re.I)

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
    # Treat 0-1 as-is; 1-100 as percent
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
    # Normalize numeric
    for k in ("seed", "idx"):
        if k in meta:
            try: meta[k] = int(meta[k])
            except: pass
    for k in ("alpha", "gamma"):
        if k in meta:
            try: meta[k] = float(meta[k])
            except: pass
    return meta

def _parse_hitk_from_text(text: str) -> Tuple[Optional[float], Optional[float]]:
    h3, h10 = None, None
    finals = RE_FINAL_HITK_LINE.findall(text)
    if finals:
        payload = finals[-1]
        pairs = dict((k, _to_fraction(v)) for k, v in RE_PAIR.findall(payload))
        h3 = pairs.get("3")
        h10 = pairs.get("10")
    if h3 is None:
        m3 = RE_BEST_HIT3.findall(text)
        if m3: h3 = _to_fraction(m3[-1])
    if h10 is None:
        m10 = RE_BEST_HIT10.findall(text)
        if m10: h10 = _to_fraction(m10[-1])
    return h3, h10

def _parse_ver_scope_and_frac(ver: str) -> Tuple[Optional[str], Optional[str], Optional[float]]:
    """
    Returns (kind, scope, remove_frac_hint)
      kind: intra|inter|both
      scope: c0p|noncore|cp (None if not present)
      remove_frac_hint: float from '_fracX' in ver (None if absent)
    """
    if not ver:
        return None, None, None
    m = re.match(r"^remove_only_(intra|inter|both)(?:_(c0p|noncore|cp))?(?:_frac([0-9]*\.?[0-9]+))?$", ver)
    if not m:
        # permissive fallback
        m2 = re.search(r"remove_only_(intra|inter|both)", ver)
        kind = m2.group(1) if m2 else None
        scope = None
        for s in ("c0p","noncore","cp"):
            if f"_{s}" in ver:
                scope = s
                break
        f = None
        mf = re.search(r"_frac([0-9]*\.?[0-9]+)", ver)
        if mf: f = float(mf.group(1))
        return kind, scope, f
    kind = m.group(1)
    scope = m.group(2) or None
    frac = float(m.group(3)) if m.group(3) else None
    return kind, scope, frac

def _parse_from_fname(name: str) -> Dict[str, Any]:
    d: Dict[str, Any] = {}
    m = RE_FNAME_FULL.match(name)
    if m:
        d.update({k: m.group(k) for k in m.groupdict().keys()})
    else:
        m2 = RE_FNAME_SIMPLE.match(name)
        if m2:
            d.update({k: m2.group(k) for k in m2.groupdict().keys()})
        # loose tokens
        for rx, key in (
            (RE_TOK_A, "alpha"), (RE_TOK_G, "gamma"), (RE_TOK_C0P, "c0p_prune"),
            (RE_TOK_R, "aug_ratio"), (RE_TOK_FMR, "aug_bound"),
            (RE_TOK_D, "deg_thr"), (RE_TOK_CM, "cm"),
        ):
            tm = rx.search(name)
            if tm and key not in d:
                d[key] = tm.group(key)
    # Cast numerics
    for k in ("seed","idx"):
        if k in d:
            try: d[k] = int(d[k])
            except: pass
    for k in ("alpha","gamma","c0p_prune","aug_ratio","aug_bound","deg_thr"):
        if k in d:
            try: d[k] = float(d[k])
            except: pass
    return d

# ----------------------------- scanning logs -> per-run performance -----------------------------
def scan_one_log(path: Path, verbose=False) -> Optional[Dict[str, Any]]:
    txt = _read_text(path)
    if txt is None:
        return None

    # Prefer header if present; fallback to filename
    meta = _extract_meta_block(txt)
    fname_meta = _parse_from_fname(path.name)

    dataset = meta.get("dataset") or fname_meta.get("dataset")
    ver     = meta.get("ver")      or fname_meta.get("ver")
    seed    = meta.get("seed")     if "seed" in meta else fname_meta.get("seed")
    idx     = meta.get("idx")      if "idx" in meta else fname_meta.get("idx")

    kind, scope, remove_frac_hint = _parse_ver_scope_and_frac(ver or "")

    alpha   = meta.get("alpha", fname_meta.get("alpha"))
    gamma   = meta.get("gamma", fname_meta.get("gamma"))
    c0p_pr  = fname_meta.get("c0p_prune")
    aug_r   = fname_meta.get("aug_ratio")
    aug_b   = fname_meta.get("aug_bound")
    deg_thr = fname_meta.get("deg_thr")
    cm      = fname_meta.get("cm")

    h3, h10 = _parse_hitk_from_text(txt)

    if dataset is None or ver is None:
        return None

    if verbose:
        print(f"[PARSE] {path.name}: ds={dataset} ver={ver} scope={scope} frac={remove_frac_hint} "
              f"seed={seed} idx={idx} Hit@3={h3} Hit@10={h10}")

    return {
        "dataset": dataset,
        "ver": ver,
        "kind": kind,
        "scope": scope,
        "remove_frac_hint": remove_frac_hint,
        "alpha": alpha,
        "gamma": gamma,
        "c0p_prune_frac": c0p_pr,
        "aug_ratio": aug_r,
        "aug_bound": aug_b,
        "deg_thr": deg_thr,
        "seed": seed,
        "idx": idx,
        "cluster_method": cm,
        "log_file": str(path),
        "test_hit3": h3,
        "test_hit10": h10,
    }

def load_perf_from_logs(log_dirs: List[str], glob_pattern: str, verbose=False) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    total = 0
    for d in log_dirs:
        base = Path(d)
        if not base.exists():
            print(f"[WARN] log dir not found: {d}")
            continue
        hits = list(base.rglob(glob_pattern))
        total += len(hits)
        if verbose:
            print(f"[SCAN] {d}: {len(hits)} files")
        for p in sorted(hits):
            r = scan_one_log(p, verbose=verbose)
            if r is not None:
                rows.append(r)
    if not rows:
        print("[FATAL] No logs parsed.", file=sys.stderr)
        sys.exit(1)
    perf = pd.DataFrame(rows)
    # de-dup (keep last) just in case
    perf = perf.sort_values("log_file").drop_duplicates(subset=["dataset","ver","seed","idx"], keep="last")
    print(f"[OK] parsed perf rows: {len(perf)} (from {total} files)")
    return perf

# ----------------------------- radius summary merge -----------------------------
def load_radius_summary(tsv_path: str) -> pd.DataFrame:
    df = pd.read_csv(tsv_path, sep="\t")
    # Expect columns like: dataset, ver, seed, idx, end_radius, delta_radius, total_removed, last_added, last_removed, ...
    # Clean types
    for col in ("seed","idx"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    for c in ("end_radius","delta_radius","total_removed","last_added","last_removed"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # Some ver in summary might not carry _fracX; keep as-is (we join on ver exactly)
    print(f"[OK] loaded radius summary rows: {len(df)} from {tsv_path}")
    return df

# ----------------------------- correlation helpers -----------------------------
def _pearson(x: pd.Series, y: pd.Series) -> Tuple[Optional[float], Optional[float], int]:
    s = pd.concat([x, y], axis=1).dropna()
    n = len(s)
    if n < 3:
        return (np.nan, np.nan, n)
    r = s.corr(method="pearson").iloc[0,1]
    if HAVE_SCIPY:
        pr, pp = pearsonr(s.iloc[:,0], s.iloc[:,1])
        return (float(pr), float(pp), n)
    else:
        return (float(r), np.nan, n)

def corr_table(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    rows = []
    for key, sub in df.groupby(group_cols, dropna=False):
        key_tup = key if isinstance(key, tuple) else (key,)
        rec = dict(zip(group_cols, key_tup))

        r1, p1, n1 = _pearson(sub["delta_radius"], sub["test_hit3"])
        r2, p2, n2 = _pearson(sub["delta_radius"], sub["test_hit10"])
        r3, p3, n3 = _pearson(sub["total_removed"], sub["delta_radius"])

        rec.update({
            "r(Δradius, Hit@3)": r1, "p1": p1,
            "r(Δradius, Hit@10)": r2, "p2": p2,
            "r(total_removed, Δradius)": r3, "p3": p3,
            "points": int(min(n1, n2, n3)),
        })
        rows.append(rec)
    out = pd.DataFrame(rows)
    return out.sort_values(group_cols).reset_index(drop=True)

def perf_table(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    rows = []
    for key, sub in df.groupby(group_cols, dropna=False):
        key_tup = key if isinstance(key, tuple) else (key,)
        rec = dict(zip(group_cols, key_tup))
        rec.update({
            "n_runs": len(sub),
            "mean_delta_radius": sub["delta_radius"].mean(skipna=True),
            "mean_total_removed": sub["total_removed"].mean(skipna=True),
            "test_hit3_mean": sub["test_hit3"].mean(skipna=True),
            "test_hit10_mean": sub["test_hit10"].mean(skipna=True),
        })
        rows.append(rec)
    out = pd.DataFrame(rows)
    return out.sort_values(group_cols).reset_index(drop=True)

# ----------------------------- main -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dirs", nargs="+", required=True, help="directories containing *.log")
    ap.add_argument("--glob", type=str, default="*.log")
    ap.add_argument("--radius-summary", type=str, required=True, help="remove_only_summary.tsv (tab-separated)")
    ap.add_argument("--dump-merged", type=str, default="remove_only_merged_per_run.csv")
    ap.add_argument("--corr_by_dataset", type=str, default="remove_only_corr_by_dataset.tsv")
    ap.add_argument("--corr_by_dataset_scope", type=str, default="remove_only_corr_by_dataset_scope.tsv")
    ap.add_argument("--perf_by_dataset_scope", type=str, default="remove_only_perf_by_dataset_scope.tsv")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    perf = load_perf_from_logs(args.log_dirs, args.glob, verbose=args.verbose)
    radius = load_radius_summary(args.radius_summary)

    # ---- merge per run (no pre-aggregation) ----
    key_cols = ["dataset","ver","seed","idx"]
    merged = perf.merge(radius[key_cols + ["end_radius","delta_radius","total_removed"]].drop_duplicates(key_cols),
                        on=key_cols, how="inner")
    print(f"[MERGE] merged rows: {len(merged)} (perf ∩ radius)")

    # sanity prints
    print("[HEAD] merged:")
    with pd.option_context("display.max_columns", 80, "display.width", 160):
        print(merged.head(8).to_string(index=False))

    # write merged (optional)
    merged.to_csv(args.dump_merged, index=False)
    print(f"[WRITE] {args.dump_merged}")

    # ---- correlations (per-run data; grouped only for reporting) ----
    c_ds = corr_table(merged, ["dataset"])
    c_dss = corr_table(merged, ["dataset","scope"])

    # ---- performance summary by dataset × scope (for the weekly readout) ----
    p_dss = perf_table(merged, ["dataset","scope"])

    # print to console (copy-paste ready)
    def _fmt(df):
        df2 = df.copy()
        for col in df2.columns:
            if df2[col].dtype.kind in "fc":
                df2[col] = df2[col].map(lambda x: f"{x:.6f}" if pd.notna(x) else "NaN")
        return df2

    print("\n==== Correlations (Pearson r) by dataset ====")
    print(_fmt(c_ds[["dataset","r(Δradius, Hit@3)","r(Δradius, Hit@10)","r(total_removed, Δradius)","points"]]).to_string(index=False))

    print("\n==== Correlations (Pearson r) by dataset × scope ====")
    print(_fmt(c_dss[["dataset","scope","r(Δradius, Hit@3)","r(Δradius, Hit@10)","r(total_removed, Δradius)","points"]]).to_string(index=False))

    print("\n==== Per-dataset × scope: performance & compactness (means over runs) ====")
    print(_fmt(p_dss[["dataset","scope","n_runs","mean_delta_radius","mean_total_removed","test_hit3_mean","test_hit10_mean"]]).to_string(index=False))

    # save tsvs (nice for later collation)
    c_ds.to_csv(args.corr_by_dataset, sep="\t", index=False)
    c_dss.to_csv(args.corr_by_dataset_scope, sep="\t", index=False)
    p_dss.to_csv(args.perf_by_dataset_scope, sep="\t", index=False)
    print(f"\n[WRITE] {args.corr_by_dataset}")
    print(f"[WRITE] {args.corr_by_dataset_scope}")
    print(f"[WRITE] {args.perf_by_dataset_scope}")

    if not HAVE_SCIPY:
        print("\n[NOTE] scipy not found; p-values omitted. Install scipy to include them.", file=sys.stderr)

if __name__ == "__main__":
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 80)
    main()
