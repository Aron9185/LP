#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, sys, re, math
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd

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

# --- helpers ---

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

def _extract_meta(text: str) -> Dict[str, Any]:
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

    if "mode" not in meta and "ver" in meta:
        if "inter" in meta["ver"]:
            meta["mode"] = "inter"
        elif "intra" in meta["ver"]:
            meta["mode"] = "intra"

    for k in ("seed", "idx"):
        if k in meta:
            try: meta[k] = int(meta[k])
            except: pass
    for k in ("alpha", "gamma"):
        if k in meta:
            try: meta[k] = float(meta[k])
            except: pass
    return meta

# --- robust metric parsing focused on your exact formats ---

RE_FINAL_HITK_LINE = re.compile(r"\[FINAL TEST\][^\n]*Hit@K\s*:\s*(.+)", re.I)
RE_PAIR = re.compile(r"(\d+)\s*=\s*([0-9]*\.?[0-9]+)")

RE_BEST_HIT3 = re.compile(r"best\s+hit@?3\s+epoch\s*=\s*\d+\s*,\s*hit@?3\s*=\s*([0-9]*\.?[0-9]+)", re.I)
RE_BEST_HIT10 = re.compile(r"best\s+hit@?10\s+epoch\s*=\s*\d+\s*,\s*hit@?10\s*=\s*([0-9]*\.?[0-9]+)", re.I)

def _parse_final_hitk(text: str) -> Tuple[Optional[float], Optional[float]]:
    """Prefer the last '[FINAL TEST] Hit@K: 1=..., 3=..., 10=...' line."""
    lines = RE_FINAL_HITK_LINE.findall(text)
    if not lines:
        return None, None
    payload = lines[-1]  # last occurrence
    pairs = dict((k, _to_fraction(v)) for k, v in RE_PAIR.findall(payload))
    h3 = pairs.get("3")
    h10 = pairs.get("10")
    return h3, h10

def _parse_best_hitk(text: str) -> Tuple[Optional[float], Optional[float]]:
    m3 = RE_BEST_HIT3.findall(text)
    m10 = RE_BEST_HIT10.findall(text)
    h3 = _to_fraction(m3[-1]) if m3 else None
    h10 = _to_fraction(m10[-1]) if m10 else None
    return h3, h10

def _parse_test_hits_from_text(text: str, verbose: bool=False) -> Tuple[Optional[float], Optional[float]]:
    # 1) Strictly prefer [FINAL TEST] Hit@K: ... (most unambiguous)
    h3, h10 = _parse_final_hitk(text)

    # 2) Fall back to 'best hit@k epoch = ..., hit@k = ...'
    if h3 is None or h10 is None:
        b3, b10 = _parse_best_hitk(text)
        if h3 is None: h3 = b3
        if h10 is None: h10 = b10

    if verbose:
        print(f"[DEBUG][PREFER_FINAL] final: h3={h3} h10={h10}")
    return h3, h10

# --- scanning & aggregation ---

def scan_one_log(path: Path, verbose=False) -> Optional[Dict[str, Any]]:
    txt = _read_text(path)
    if txt is None:
        return None

    meta = _extract_meta(txt)
    if not meta.get("dataset") or not meta.get("mode"):
        if verbose:
            print(f"[SKIP] {path.name}: missing dataset/mode in header")
        return None

    hit3, hit10 = _parse_test_hits_from_text(txt, verbose=verbose)

    if verbose:
        print(f"[PARSE] {path.name}: ds={meta.get('dataset')} "
              f"mode={meta.get('mode')} alpha={meta.get('alpha')} "
              f"seed={meta.get('seed')} idx={meta.get('idx')} "
              f"Hit@3={hit3} Hit@10={hit10}")

    return {
        "dataset": meta.get("dataset"),
        "mode": meta.get("mode"),
        "alpha": meta.get("alpha"),
        "gamma": meta.get("gamma"),
        "seed": meta.get("seed"),
        "idx": meta.get("idx"),
        "run_tag": meta.get("run_tag"),
        "log_file": str(path),
        "test_hit3": hit3,
        "test_hit10": hit10,
    }

def aggregate(df: pd.DataFrame, by_cols: List[str]) -> pd.DataFrame:
    def _agg(sub: pd.DataFrame, col: str) -> pd.Series:
        s = sub[col]
        n = int(s.notna().sum())
        return pd.Series({
            f"{col}_mean": s.mean(skipna=True),
            f"{col}_var":  s.var(skipna=True, ddof=1) if n > 1 else math.nan,
            f"{col}_std":  s.std(skipna=True, ddof=1) if n > 1 else math.nan,
            f"{col}_n_eff": float(n),
        })

    rows = []
    for key, sub in df.groupby(by_cols, dropna=False):
        rec = dict(zip(by_cols, key if isinstance(key, tuple) else (key,)))
        rec.update(_agg(sub, "test_hit3").to_dict())
        rec.update(_agg(sub, "test_hit10").to_dict())
        rec["n_runs"] = float(len(sub))
        rows.append(rec)

    out = pd.DataFrame(rows)
    cols = by_cols + [
        "n_runs",
        "test_hit3_mean","test_hit3_var","test_hit3_std","test_hit3_n_eff",
        "test_hit10_mean","test_hit10_var","test_hit10_std","test_hit10_n_eff",
    ]
    cols = [c for c in cols if c in out.columns]
    return out.sort_values(by=by_cols).reset_index(drop=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", type=str, default=None)
    ap.add_argument("--log-dirs", nargs="+", default=None)
    ap.add_argument("--glob", type=str, default="*.log")
    ap.add_argument("--dump-parsed", type=str, default="c0p_parsed.csv")
    ap.add_argument("--out", type=str, default="c0p_agg_by_dataset_mode.csv")
    ap.add_argument("--out-by-alpha", type=str, default="c0p_agg_by_dataset_mode_alpha.csv")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    parsed_rows: List[Dict[str, Any]] = []

    total_files = 0
    if args.log_dirs:
        for d in args.log_dirs:
            base = Path(d)
            if not base.exists():
                print(f"[WARN] log dir not found: {d}")
                continue
            hits = list(base.rglob(args.glob))
            total_files += len(hits)
            if args.verbose:
                print(f"[SCAN] {d}: {len(hits)} files matching {args.glob}")
            for p in sorted(hits):
                row = scan_one_log(p, verbose=args.verbose)
                if row is not None:
                    parsed_rows.append(row)
    else:
        print("[WARN] No --log-dirs provided; nothing to parse.")

    if len(parsed_rows) == 0:
        print("[FATAL] No parsed rows from logs. Check --log-dirs / --glob.", file=sys.stderr)
        sys.exit(1)

    parsed = pd.DataFrame(parsed_rows)
    parsed.to_csv(args.dump_parsed, index=False)
    print(f"[WRITE] {args.dump_parsed} ({len(parsed)} rows)")
    n3 = parsed["test_hit3"].notna().sum()
    n10 = parsed["test_hit10"].notna().sum()
    print(f"[STATS] files scanned={total_files} | parsed rows={len(parsed)} | with Hit@3={n3} | with Hit@10={n10}")

    by_ds_mode = aggregate(parsed, ["dataset","mode"])
    by_ds_mode.to_csv(args.out, index=False)
    print(f"[WRITE] {args.out} ({len(by_ds_mode)} rows)")

    by_ds_mode_alpha = aggregate(parsed, ["dataset","mode","alpha"])
    by_ds_mode_alpha.to_csv(args.out_by_alpha, index=False)
    print(f"[WRITE] {args.out_by_alpha} ({len(by_ds_mode_alpha)} rows)")

    if args.verbose:
        print("\n[PREVIEW] agg by dataset+mode:")
        print(by_ds_mode.to_string(index=False))
        print("\n[PREVIEW] agg by dataset+mode+alpha:")
        print(by_ds_mode_alpha.to_string(index=False))

if __name__ == "__main__":
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 40)
    main()
