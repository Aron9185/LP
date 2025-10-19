#!/usr/bin/env python3
"""
summarize_hits.py

Usage:
  python summarize_hits.py path/to/log1.log path/to/log2.log ... [--csv results.csv] [--ddof 1] [--debug]

Parses:
- best@3 / best@10 from:
    "best hit@3 epoch = ..., hit@3 = ..., val = ..."
    "best hit@10 epoch = ..., hit@10 = ..., val = ..."
- Final test ROC/AP from (prefer last occurrence):
    "[FINAL TEST] test_roc = 0.95606, test_ap = 0.96045"
  Fallback:
    "best link prediction epoch = ..., Val_roc = ..., val_ap = ..., test_roc = ..., test_ap = ..."
- Test Hit@K from either:
    "[FINAL TEST] Hit@K: 1=..., 3=..., 10=..., 20=..., 50=..., 100=..."
    "Hit@K for test: 1=..., 3=..., 10=..., 20=..., 50=..., 100=..."
"""

import argparse
import csv
import re
import statistics
from pathlib import Path
from typing import Dict, Optional, Tuple, List

# Robust float (accepts 1, 1., .5, 1e-3, -0.2, etc.)
FLOAT = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"

# Best lines (tolerant to spaces/case)
BEST_H3_RE  = re.compile(r"best\s+hit@3\s+epoch\s*=\s*\d+,\s*hit@3\s*=\s*(" + FLOAT + r")", re.IGNORECASE)
BEST_H10_RE = re.compile(r"best\s+hit@10\s+epoch\s*=\s*\d+,\s*hit@10\s*=\s*(" + FLOAT + r")", re.IGNORECASE)

# New final ROC/AP line (multiline to match line starts)
FINAL_ROCAP_LINE_RE = re.compile(
    r"^\[FINAL\s*TEST\][^\n]*?test_roc\s*=\s*(" + FLOAT + r")\s*,\s*test_ap\s*=\s*(" + FLOAT + r")",
    re.IGNORECASE | re.MULTILINE,
)

# Fallback: older one-line with test_roc/test_ap embedded
BEST_LINK_PRED_RE = re.compile(
    r"best\s+link\s+prediction\s+epoch[^\n]*?test_roc\s*=\s*(" + FLOAT + r")[^\n]*?test_ap\s*=\s*(" + FLOAT + r")",
    re.IGNORECASE,
)

# Hit@K lines (new + old)
TEST_LINE_FINAL_RE = re.compile(r"^\[FINAL\s*TEST\]\s*Hit@K:\s*(.+)$", re.IGNORECASE | re.MULTILINE)
TEST_LINE_OLD_RE   = re.compile(r"Hit@K\s+for\s+test:\s*(.+)", re.IGNORECASE)

# K=V pairs (e.g., "10=0.7741935483")
KV_PAIR_RE = re.compile(r"(\d+)\s*=\s*(" + FLOAT + r")")

def parse_log(path: Path, debug: bool=False) -> Dict[str, Optional[float]]:
    """
    Returns dict with keys:
      best_hit3, best_hit10, test_hit3, test_hit10, test_roc, test_ap
    Any missing value is None.
    """
    best_hit3 = None
    best_hit10 = None
    test_hit3 = None
    test_hit10 = None
    test_roc = None
    test_ap = None

    last_test_map: Dict[int, float] = {}

    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        print(f"[WARN] Failed to read {path}: {e}")
        return {k: None for k in ("best_hit3","best_hit10","test_hit3","test_hit10","test_roc","test_ap")}

    # --- Best@3 / Best@10 ---
    m3 = BEST_H3_RE.search(text)
    if m3:
        try:
            best_hit3 = float(m3.group(1))
            if debug: print(f"[DEBUG] {path.name}: best_hit3={best_hit3}")
        except ValueError:
            pass

    m10 = BEST_H10_RE.search(text)
    if m10:
        try:
            best_hit10 = float(m10.group(1))
            if debug: print(f"[DEBUG] {path.name}: best_hit10={best_hit10}")
        except ValueError:
            pass

    # --- Final ROC/AP ---
    # Prefer the explicit [FINAL TEST] line; take the last one if multiple
    rocap_matches = FINAL_ROCAP_LINE_RE.findall(text)
    if rocap_matches:
        try:
            roc_str, ap_str = rocap_matches[-1]
            test_roc = float(roc_str)
            test_ap  = float(ap_str)
            if debug: print(f"[DEBUG] {path.name}: FINAL_TEST roc={test_roc}, ap={test_ap}")
        except ValueError:
            pass
    else:
        # Fallback to the "best link prediction epoch ..., test_roc=..., test_ap=..." line
        lp_matches = BEST_LINK_PRED_RE.findall(text)
        if lp_matches:
            try:
                roc_str, ap_str = lp_matches[-1]
                test_roc = float(roc_str)
                test_ap  = float(ap_str)
                if debug: print(f"[DEBUG] {path.name}: LINK_PRED roc={test_roc}, ap={test_ap}")
            except ValueError:
                pass

    # --- Hit@K (prefer new, fallback to old) ---
    payload = None
    final_lines = TEST_LINE_FINAL_RE.findall(text)
    old_lines = TEST_LINE_OLD_RE.findall(text)
    if final_lines:
        payload = final_lines[-1]
        if debug: print(f"[DEBUG] {path.name}: Hit@K payload (FINAL)='{payload}'")
    elif old_lines:
        payload = old_lines[-1]
        if debug: print(f"[DEBUG] {path.name}: Hit@K payload (OLD)='{payload}'")

    if payload:
        for k, v in KV_PAIR_RE.findall(payload):
            try:
                last_test_map[int(k)] = float(v)
            except ValueError:
                continue
        test_hit3  = last_test_map.get(3, None)
        test_hit10 = last_test_map.get(10, None)
        if debug: print(f"[DEBUG] {path.name}: test_hit3={test_hit3}, test_hit10={test_hit10}")

    return {
        "best_hit3": best_hit3,
        "best_hit10": best_hit10,
        "test_hit3": test_hit3,
        "test_hit10": test_hit10,
        "test_roc": test_roc,
        "test_ap": test_ap,
    }

def mean_and_var(values: List[Optional[float]], ddof: int) -> Tuple[Optional[float], Optional[float]]:
    vals = [v for v in values if v is not None]
    if not vals:
        return None, None
    if len(vals) == 1:
        return float(vals[0]), 0.0
    mu = statistics.fmean(vals)
    var = statistics.variance(vals) if ddof == 1 else statistics.pvariance(vals)
    return mu, var

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+", help="Paths to log files to summarize.")
    ap.add_argument("--csv", type=str, default="", help="Optional path to write per-file metrics as CSV.")
    ap.add_argument("--ddof", type=int, default=1, choices=[0,1], help="0=population variance, 1=sample variance (default).")
    ap.add_argument("--debug", action="store_true", help="Print regex matches for troubleshooting.")
    args = ap.parse_args()

    rows = []
    for p in args.logs:
        path = Path(p)
        metrics = parse_log(path, debug=args.debug)
        rows.append((path.name, metrics))

    # Collect columns
    best3  = [m["best_hit3"] for _, m in rows]
    best10 = [m["best_hit10"] for _, m in rows]
    test3  = [m["test_hit3"] for _, m in rows]
    test10 = [m["test_hit10"] for _, m in rows]
    roc    = [m["test_roc"]  for _, m in rows]
    apv    = [m["test_ap"]   for _, m in rows]

    # Print per-file
    print("\nPer-file metrics:")
    header = f"{'file':<40}  {'best@3':>10}  {'best@10':>10}  {'test@3':>10}  {'test@10':>10}  {'test_roc':>10}  {'test_ap':>10}"
    print(header)
    for fname, m in rows:
        def fmt(x): return f"{x:.6f}" if isinstance(x, float) else ("NA" if x is None else str(x))
        print(f"{fname:<40}  "
              f"{fmt(m['best_hit3']):>10}  "
              f"{fmt(m['best_hit10']):>10}  "
              f"{fmt(m['test_hit3']):>10}  "
              f"{fmt(m['test_hit10']):>10}  "
              f"{fmt(m['test_roc']):>10}  "
              f"{fmt(m['test_ap']):>10}")

    # Aggregates
    def show(label, vals):
        mu, var = mean_and_var(vals, args.ddof)
        count = len([v for v in vals if v is not None])
        print(f"{label:<12}  count={count:<3}  mean={mu if mu is not None else 'NA'}  var={var if var is not None else 'NA'}")

    print("\nAggregates (ddof={}):".format(args.ddof))
    show("best@3", best3)
    show("best@10", best10)
    show("test@3", test3)
    show("test@10", test10)
    show("test_roc", roc)
    show("test_ap", apv)

    # Optional CSV
    if args.csv:
        outp = Path(args.csv)
        with outp.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["file", "best_hit3", "best_hit10", "test_hit3", "test_hit10", "test_roc", "test_ap"])
            for fname, m in rows:
                w.writerow([
                    fname,
                    m["best_hit3"] if m["best_hit3"] is not None else "",
                    m["best_hit10"] if m["best_hit10"] is not None else "",
                    m["test_hit3"] if m["test_hit3"] is not None else "",
                    m["test_hit10"] if m["test_hit10"] is not None else "",
                    m["test_roc"] if m["test_roc"] is not None else "",
                    m["test_ap"] if m["test_ap"] is not None else "",
                ])
        print(f"\nWrote CSV -> {outp}")

if __name__ == "__main__":
    main()
