#!/usr/bin/env python3
"""
summarize_hits.py

Usage:
  python summarize_hits.py path/to/log1.log path/to/log2.log ... [--csv results.csv] [--ddof 1]

What it does:
- Extracts best_hit@3, best_hit@10 from lines like:
    "best hit@3 epoch = 675, hit@3 = 0.6717267552, val = ..."
    "best hit@10 epoch = 653, hit@10 = 0.7438330170, val = ..."
- Extracts test Hit@3, Hit@10 from the (last) line like:
    "Hit@K for test: 1=..., 3=..., 10=..., 20=..., 50=..., 100=..."

It then prints count, mean, and variance across all files, and (optionally) writes a CSV of per-file metrics.
"""

import argparse
import csv
import re
import statistics
from pathlib import Path
from typing import Dict, Optional, Tuple, List

# Regex for best lines (tolerant to spaces)
BEST_H3_RE = re.compile(r"best\s+hit@3\s+epoch\s*=\s*\d+,\s*hit@3\s*=\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)
BEST_H10_RE = re.compile(r"best\s+hit@10\s+epoch\s*=\s*\d+,\s*hit@10\s*=\s*([0-9]*\.?[0-9]+)", re.IGNORECASE)

# Regex for lines containing the test hit@K summary
TEST_LINE_RE = re.compile(r"Hit@K\s+for\s+test:\s*(.*)", re.IGNORECASE)
# Within that line, extract K=value pairs like "3=0.66"
KV_PAIR_RE = re.compile(r"(\d+)\s*=\s*([0-9]*\.?[0-9]+)")

def parse_log(path: Path) -> Dict[str, Optional[float]]:
    """
    Returns dict with keys:
      best_hit3, best_hit10, test_hit3, test_hit10
    Any missing value is None.
    """
    best_hit3 = None
    best_hit10 = None
    test_hit3 = None
    test_hit10 = None

    last_test_map: Dict[int, float] = {}

    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        print(f"[WARN] Failed to read {path}: {e}")
        return {"best_hit3": None, "best_hit10": None, "test_hit3": None, "test_hit10": None}

    # Best lines
    m3 = BEST_H3_RE.search(text)
    if m3:
        try:
            best_hit3 = float(m3.group(1))
        except ValueError:
            pass

    m10 = BEST_H10_RE.search(text)
    if m10:
        try:
            best_hit10 = float(m10.group(1))
        except ValueError:
            pass

    # We may have multiple "Hit@K for test:" lines; we want the LAST one
    test_lines = TEST_LINE_RE.findall(text)
    if test_lines:
        payload = test_lines[-1]  # last match contents after the colon
        for k, v in KV_PAIR_RE.findall(payload):
            try:
                k_i = int(k)
                last_test_map[k_i] = float(v)
            except ValueError:
                continue

        test_hit3 = last_test_map.get(3, None)
        test_hit10 = last_test_map.get(10, None)

    return {
        "best_hit3": best_hit3,
        "best_hit10": best_hit10,
        "test_hit3": test_hit3,
        "test_hit10": test_hit10,
    }

def mean_and_var(values: List[float], ddof: int) -> Tuple[Optional[float], Optional[float]]:
    vals = [v for v in values if v is not None]
    if not vals:
        return None, None
    if len(vals) == 1:
        # variance undefined for ddof=1 with a single sample; fall back to 0.0
        return float(vals[0]), 0.0
    mu = statistics.fmean(vals)
    # sample variance if ddof=1; population variance if ddof=0
    if ddof == 1:
        var = statistics.variance(vals)
    else:
        var = statistics.pvariance(vals)
    return mu, var

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+", help="Paths to log files to summarize.")
    ap.add_argument("--csv", type=str, default="", help="Optional path to write per-file metrics as CSV.")
    ap.add_argument("--ddof", type=int, default=1, choices=[0,1], help="0=population variance, 1=sample variance (default).")
    args = ap.parse_args()

    rows = []
    for p in args.logs:
        path = Path(p)
        metrics = parse_log(path)
        rows.append((path.name, metrics))

    # Collect columns
    best3 = [m["best_hit3"] for _, m in rows]
    best10 = [m["best_hit10"] for _, m in rows]
    test3 = [m["test_hit3"] for _, m in rows]
    test10 = [m["test_hit10"] for _, m in rows]

    # Print per-file
    print("\nPer-file metrics:")
    print(f"{'file':<40}  {'best@3':>10}  {'best@10':>10}  {'test@3':>10}  {'test@10':>10}")
    for fname, m in rows:
        print(f"{fname:<40}  "
              f"{(m['best_hit3'] if m['best_hit3'] is not None else 'NA'):>10}  "
              f"{(m['best_hit10'] if m['best_hit10'] is not None else 'NA'):>10}  "
              f"{(m['test_hit3'] if m['test_hit3'] is not None else 'NA'):>10}  "
              f"{(m['test_hit10'] if m['test_hit10'] is not None else 'NA'):>10}")

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

    # Optional CSV
    if args.csv:
        outp = Path(args.csv)
        with outp.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["file", "best_hit3", "best_hit10", "test_hit3", "test_hit10"])
            for fname, m in rows:
                w.writerow([fname,
                            m["best_hit3"] if m["best_hit3"] is not None else "",
                            m["best_hit10"] if m["best_hit10"] is not None else "",
                            m["test_hit3"] if m["test_hit3"] is not None else "",
                            m["test_hit10"] if m["test_hit10"] is not None else ""])
        print(f"\nWrote CSV -> {outp}")

if __name__ == "__main__":
    main()
