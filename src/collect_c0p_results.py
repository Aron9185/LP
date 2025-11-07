
#!/usr/bin/env python3
import argparse, re, json, os, sys
from pathlib import Path
from typing import Dict, Any, List
import statistics as stats
import csv
import math

# Broader metric patterns (prefer test-specific before generic).
PATTERNS = {
    # "best hit@3 epoch = 241, hit@3 = 0.6679 ..."
    "best@3": [
        re.compile(r"best\s+hit@?3[^=]*=\s*([0-9]*\.?[0-9]+)", re.IGNORECASE),
        re.compile(r"best@3\s*[:=]\s*([0-9]*\.?[0-9]+)", re.IGNORECASE),
        re.compile(r"best_at_3\s*[:=]\s*([0-9]*\.?[0-9]+)", re.IGNORECASE),
    ],
    # "best hit@10 epoch = 264, hit@10 = 0.7495 ..."
    "best@10": [
        re.compile(r"best\s+hit@?10[^=]*=\s*([0-9]*\.?[0-9]+)", re.IGNORECASE),
        re.compile(r"best@10\s*[:=]\s*([0-9]*\.?[0-9]+)", re.IGNORECASE),
        re.compile(r"best_at_10\s*[:=]\s*([0-9]*\.?[0-9]+)", re.IGNORECASE),
    ],
    # "[FINAL TEST] Hit@K: 1=..., 3=0.5806, 10=0.7191, ..."
    "test@3": [
        re.compile(r"\[FINAL\s+TEST\][^\n]*\b3\s*=\s*([0-9]*\.?[0-9]+)", re.IGNORECASE),
        re.compile(r"test@3\s*[:=]\s*([0-9]*\.?[0-9]+)", re.IGNORECASE),
        # very loose fallback if your format drifts
        re.compile(r"hit@?3[^=\n]*=\s*([0-9]*\.?[0-9]+)", re.IGNORECASE),
    ],
    "test@10": [
        re.compile(r"\[FINAL\s+TEST\][^\n]*\b10\s*=\s*([0-9]*\.?[0-9]+)", re.IGNORECASE),
        re.compile(r"test@10\s*[:=]\s*([0-9]*\.?[0-9]+)", re.IGNORECASE),
        re.compile(r"hit@?10[^=\n]*=\s*([0-9]*\.?[0-9]+)", re.IGNORECASE),
    ],
    # "[FINAL TEST] test_roc = 0.96234, test_ap = 0.96440"
    "test_roc": [
        re.compile(r"\[FINAL\s+TEST\][^\n]*test_roc\s*=\s*([0-9]*\.?[0-9]+)", re.IGNORECASE),
        re.compile(r"(?:test_roc|roc[_\s]?auc)\s*[:=]\s*([0-9]*\.?[0-9]+)", re.IGNORECASE),
    ],
    "test_ap": [
        re.compile(r"\[FINAL\s+TEST\][^\n]*test_ap\s*=\s*([0-9]*\.?[0-9]+)", re.IGNORECASE),
        re.compile(r"(?:test_ap|average[_\s]?precision)\s*[:=]\s*([0-9]*\.?[0-9]+)", re.IGNORECASE),
    ],
}

# More permissive filename regex: allow multiple underscores in ver (aron_desc_intra, etc.).
FNAME_RE = re.compile(
    r"(?P<ds>[^_/]+)_(?P<ver>aron_[a-z0-9_]+)_a(?P<alpha>\d+\.\d+)_g(?P<gamma>\d+\.\d+)_c0p(?P<c0p>\d+\.\d+)_seed(?P<seed>\d+)",
    re.IGNORECASE
)

def tail_metrics(path: Path) -> Dict[str, float]:
    text = path.read_text(errors="ignore")
    out = {}
    for key, regs in PATTERNS.items():
        last_val = None
        for rg in regs:
            for m in rg.finditer(text):
                last_val = m.group(1)
            if last_val is not None:
                break
        if last_val is not None:
            try:
                out[key] = float(last_val)
            except:
                pass
    return out

def parse_name(path: Path) -> Dict[str, Any]:
    m = FNAME_RE.search(path.name)
    if not m:
        return {}
    g = m.groupdict()
    g["alpha"] = float(g["alpha"])
    g["gamma"] = float(g["gamma"])
    g["c0p"] = float(g["c0p"])
    g["seed"] = int(g["seed"])
    return g

def mean_std(xs: List[float]):
    if not xs:
        return (math.nan, math.nan)
    if len(xs) == 1:
        return (xs[0], 0.0)
    return (sum(xs)/len(xs), stats.pstdev(xs))

def main():
    ap = argparse.ArgumentParser(description="Collect and aggregate C0p sweep results from logs.")
    ap.add_argument("--log_glob", type=str, required=True, help="Glob like 'logs/1103/Cora_aron_desc_intra_*.log'")
    ap.add_argument("--out_csv", type=str, default="c0p_sweep_summary.csv")
    ap.add_argument("--primary_metric", type=str, default="test@10", choices=["test@10","test@3","best@10","best@3","test_roc","test_ap"])
    args = ap.parse_args()

    paths = sorted(Path().glob(args.log_glob))
    if not paths:
        print(f"[COLLECT] No logs matched: {args.log_glob}")
        sys.exit(2)

    print(f"[COLLECT] Found {len(paths)} logs. Parsing...")

    rows = []
    for p in paths:
        meta = parse_name(p)
        if not meta:
            print(f"[SKIP] Unrecognized filename: {p.name}")
            continue
        metrics = tail_metrics(p)
        row = { **meta, **metrics, "file": str(p) }
        rows.append(row)

    if not rows:
        print("[COLLECT] No logs had recognizable names. Fix FNAME_RE or filenames.")
        sys.exit(3)

    # Group by (dataset, ver, alpha, gamma, c0p)
    from collections import defaultdict
    groups = defaultdict(list)
    for r in rows:
        key = (r.get("ds"), r.get("ver"), r.get("alpha"), r.get("gamma"), r.get("c0p"))
        groups[key].append(r)

    agg_rows = []
    for (ds, ver, alpha, gamma, c0p), items in groups.items():
        def take(metric):
            vals = [x[metric] for x in items if metric in x]
            return mean_std(vals)
        best3_m, best3_s = take("best@3")
        best10_m, best10_s = take("best@10")
        test3_m, test3_s = take("test@3")
        test10_m, test10_s = take("test@10")
        roc_m, roc_s = take("test_roc")
        ap_m, ap_s = take("test_ap")

        agg_rows.append({
            "dataset": ds, "ver": ver, "alpha": alpha, "gamma": gamma, "c0p_prune_frac": c0p,
            "n_runs": len(items),
            "best@3_mean": best3_m, "best@3_std": best3_s,
            "best@10_mean": best10_m, "best@10_std": best10_s,
            "test@3_mean": test3_m, "test@3_std": test3_s,
            "test@10_mean": test10_m, "test@10_std": test10_s,
            "test_roc_mean": roc_m, "test_roc_std": roc_s,
            "test_ap_mean": ap_m, "test_ap_std": ap_s,
        })

    pm = args.primary_metric + "_mean"
    def keyfunc(r):
        v = r.get(pm)
        try:
            return (v if v==v else -1.0)  # NaN-safe
        except:
            return -1.0
    agg_rows.sort(key=keyfunc, reverse=True)

    outp = Path(args.out_csv)
    with outp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(agg_rows[0].keys()))
        w.writeheader()
        for r in agg_rows:
            w.writerow(r)

    print(f"[COLLECT] Wrote summary: {outp}")
    print(f"[COLLECT] Top by {args.primary_metric}:")
    top = agg_rows[:10]
    for i, r in enumerate(top, 1):
        print(f"{i:2d}. ds={r['dataset']} ver={r['ver']} a={r['alpha']:.2f} g={r['gamma']:.2f} c0p={r['c0p_prune_frac']:.2f} "
              f"| {args.primary_metric}={r.get(pm)}  (n={r['n_runs']})")

if __name__ == "__main__":
    main()
