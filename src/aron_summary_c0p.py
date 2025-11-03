#!/usr/bin/env python3
import argparse, csv, glob, re, statistics, sys
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from collections import defaultdict

# ------------------------- helpers -------------------------
FLOAT = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"

# Prefer capturing the explicit "hit@K = <value>" that appears on the same
# 'best ...' line (e.g., "best hit@3 epoch = 153, hit@3 = 0.6923, ...")
def BEST_HIT_VALUE_RE(k: int):
    return re.compile(
        rf"best[^\n]*?\bhit\s*@\s*{k}\b[^\n]*?\bhit\s*@\s*{k}\b\s*=\s*({FLOAT})",
        re.IGNORECASE,
    )

# Simple fallback where logs are "best hit@K = <value>"
def BEST_HIT_SIMPLE_RE(k: int):
    return re.compile(
        rf"best\s*hit\s*@\s*{k}\b[^\n]*?=\s*({FLOAT})",
        re.IGNORECASE,
    )

FINAL_ROCAP_LINE_RE = re.compile(
    r"^\[FINAL\s*TEST\][^\n]*?test_roc\s*=\s*(" + FLOAT + r")\s*,\s*test_ap\s*=\s*(" + FLOAT + r")",
    re.IGNORECASE | re.MULTILINE,
)
BEST_LINK_PRED_RE = re.compile(
    r"best\s+link\s+prediction\s+epoch[^\n]*?test_roc\s*=\s*(" + FLOAT + r")[^\n]*?test_ap\s*=\s*(" + FLOAT + r")",
    re.IGNORECASE,
)

# Final Hit@K line variants
TEST_LINE_FINAL_RE = re.compile(r"^\[FINAL\s*TEST\][^\n]*Hit\s*@\s*K:\s*(.+)$", re.IGNORECASE | re.MULTILINE)
TEST_LINE_OLD_RE   = re.compile(r"Hit\s*@\s*K\s*(?:for\s+test)?\s*:\s*(.+)", re.IGNORECASE)

# generic KV within Hit@K: "3=0.712, 5=..., 10=..."
KV_PAIR_RE = re.compile(r"(\d+)\s*=\s*(" + FLOAT + r")")

ALPHA_RE = re.compile(r"\balpha\s*=\s*(" + FLOAT + r")", re.IGNORECASE)
GAMMA_RE = re.compile(r"\bgamma\s*=\s*(" + FLOAT + r")", re.IGNORECASE)

# ------------------------- filename meta -------------------------
def parse_meta_from_filename(name: str) -> Dict[str, Optional[str]]:
    stem = name[:-4] if name.endswith(".log") else name
    parts = stem.split("_")
    meta = {
        "dataset": None, "setting": None, "cluster": None,
        "r": None, "fmr": None, "d": None, "seed": None, "idx": None,
        "taga": None, "g": None, "ag_tag": None,
    }
    if not parts:
        return meta

    # dataset: first token (e.g., citeseer, cora)
    meta["dataset"] = parts[0]

    # simply look for "inter" / "intra" anywhere
    for p in parts:
        if p in ("inter", "intra"):
            meta["setting"] = p
            break

    for p in parts:
        if re.fullmatch(r"r[0-9.]+", p):
            meta["r"] = p[1:]
        elif p.startswith("fmr") and re.fullmatch(r"fmr[0-9.]+", p):
            meta["fmr"] = p[3:]
        # STRICT: only accept 'd' followed by numeric (avoid 'desc' -> 'esc')
        elif re.fullmatch(r"d[0-9.]+", p):
            meta["d"] = p[1:]
        elif p in ("gmm", "louvain"):
            meta["cluster"] = p
        elif p.startswith("seed") and p[4:].isdigit():
            meta["seed"] = p[4:]
        elif p.startswith("idx") and p[3:].isdigit():
            meta["idx"] = p[3:]
        elif p.startswith("taga") and p[4:]:
            meta["taga"] = p[4:]   # alpha (string)
        elif p.startswith("g") and len(p) > 1 and re.fullmatch(r"[0-9.]+", p[1:]):
            meta["g"] = p[1:]     # gamma (string)
        elif p.startswith("a") and len(p) > 1 and re.fullmatch(r"[0-9.]+", p[1:]) and meta["taga"] is None:
            meta["taga"] = p[1:]  # accept a0.80

    return meta

def _parse_alpha_gamma_from_name(stem: str):
    a = g = None
    for p in stem.split("_"):
        if p.startswith("taga") and p[4:] and re.fullmatch(r"[0-9.]+", p[4:]):
            try: a = float(p[4:])
            except: pass
        elif p.startswith("a") and p[1:] and re.fullmatch(r"[0-9.]+", p[1:]):
            try: a = float(p[1:])
            except: pass
        elif p.startswith("gamma") and p[5:] and re.fullmatch(r"[0-9.]+", p[5:]):
            try: g = float(p[5:])
            except: pass
        elif p.startswith("g") and p[1:] and re.fullmatch(r"[0-9.]+", p[1:]):
            try: g = float(p[1:])
            except: pass
    return a, g

def _parse_alpha_gamma_from_text(text: str):
    a = g = None
    ma = ALPHA_RE.search(text)
    mg = GAMMA_RE.search(text)
    if ma:
        try: a = float(ma.group(1))
        except: pass
    if mg:
        try: g = float(mg.group(1))
        except: pass
    return a, g

# ------------------------- log parsing -------------------------
def parse_log(path: Path, debug: bool=False) -> Dict[str, Optional[float]]:
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        print(f"[WARN] Failed to read {path}: {e}")
        return {k: None for k in ("best_hit3","best_hit10","test_hit3","test_hit10","test_roc","test_ap")}

    best_hit3 = best_hit10 = test_hit3 = test_hit10 = test_roc = test_ap = None

    # --- best@3 / best@10 (robust) ---
    def _best_hit(txt: str, k: int) -> Optional[float]:
        m = list(BEST_HIT_VALUE_RE(k).finditer(txt))
        if m:
            try: return float(m[-1].group(1))
            except: pass
        m = list(BEST_HIT_SIMPLE_RE(k).finditer(txt))
        if m:
            try: return float(m[-1].group(1))
            except: pass
        # Ultra-fallback: scan lines containing "best" and "@K" and pull the hit@K value on that line
        for line in reversed(txt.splitlines()):
            low = line.lower()
            if "best" in low and f"@{k}" in low:
                mm = re.search(rf"hit\s*@\s*{k}\s*=\s*({FLOAT})", line, re.IGNORECASE)
                if mm:
                    try: return float(mm.group(1))
                    except: pass
        return None

    best_hit3  = _best_hit(text, 3)
    best_hit10 = _best_hit(text, 10)

    # --- ROC/AP ---
    rocap = FINAL_ROCAP_LINE_RE.findall(text)
    if rocap:
        try:
            test_roc = float(rocap[-1][0]); test_ap = float(rocap[-1][1])
        except: pass
    else:
        lp = BEST_LINK_PRED_RE.findall(text)
        if lp:
            try:
                test_roc = float(lp[-1][0]); test_ap = float(lp[-1][1])
            except: pass

    # --- test Hit@K payload (numerous variants) ---
    payload = None
    finals = TEST_LINE_FINAL_RE.findall(text)
    olds   = TEST_LINE_OLD_RE.findall(text)
    if finals:
        payload = finals[-1]
    elif olds:
        payload = olds[-1]
    else:
        # last line containing "Hit@K" anywhere
        lines = [ln for ln in text.splitlines() if "hit@k" in ln.lower()]
        if lines:
            payload = lines[-1]

    if payload:
        last_map: Dict[int, float] = {}
        for k, v in KV_PAIR_RE.findall(payload):
            try: last_map[int(k)] = float(v)
            except: continue
        test_hit3  = last_map.get(3, test_hit3)
        test_hit10 = last_map.get(10, test_hit10)

    if debug:
        print(f"[DEBUG] {path.name}: best@3={best_hit3}, best@10={best_hit10}, test@3={test_hit3}, test@10={test_hit10}, roc={test_roc}, ap={test_ap}")

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
    if not vals: return None, None
    if len(vals) == 1: return float(vals[0]), 0.0
    mu = statistics.fmean(vals)
    var = statistics.variance(vals) if ddof == 1 else statistics.pvariance(vals)
    return mu, var

def _fmt_cell(x):
    if x is None: return ""
    return f"{x:.6f}" if isinstance(x, float) else str(x)

def _expand_globs(paths):
    out = []
    for patt in paths:
        hits = glob.glob(patt)
        if not hits:
            print(f"[WARN] No files match: {patt}")
        out.extend(sorted(hits))
    return out

# ------------------------- seed merge -------------------------
def _seed_merge(rows, ddof=1, debug=False):
    """
    Group by (dataset, setting, alpha, gamma) and compute mean/var/std per metric.
    Also report which files contributed no metrics to help debugging.
    """
    key_fields = ["dataset", "setting", "alpha", "gamma"]
    groups = defaultdict(list)
    for r in rows:
        key = tuple(r.get(k) for k in key_fields)
        groups[key].append(r)

    merged_rows = []
    for key, items in groups.items():
        def vals(col):
            return [x.get(col) for x in items if x.get(col) is not None]
        def agg(col):
            m, v = mean_and_var(vals(col), ddof)
            return m, v, (v ** 0.5) if v is not None else None

        # Check empties for this group (for user debug)
        empty_files = []
        for it in items:
            if all(it.get(c) is None for c in ("best_hit3","best_hit10","test_hit3","test_hit10","test_roc","test_ap")):
                empty_files.append(it.get("file"))
        if empty_files and debug:
            print(f"[DEBUG] Empty metrics for key={key}; files with missing metrics: {empty_files}")

        rec = {k: v for k, v in zip(key_fields, key)}
        rec["n_runs"] = len(items)

        for col, out_prefix in [
            ("best_hit3", "best@3"),
            ("best_hit10", "best@10"),
            ("test_hit3", "test@3"),
            ("test_hit10", "test@10"),
            ("test_roc", "test_roc"),
            ("test_ap",  "test_ap"),
        ]:
            m, v, s = agg(col)
            rec[f"{out_prefix}_mean"] = m
            rec[f"{out_prefix}_var"]  = v
            rec[f"{out_prefix}_std"]  = s

        merged_rows.append(rec)

    merged_rows.sort(key=lambda r: (
        str(r.get("dataset") or ""),
        str(r.get("setting") or ""),
        -(r.get("alpha") if isinstance(r.get("alpha"), (int,float)) else -1e9),
        -(r.get("gamma") if isinstance(r.get("gamma"), (int,float)) else -1e9),
    ))
    return merged_rows

# ------------------------- main -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+", help="Paths or globs to log files (e.g., logs/1030/*.log)")
    ap.add_argument("--csv", type=str, default="", help="Write per-file metrics + metadata CSV.")
    ap.add_argument("--ddof", type=int, default=1, choices=[0,1], help="0=population variance, 1=sample variance (default).")
    ap.add_argument("--debug", action="store_true", help="Verbose parse debugging and list empty-metric files per group.")
    ap.add_argument("--seed-merge", action="store_true", help="Show seed-merged table grouped by dataset × setting × alpha × gamma.")
    ap.add_argument("--seed-merge-csv", type=str, default="", help="CSV path for the seed-merged table.")
    ap.add_argument("--overall", action="store_true", help="Also print overall aggregates across all logs.")
    args = ap.parse_args()

    files = _expand_globs(args.logs)
    if not files:
        print("[ERR] No files to parse after glob expansion.")
        sys.exit(2)

    rows = []
    for fp in files:
        path = Path(fp)
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception as e:
            print(f"[WARN] Failed to read {fp}: {e}")
            continue

        meta = parse_meta_from_filename(path.name)
        stem = path.name[:-4] if path.name.endswith(".log") else path.name

        a_name, g_name = _parse_alpha_gamma_from_name(stem)
        a_txt,  g_txt  = _parse_alpha_gamma_from_text(text)
        alpha = a_txt if a_txt is not None else a_name
        gamma = g_txt if g_txt is not None else g_name

        metrics = parse_log(path, debug=args.debug)

        row = {
            "file": path.name,
            "dataset": meta.get("dataset"),
            "setting": meta.get("setting"),
            "cluster": meta.get("cluster"),
            "r": meta.get("r"),
            "fmr": meta.get("fmr"),
            "d": meta.get("d"),
            "seed": meta.get("seed"),
            "idx": meta.get("idx"),
            "taga": meta.get("taga"),
            "g": meta.get("g"),
            "alpha": alpha,
            "gamma": gamma,
            "ag_tag": meta.get("ag_tag") or (f"a{alpha}_g{gamma}" if alpha is not None and gamma is not None else None),
            **metrics,
        }
        rows.append(row)

    if not rows:
        print("[ERR] No rows parsed (all files failed to read?).")
        sys.exit(2)

    header_cols = ["file","dataset","setting","cluster","r","fmr","d","seed","idx","taga","g",
                   "alpha","gamma","ag_tag","best_hit3","best_hit10","test_hit3","test_hit10","test_roc","test_ap"]

    print("\nPer-file metrics:")
    print("  ".join(f"{c:>12}" if c != "file" else f"{c:<55}" for c in header_cols))
    for r in rows:
        print("  ".join([
            f"{_fmt_cell(r.get('file')):<55}",
            f"{_fmt_cell(r.get('dataset')):>12}",
            f"{_fmt_cell(r.get('setting')):>12}",
            f"{_fmt_cell(r.get('cluster')):>12}",
            f"{_fmt_cell(r.get('r')):>12}",
            f"{_fmt_cell(r.get('fmr')):>12}",
            f"{_fmt_cell(r.get('d')):>12}",
            f"{_fmt_cell(r.get('seed')):>12}",
            f"{_fmt_cell(r.get('idx')):>12}",
            f"{_fmt_cell(r.get('taga')):>12}",
            f"{_fmt_cell(r.get('g')):>12}",
            f"{_fmt_cell(r.get('alpha')):>12}",
            f"{_fmt_cell(r.get('gamma')):>12}",
            f"{_fmt_cell(r.get('ag_tag')):>12}",
            f"{_fmt_cell(r.get('best_hit3')):>12}",
            f"{_fmt_cell(r.get('best_hit10')):>12}",
            f"{_fmt_cell(r.get('test_hit3')):>12}",
            f"{_fmt_cell(r.get('test_hit10')):>12}",
            f"{_fmt_cell(r.get('test_roc')):>12}",
            f"{_fmt_cell(r.get('test_ap')):>12}",
        ]))

    if args.seed_merge:
        merged = _seed_merge(rows, ddof=args.ddof, debug=args.debug)
        mcols = [
            "dataset","setting","alpha","gamma","n_runs",
            "best@3_mean","best@3_var","best@3_std",
            "best@10_mean","best@10_var","best@10_std",
            "test@3_mean","test@3_var","test@3_std",
            "test@10_mean","test@10_var","test@10_std",
            "test_roc_mean","test_roc_var","test_roc_std",
            "test_ap_mean","test_ap_var","test_ap_std",
        ]
        print("\nSeed-merged by dataset × setting × alpha × gamma:")
        print("  ".join(f"{c:>14}" for c in mcols))
        for rec in merged:
            print("  ".join(f"{_fmt_cell(rec.get(c)):>14}" for c in mcols))

        if args.seed_merge_csv:
            with open(args.seed_merge_csv, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(mcols)
                for rec in merged:
                    w.writerow([rec.get(c, "") for c in mcols])
            print(f"\nWrote seed-merged CSV -> {args.seed_merge_csv}")

    if args.overall:
        def _vals(col): return [x.get(col) for x in rows]
        def _show(label, vals):
            mu, var = mean_and_var(vals, args.ddof)
            n = len([v for v in vals if v is not None])
            print(f"{label:<10}  count={n:<3}  mean={mu if mu is not None else 'NA'}  var={var if var is not None else 'NA'}")

        print("\n[Overall aggregates across all logs] (ddof={}):".format(args.ddof))
        _show("best@3",  _vals("best_hit3"))
        _show("best@10", _vals("best_hit10"))
        _show("test@3",  _vals("test_hit3"))
        _show("test@10", _vals("test_hit10"))
        _show("test_roc",_vals("test_roc"))
        _show("test_ap", _vals("test_ap"))

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(header_cols)
            for r in rows:
                w.writerow([r.get(c, "") for c in header_cols])
        print(f"\nWrote CSV -> {args.csv}")

if __name__ == "__main__":
    main()
