#!/usr/bin/env python3
# aron_added_edges_summary.py
import argparse, csv, glob, os, re, sys
from collections import defaultdict

META_PATTERNS = {
    "dataset":     re.compile(r"\bdataset=([^\s]+)"),
    "ver":         re.compile(r"\bver=([^\s]+)"),
    "mode":        re.compile(r"\bmode=([^\s]+)"),
    "method":      re.compile(r"\bmethod=([^\s]+)"),
    "seed":        re.compile(r"\bseed=(\d+)"),
    "idx":         re.compile(r"\bidx=(\d+)"),
    "run_tag":     re.compile(r"\brun_tag=([^\n\r]+)"),
    "aug_ratio":   re.compile(r"\baug_ratio=([0-9.]+)"),
    "aug_bound":   re.compile(r"\baug_bound=([0-9.]+)"),
    "degree_thr":  re.compile(r"\bdegree_thr=([0-9.]+)"),
    "restricted":  re.compile(r"\brestricted=([01])"),
    "alpha":       re.compile(r"\balpha=([0-9.]+)"),
    "gamma":       re.compile(r"\bgamma=([0-9.]+)"),
}
RE_E0               = re.compile(r"\[AUG-INIT\].*?\bE0=(\d+)")
RE_THIS_EPOCH_ADD   = re.compile(r"\[AUG-[^\]]+\].*?\bthis_epoch_add=(\d+)")
RE_GLOBAL_USED_ADDS = re.compile(r"\bglobal_used_adds=(\d+)")

def _fmt_ag(alpha, gamma):
    """Build the tag 'a0.75_g0.90'."""
    try:
        a = f"{float(alpha):.2f}"
    except Exception:
        a = "NA"
    try:
        g = f"{float(gamma):.2f}"
    except Exception:
        g = "NA"
    return f"a{a}_g{g}"

def parse_log(path, verbose=False):
    meta = {k: None for k in META_PATTERNS}
    e0 = None
    total_add = 0
    epochs_with_add = 0
    last_global_used = 0

    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                # E0
                m = RE_E0.search(line)
                if m:
                    e0 = int(m.group(1))

                # this_epoch_add
                m = RE_THIS_EPOCH_ADD.search(line)
                if m:
                    add = int(m.group(1))
                    total_add += add
                    if add > 0:
                        epochs_with_add += 1
                    if verbose:
                        print(f"[DEBUG] {os.path.basename(path)}: +{add} (running total={total_add})")

                # global_used_adds (track last seen)
                m = RE_GLOBAL_USED_ADDS.search(line)
                if m:
                    last_global_used = int(m.group(1))

                # meta fields (cheap, do every line)
                for key, pat in META_PATTERNS.items():
                    if meta[key] is None:
                        mm = pat.search(line)
                        if mm:
                            meta[key] = mm.group(1).strip()

    except FileNotFoundError:
        print(f"[WARN] File not found: {path}", file=sys.stderr)
        return None

    # Convert numeric fields where applicable
    for k in ("seed", "idx"):
        if meta[k] is not None:
            meta[k] = int(meta[k])
    for k in ("aug_ratio", "aug_bound", "degree_thr", "alpha", "gamma"):
        if meta[k] is not None:
            try:
                meta[k] = float(meta[k])
            except:
                pass

    # sanity
    ratio_of_e0 = (total_add / e0) if e0 not in (None, 0) else None
    mismatch = (last_global_used != 0 and last_global_used != total_add)
    if mismatch:
        print(f"[WARN] {os.path.basename(path)}: total_add ({total_add}) != last_global_used_adds ({last_global_used})", file=sys.stderr)

    return {
        "file": path,
        "dataset": meta["dataset"],
        "ver": meta["ver"],
        "mode": meta["mode"],
        "method": meta["method"],
        "seed": meta["seed"],
        "idx": meta["idx"],
        "run_tag": meta["run_tag"],
        "aug_ratio": meta["aug_ratio"],
        "aug_bound": meta["aug_bound"],
        "degree_thr": meta["degree_thr"],
        "restricted": meta["restricted"],
        "alpha": meta["alpha"],
        "gamma": meta["gamma"],
        "E0": e0,
        "total_added_edges": total_add,
        "epochs_with_add": epochs_with_add,
        "final_global_used_adds": last_global_used,
        "add_ratio_of_E0": ratio_of_e0,
        "ag_tag": _fmt_ag(meta["alpha"], meta["gamma"]),
    }

def _ag_sort_key(tag):
    # sort by numeric (alpha, gamma) parsed from 'aX.XX_gY.YY'
    try:
        a = float(tag.split("_")[0][1:])
        g = float(tag.split("_")[1][1:])
        return (a, g)
    except Exception:
        return (float("inf"), float("inf"))

def main():
    ap = argparse.ArgumentParser(description="Summarize added edges from ARON logs, with α/γ split into separate columns.")
    ap.add_argument("pattern", help="Glob for log files, e.g. logs/1029/*.log")
    ap.add_argument("--out", default="added_edges_summary_wide.csv", help="CSV output path")
    ap.add_argument("--verbose", action="store_true", help="Print debug lines while parsing")
    ap.add_argument("--also_long", action="store_true", help="Additionally write the original long-format CSV beside the wide one")
    args = ap.parse_args()

    files = sorted(glob.glob(args.pattern))
    if not files:
        print(f"[ERR] No files match: {args.pattern}", file=sys.stderr)
        sys.exit(2)

    long_rows = []
    for p in files:
        res = parse_log(p, verbose=args.verbose)
        if res is None:
            continue
        long_rows.append(res)

        # terse per-file line for terminal
        e0 = res["E0"] or 0
        add = res["total_added_edges"]
        ratio = (add / e0) if e0 else 0.0
        print(
            f"{os.path.basename(p)} | dataset={res['dataset']} ver={res['ver']} mode={res['mode']} method={res['method']} "
            f"seed={res['seed']} idx={res['idx']} run_tag={res['run_tag']} | E0={e0} total_add={add} "
            f"final_global_used={res['final_global_used_adds']} add/E0={ratio:.6f} epochs_with_add={res['epochs_with_add']} ag={res['ag_tag']}"
        )

    if args.also_long:
        long_out = os.path.splitext(args.out)[0] + "_long.csv"
        fieldnames_long = [
            "file","dataset","ver","mode","method","seed","idx","run_tag",
            "aug_ratio","aug_bound","degree_thr","restricted","alpha","gamma","ag_tag",
            "E0","total_added_edges","final_global_used_adds","add_ratio_of_E0","epochs_with_add"
        ]
        with open(long_out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames_long)
            w.writeheader()
            for r in long_rows:
                w.writerow(r)
        print(f"[OK] Wrote {len(long_rows)} rows → {long_out}")

    # -------- Pivot to WIDE by (alpha, gamma) --------
    metrics = ["final_global_used_adds", "add_ratio_of_E0", "total_added_edges", "epochs_with_add"]
    key_cols = ["dataset","ver","mode","method","aug_ratio","aug_bound","degree_thr","restricted","seed","idx"]

    # Aggregate container per key
    by_key = {}
    ag_set_global = set()

    for r in long_rows:
        # build the key
        key = tuple(r.get(k) for k in key_cols)
        ag = r["ag_tag"]
        ag_set_global.add(ag)
        if key not in by_key:
            by_key[key] = {
                **{k: r.get(k) for k in key_cols},
                "E0": r.get("E0"),
            }
        else:
            # sanity check E0 consistency
            if by_key[key]["E0"] is None and r.get("E0") is not None:
                by_key[key]["E0"] = r.get("E0")
            elif (by_key[key]["E0"] is not None and r.get("E0") is not None
                  and by_key[key]["E0"] != r.get("E0")):
                print(f"[WARN] E0 mismatch for key={key}: {by_key[key]['E0']} vs {r.get('E0')}", file=sys.stderr)

        # fill metrics under this α/γ tag
        for m in metrics:
            by_key[key][f"{m}[{ag}]"] = r.get(m)

    # Flatten rows
    ag_list = sorted(list(ag_set_global), key=_ag_sort_key)
    metric_cols = [f"{m}[{ag}]" for ag in ag_list for m in metrics]

    fieldnames_wide = key_cols + ["E0"] + metric_cols

    rows_wide = []
    for key, row in by_key.items():
        rows_wide.append(row)

    # Write WIDE CSV
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames_wide, extrasaction="ignore")
        w.writeheader()
        for r in rows_wide:
            w.writerow(r)

    print(f"[wide] keys={len(rows_wide)} | alpha-gamma combos={ag_list}")
    print(f"[OK] Wrote wide CSV → {args.out}")

if __name__ == "__main__":
    main()
