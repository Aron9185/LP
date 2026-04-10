#!/usr/bin/env python3
import argparse, math, re
from pathlib import Path
from statistics import mean, stdev
import matplotlib.pyplot as plt
import pandas as pd

def to_float(x):
    try: return float(x)
    except: return math.nan

def clean_text(t): return t.replace('\r', '\n')

def extract_last(pat, txt, flags=0):
    ms = list(re.finditer(pat, txt, flags))
    return ms[-1].groups() if ms else None

def parse_filename(path: Path):
    stem = path.stem
    m = re.match(r"^(?P<dataset>.+?)_es(?P<edit_start>[^_]+)_cp(?P<compact>[^_]+)_pv(?P<preserve>[^_]+)_pull(?P<pull>[^_]+)_rr(?P<rr>[^_]+)_rc(?P<rc>[^_]+)_k(?P<gmmk>[^_]+)_tau(?P<tau>[^_]+)_fz(?P<fz>[^_]+)_seed(?P<seed>\d+)_idx(?P<idx>\d+)$", stem)
    if not m:
        return {"file": path.name}
    d = m.groupdict()
    return {
        "file": path.name, "dataset": d["dataset"], "edit_start": int(d["edit_start"]),
        "compactness_weight": float(d["compact"]), "preserve_weight": float(d["preserve"]),
        "pull_strength": float(d["pull"]), "retain_recon": float(d["rr"]), "retain_cl": float(d["rc"]),
        "gmm_k": int(d["gmmk"]), "gmm_tau": float(d["tau"]), "freeze_targets": int(d["fz"]),
        "seed": int(d["seed"]), "idx": int(d["idx"]),
    }

def parse_log(path: Path):
    row = parse_filename(path)
    text = clean_text(path.read_text(encoding='utf-8', errors='ignore'))
    m = extract_last(r"\[BEST CHECKPOINT BY VAL ROC\] epoch = (\d+), val_roc = ([0-9eE+.\-]+), val_ap = ([0-9eE+.\-]+)", text)
    if m:
        row["best_val_epoch"] = int(m[0]); row["best_val_roc"] = to_float(m[1]); row["best_val_ap"] = to_float(m[2])
    m = extract_last(r"\[FINAL TEST\] test_roc = ([0-9eE+.\-]+), test_ap = ([0-9eE+.\-]+)", text)
    if m:
        row["final_test_roc"] = to_float(m[0]); row["final_test_ap"] = to_float(m[1])
    m = extract_last(r"\[FINAL TEST\] Hit@K: 1=([0-9eE+.\-]+), 3=([0-9eE+.\-]+), 10=([0-9eE+.\-]+), 20=([0-9eE+.\-]+), 50=([0-9eE+.\-]+), 100=([0-9eE+.\-]+)", text)
    if m:
        row["final_hit10"] = to_float(m[2])
    m = extract_last(r"best hit@10 epoch = (\d+), hit@10 = ([0-9eE+.\-]+), test_roc_at_peak = ([0-9eE+.\-]+)", text)
    if m:
        row["best_hit10_epoch"] = int(m[0]); row["best_hit10"] = to_float(m[1]); row["best_hit10_test_roc_at_peak"] = to_float(m[2])
    m = extract_last(r"\[SANITY SUMMARY POST-EDIT\] epoch=(\d+) radius_before=([0-9eE+.\-]+) radius_after=([0-9eE+.\-]+) delta=([0-9eE+.\-]+) radius_anchor=([0-9eE+.\-]+|nan) delta_anchor=([0-9eE+.\-]+|nan)", text)
    if m:
        row["post_edit_epoch"] = int(m[0]); row["radius_before"] = to_float(m[1]); row["radius_after"] = to_float(m[2]); row["radius_delta"] = to_float(m[3])
    else:
        m = extract_last(r"\[SANITY SUMMARY\] best_val_epoch=(\d+) val_roc=([0-9eE+.\-]+) radius_before=([0-9eE+.\-]+) radius_after=([0-9eE+.\-]+) delta=([0-9eE+.\-]+) radius_anchor=([0-9eE+.\-]+|nan) delta_anchor=([0-9eE+.\-]+|nan)", text)
        if m:
            row["post_edit_epoch"] = int(m[0]); row["radius_before"] = to_float(m[2]); row["radius_after"] = to_float(m[3]); row["radius_delta"] = to_float(m[4])
    m = extract_last(r"Total training time ([0-9eE+.\-]+)", text)
    if m:
        row["training_time_sec"] = to_float(m[0])
    return row

def summarize_runs(df):
    group_cols = ["dataset","edit_start","compactness_weight","preserve_weight","pull_strength","retain_recon","retain_cl","gmm_k","gmm_tau","freeze_targets"]
    metrics = ["best_val_roc","best_val_ap","final_test_roc","final_test_ap","final_hit10","best_hit10","radius_before","radius_after","radius_delta","training_time_sec"]
    rows = []
    for keys, g in df.groupby(group_cols, dropna=False):
        row = dict(zip(group_cols, keys)); row["runs"] = len(g)
        for m in metrics:
            vals = pd.to_numeric(g[m], errors='coerce').dropna().tolist() if m in g.columns else []
            if not vals: row[f"{m}_mean"] = math.nan; row[f"{m}_std"] = math.nan
            elif len(vals) == 1: row[f"{m}_mean"] = vals[0]; row[f"{m}_std"] = 0.0
            else: row[f"{m}_mean"] = mean(vals); row[f"{m}_std"] = stdev(vals)
        rows.append(row)
    out = pd.DataFrame(rows)
    if len(out):
        out["score"] = out["final_hit10_mean"].fillna(0.0)*100.0 + out["final_test_roc_mean"].fillna(0.0)*10.0 + (-out["radius_delta_mean"].fillna(0.0))*5.0
        out = out.sort_values(["dataset","score"], ascending=[True,False], kind='stable')
    return out

def make_plots(summary_df, outdir: Path):
    for dataset, g in summary_df.groupby("dataset", dropna=False):
        g = g.sort_values("score", ascending=False).head(12).copy()
        if g.empty: continue
        fig, ax = plt.subplots(figsize=(10,5))
        x = range(len(g))
        labels = [f"es{int(r.edit_start)}|cp{r.compactness_weight}|pv{r.preserve_weight}|pull{r.pull_strength}|k{int(r.gmm_k)}|tau{r.gmm_tau}|fz{int(r.freeze_targets)}" for _, r in g.iterrows()]
        ax.bar(list(x), g["final_hit10_mean"])
        ax.set_title(f"{dataset} top configs by final Hit@10")
        ax.set_ylabel("final Hit@10 mean")
        ax.set_xticks(list(x)); ax.set_xticklabels(labels, rotation=75, ha='right')
        fig.tight_layout(); fig.savefig(outdir / f"{dataset}_topconfigs_hit10.png", dpi=200); plt.close(fig)
        fig, ax = plt.subplots(figsize=(6,5))
        ax.scatter(g["final_test_roc_mean"], g["radius_delta_mean"])
        for _, r in g.iterrows():
            ax.annotate(f"es{int(r.edit_start)},cp{r.compactness_weight},pv{r.preserve_weight},k{int(r.gmm_k)}", (r["final_test_roc_mean"], r["radius_delta_mean"]), fontsize=7)
        ax.set_title(f"{dataset} ROC vs radius delta")
        ax.set_xlabel("final test ROC mean"); ax.set_ylabel("radius delta mean")
        fig.tight_layout(); fig.savefig(outdir / f"{dataset}_roc_vs_radius.png", dpi=200); plt.close(fig)

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('logdir'); ap.add_argument('--pattern', default='*.log'); ap.add_argument('--outdir', default=None); args = ap.parse_args()
    logdir = Path(args.logdir); outdir = Path(args.outdir) if args.outdir else logdir / 'results'; outdir.mkdir(parents=True, exist_ok=True)
    files = sorted(logdir.glob(args.pattern))
    if not files: raise SystemExit(f'No log files found in {logdir}')
    rows = []
    for fp in files:
        try: rows.append(parse_log(fp))
        except Exception as e: rows.append({'file': fp.name, 'parse_error': str(e)})
    run_df = pd.DataFrame(rows)
    sort_cols = [c for c in ['dataset','seed','idx'] if c in run_df.columns]
    if sort_cols: run_df = run_df.sort_values(sort_cols, kind='stable')
    summary_df = summarize_runs(run_df)
    make_plots(summary_df, outdir)
    run_csv = outdir / 'search_run_table.csv'; summary_csv = outdir / 'search_summary_table.csv'; best_csv = outdir / 'search_best_configs.csv'
    run_df.to_csv(run_csv, index=False); summary_df.to_csv(summary_csv, index=False); summary_df.groupby('dataset', dropna=False).head(5).to_csv(best_csv, index=False)
    print(f'Saved: {run_csv}'); print(f'Saved: {summary_csv}'); print(f'Saved: {best_csv}')

if __name__ == '__main__':
    main()
