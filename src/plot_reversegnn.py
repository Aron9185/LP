#!/usr/bin/env python3
import argparse
import math
import re
from pathlib import Path
from statistics import mean, stdev

import pandas as pd


def to_float(x):
    try:
        return float(x)
    except Exception:
        return math.nan


def clean_text(text: str) -> str:
    # tqdm 進度條會有 \r，先清掉
    return text.replace("\r", "\n")


def extract_first(pattern: str, text: str, flags=0):
    m = re.search(pattern, text, flags)
    return m.groups() if m else None


def extract_last(pattern: str, text: str, flags=0):
    matches = list(re.finditer(pattern, text, flags))
    return matches[-1].groups() if matches else None


def parse_filename(path: Path):
    """
    expected:
      citeseer_revgnn_no_2stage_seed0_idx0.log
      Cora_ML_revgnn_no_2stage_seed3_idx3.log
    """
    name = path.stem
    m = re.match(r"^(?P<dataset>.+?)_revgnn_no_2stage_seed(?P<seed>\d+)_idx(?P<idx>\d+)$", name)
    if m:
        return m.group("dataset"), int(m.group("seed")), int(m.group("idx"))

    # fallback
    seed = math.nan
    idx = math.nan
    m_seed = re.search(r"_seed(\d+)", name)
    m_idx = re.search(r"_idx(\d+)", name)
    if m_seed:
        seed = int(m_seed.group(1))
    if m_idx:
        idx = int(m_idx.group(1))

    dataset = re.sub(r"_seed\d+.*$", "", name)
    return dataset, seed, idx


def parse_log(path: Path):
    text = clean_text(path.read_text(encoding="utf-8", errors="ignore"))
    dataset, seed, idx = parse_filename(path)

    row = {
        "file": path.name,
        "dataset": dataset,
        "seed": seed,
        "idx": idx,
    }

    # run meta
    m = extract_first(
        r"\[PATH\] edited-decoder path active \| decoder=([^\s|]+) \| freeze_c0p=([^\s|]+) \| decoded_graph_augment=([^\s|]+).*?separate_edit_training=([^\s|]+).*?retain_recon=([^\s|]+) \| retain_cl=([^\s|]+).*?compactness_weight=([^\s|]+) \| preserve_weight=([^\s|]+).*?phase2_freeze_encoder=([^\s|]+) \| edit_phase_encoder_lr_scale=([^\s|]+)",
        text,
        flags=re.S,
    )
    if m:
        (
            row["decoder_type"],
            row["freeze_c0p"],
            row["decoded_graph_augment"],
            row["separate_edit_training"],
            row["retain_recon"],
            row["retain_cl"],
            row["compactness_weight"],
            row["preserve_weight"],
            row["phase2_freeze_encoder"],
            row["edit_phase_encoder_lr_scale"],
        ) = m

    m = extract_first(r"degree_threshold \(incl self-loop\)\s*:\s*([0-9eE+.\-]+)", text)
    if m:
        row["degree_threshold_incl_self"] = to_float(m[0])

    m = extract_first(
        r"\[AUG-INIT\] N=(\d+) \| E0=(\d+) \| aug_ratio\(global\)=([0-9eE+.\-]+) \| aug_bound\(per-node\)=([0-9eE+.\-]+)",
        text,
    )
    if m:
        row["num_nodes"] = int(m[0])
        row["E0"] = int(m[1])
        row["aug_ratio_global"] = to_float(m[2])
        row["aug_bound_per_node"] = to_float(m[3])

    m = extract_first(
        r"\[EDIT-PHASE\] entering phase-2 at epoch (\d+) \| retain_recon=([0-9eE+.\-]+) retain_cl=([0-9eE+.\-]+) \| freeze_encoder=([^\s|]+) encoder_lr_scale=([0-9eE+.\-]+)",
        text,
    )
    if m:
        row["phase2_epoch"] = int(m[0])
        row["phase2_retain_recon"] = to_float(m[1])
        row["phase2_retain_cl"] = to_float(m[2])
        row["phase2_freeze_encoder_runtime"] = m[3]
        row["phase2_encoder_lr_scale_runtime"] = to_float(m[4])

    m = extract_first(r"\[EDIT\] fixed c0p size = (\d+)", text)
    if m:
        row["fixed_c0p_size"] = int(m[0])

    # best observed test roc during training
    m = extract_last(
        r"\[TRAIN-OBSERVED BEST TEST ROC\] epoch = (\d+), val_roc = ([0-9eE+.\-]+), val_ap = ([0-9eE+.\-]+), test_roc = ([0-9eE+.\-]+), test_ap = ([0-9eE+.\-]+)",
        text,
    )
    if m:
        row["best_test_epoch_observed"] = int(m[0])
        row["best_test_val_roc_observed"] = to_float(m[1])
        row["best_test_val_ap_observed"] = to_float(m[2])
        row["best_test_roc_observed"] = to_float(m[3])
        row["best_test_ap_observed"] = to_float(m[4])

    # best checkpoint by val roc
    m = extract_last(
        r"\[BEST CHECKPOINT BY VAL ROC\] epoch = (\d+), val_roc = ([0-9eE+.\-]+), val_ap = ([0-9eE+.\-]+)",
        text,
    )
    if m:
        row["best_val_epoch"] = int(m[0])
        row["best_val_roc"] = to_float(m[1])
        row["best_val_ap"] = to_float(m[2])

    # final test
    m = extract_last(
        r"\[FINAL TEST\] test_roc = ([0-9eE+.\-]+), test_ap = ([0-9eE+.\-]+)",
        text,
    )
    if m:
        row["final_test_roc"] = to_float(m[0])
        row["final_test_ap"] = to_float(m[1])

    m = extract_last(
        r"\[FINAL TEST\] Hit@K: 1=([0-9eE+.\-]+), 3=([0-9eE+.\-]+), 10=([0-9eE+.\-]+), 20=([0-9eE+.\-]+), 50=([0-9eE+.\-]+), 100=([0-9eE+.\-]+)",
        text,
    )
    if m:
        row["final_hit1"] = to_float(m[0])
        row["final_hit3"] = to_float(m[1])
        row["final_hit10"] = to_float(m[2])
        row["final_hit20"] = to_float(m[3])
        row["final_hit50"] = to_float(m[4])
        row["final_hit100"] = to_float(m[5])

    # best hit@k lines
    for k in [1, 3, 10, 20, 50, 100]:
        m = extract_last(
            rf"best hit@{k} epoch = (\d+), hit@{k} = ([0-9eE+.\-]+), test_roc_at_peak = ([0-9eE+.\-]+)",
            text,
        )
        if m:
            row[f"best_hit{k}_epoch"] = int(m[0])
            row[f"best_hit{k}"] = to_float(m[1])
            row[f"best_hit{k}_test_roc_at_peak"] = to_float(m[2])

    # sanity summary
    m = extract_last(
        r"\[SANITY SUMMARY\] best_val_epoch=(\d+) val_roc=([0-9eE+.\-]+) "
        r"radius_before=([0-9eE+.\-]+) radius_after=([0-9eE+.\-]+) "
        r"delta=([0-9eE+.\-]+) radius_anchor=([0-9eE+.\-]+|nan) "
        r"delta_anchor=([0-9eE+.\-]+|nan) delta_prev_rewrite=([0-9eE+.\-]+|nan) "
        r"rewrite_applied=(\d+) edit_recon=([0-9eE+.\-]+) edit_compact=([0-9eE+.\-]+)",
        text,
    )
    if m:
        row["sanity_best_val_epoch"] = int(m[0])
        row["sanity_val_roc"] = to_float(m[1])
        row["radius_before"] = to_float(m[2])
        row["radius_after"] = to_float(m[3])
        row["radius_delta"] = to_float(m[4])
        row["radius_anchor"] = to_float(m[5])
        row["radius_delta_anchor"] = to_float(m[6])
        row["delta_prev_rewrite"] = to_float(m[7])
        row["rewrite_applied"] = int(m[8])
        row["sanity_edit_recon"] = to_float(m[9])
        row["sanity_edit_compact"] = to_float(m[10])

    # final radius stats
    for label, prefix in [
        ("all_non_noise", "radius_all_non_noise"),
        (r"core\(c0p\)", "radius_core_c0p"),
        ("noncore", "radius_noncore"),
    ]:
        m = extract_last(
            rf"\[FINAL-RADIUS\] {label}: mean=([0-9eE+.\-]+) std=([0-9eE+.\-]+) p50=([0-9eE+.\-]+) p90=([0-9eE+.\-]+) max=([0-9eE+.\-]+) n=(\d+)",
            text,
        )
        if m:
            row[f"{prefix}_mean"] = to_float(m[0])
            row[f"{prefix}_std"] = to_float(m[1])
            row[f"{prefix}_p50"] = to_float(m[2])
            row[f"{prefix}_p90"] = to_float(m[3])
            row[f"{prefix}_max"] = to_float(m[4])
            row[f"{prefix}_n"] = int(m[5])

    m = extract_last(r"Total training time ([0-9eE+.\-]+)", text)
    if m:
        row["training_time_sec"] = to_float(m[0])

    m = extract_last(r"val_acc:\[([0-9eE+.\-]+)\]", text)
    if m:
        row["val_acc"] = to_float(m[0])

    m = extract_last(r"test_acc:\[([0-9eE+.\-]+)\]", text)
    if m:
        row["test_acc"] = to_float(m[0])

    return row


def summarize(df: pd.DataFrame):
    metrics = [
        "best_val_roc",
        "best_val_ap",
        "final_test_roc",
        "final_test_ap",
        "final_hit1",
        "final_hit3",
        "final_hit10",
        "final_hit20",
        "final_hit50",
        "final_hit100",
        "best_hit1",
        "best_hit3",
        "best_hit10",
        "best_hit20",
        "best_hit50",
        "best_hit100",
        "radius_before",
        "radius_after",
        "radius_delta",
        "training_time_sec",
        "val_acc",
        "test_acc",
    ]
    metrics = [m for m in metrics if m in df.columns]

    rows = []
    for dataset, g in df.groupby("dataset", dropna=False):
        row = {"dataset": dataset, "runs": len(g)}
        for m in metrics:
            vals = pd.to_numeric(g[m], errors="coerce").dropna().tolist()
            if not vals:
                row[f"{m}_mean"] = math.nan
                row[f"{m}_std"] = math.nan
            elif len(vals) == 1:
                row[f"{m}_mean"] = vals[0]
                row[f"{m}_std"] = 0.0
            else:
                row[f"{m}_mean"] = mean(vals)
                row[f"{m}_std"] = stdev(vals)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("logdir", type=str, help="Directory containing the log files")
    parser.add_argument("--pattern", type=str, default="*.log")
    parser.add_argument("--outdir", type=str, default=None)
    args = parser.parse_args()

    logdir = Path(args.logdir)
    outdir = Path(args.outdir) if args.outdir else logdir
    outdir.mkdir(parents=True, exist_ok=True)

    files = sorted(logdir.glob(args.pattern))
    if not files:
        raise SystemExit(f"No log files found in {logdir} with pattern {args.pattern}")

    rows = []
    for fp in files:
        try:
            rows.append(parse_log(fp))
        except Exception as e:
            rows.append({"file": fp.name, "parse_error": str(e)})

    df = pd.DataFrame(rows)
    sort_cols = [c for c in ["dataset", "seed", "idx"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(by=sort_cols, kind="stable")

    summary_df = summarize(df)

    run_csv = outdir / "revgnn_run_table.csv"
    summary_csv = outdir / "revgnn_summary_table.csv"

    df.to_csv(run_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    preview_cols = [c for c in [
        "dataset",
        "seed",
        "best_val_epoch",
        "best_val_roc",
        "final_test_roc",
        "final_test_ap",
        "final_hit3",
        "best_hit3",
        "final_hit10",
        "best_hit10",
        "radius_before",
        "radius_after",
        "radius_delta",
        "training_time_sec",
    ] if c in df.columns]

    print("\n=== Per-run table preview ===")
    print(df[preview_cols].to_string(index=False))

    print("\n=== Dataset summary preview ===")
    print(summary_df.to_string(index=False))

    print(f"\nSaved run table -> {run_csv}")
    print(f"Saved summary table -> {summary_csv}")


if __name__ == "__main__":
    main()