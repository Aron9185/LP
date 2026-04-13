#!/usr/bin/env python3
import argparse
import math
import re
from pathlib import Path
from statistics import mean, stdev

import pandas as pd


def to_float(x):
    try:
        if isinstance(x, str) and x.lower() == "nan":
            return math.nan
        return float(x)
    except Exception:
        return math.nan


def clean_text(text: str) -> str:
    return text.replace("\r", "\n")


def extract_first(pattern: str, text: str, flags=0):
    m = re.search(pattern, text, flags)
    return m.groups() if m else None


def extract_last(pattern: str, text: str, flags=0):
    matches = list(re.finditer(pattern, text, flags))
    return matches[-1].groups() if matches else None


def infer_exp_from_path(path: Path):
    parent = path.parent.name
    if parent in {"revgnn_no", "coeba_v6"}:
        return parent

    name = path.stem
    if "_revgnn_no_" in name or name.endswith("_revgnn_no"):
        return "revgnn_no"
    if "_coeba_v6_" in name or name.endswith("_coeba_v6"):
        return "coeba_v6"
    return "unknown"


def parse_filename(path: Path):
    name = path.stem
    exp = infer_exp_from_path(path)

    m = re.match(
        r"^(?P<dataset>.+?)_(?P<exp>revgnn_no|coeba_v6)_(?P<tag>.+?)_seed(?P<seed>\d+)_idx(?P<idx>\d+)$",
        name,
    )
    if m:
        return {
            "dataset": m.group("dataset"),
            "exp": m.group("exp"),
            "tag": m.group("tag"),
            "seed": int(m.group("seed")),
            "idx": int(m.group("idx")),
        }

    m = re.match(
        r"^(?P<dataset>.+?)_(?P<exp>revgnn_no|coeba_v6)_seed(?P<seed>\d+)_idx(?P<idx>\d+)$",
        name,
    )
    if m:
        return {
            "dataset": m.group("dataset"),
            "exp": m.group("exp"),
            "tag": "",
            "seed": int(m.group("seed")),
            "idx": int(m.group("idx")),
        }

    seed = math.nan
    idx = math.nan
    m_seed = re.search(r"_seed(\d+)", name)
    m_idx = re.search(r"_idx(\d+)", name)
    if m_seed:
        seed = int(m_seed.group(1))
    if m_idx:
        idx = int(m_idx.group(1))

    dataset = re.sub(r"_(revgnn_no|coeba_v6).*?$", "", name)
    return {
        "dataset": dataset,
        "exp": exp,
        "tag": "",
        "seed": seed,
        "idx": idx,
    }


def parse_log(path: Path):
    text = clean_text(path.read_text(encoding="utf-8", errors="ignore"))
    meta = parse_filename(path)

    row = {
        "file": path.name,
        "dataset": meta["dataset"],
        "exp": meta["exp"],
        "tag": meta["tag"],
        "seed": meta["seed"],
        "idx": meta["idx"],
    }

    m = extract_first(r"Dataset:\s*([^\n]+)", text)
    if m:
        row["dataset_from_log"] = m[0].strip()

    m = extract_first(r"split_mode=([^\n]+)", text)
    if m:
        row["split_mode"] = m[0].strip()

    m = extract_first(r"ver=([^\n]+)", text)
    if m:
        row["ver"] = m[0].strip()

    m = extract_first(r"use_edited_decoder=([^\n]+)", text)
    if m:
        row["use_edited_decoder"] = m[0].strip()

    m = extract_first(
        r"\[PATH\] edited-decoder path active \| decoder=([^\s|]+) \| freeze_c0p=([^\s|]+) \| decoded_graph_augment=([^\s|]+).*?accumulate_base=([^\s|]+).*?edit_start_epoch=([0-9eE+.\-]+).*?separate_edit_training=([^\s|]+).*?retain_recon=([0-9eE+.\-]+) \| retain_cl=([0-9eE+.\-]+).*?compactness_weight=([0-9eE+.\-]+) \| preserve_weight=([0-9eE+.\-]+).*?phase2_freeze_encoder=([^\s|]+) \| edit_phase_encoder_lr_scale=([0-9eE+.\-]+)",
        text,
        flags=re.S,
    )
    if m:
        (
            row["decoder_type"],
            row["freeze_c0p"],
            row["decoded_graph_augment"],
            row["decoded_accumulate_base"],
            row["edit_start_epoch"],
            row["separate_edit_training"],
            row["retain_recon"],
            row["retain_cl"],
            row["compactness_weight"],
            row["preserve_weight"],
            row["phase2_freeze_encoder"],
            row["edit_phase_encoder_lr_scale"],
        ) = m
        row["edit_start_epoch"] = int(float(row["edit_start_epoch"]))
        row["retain_recon"] = to_float(row["retain_recon"])
        row["retain_cl"] = to_float(row["retain_cl"])
        row["compactness_weight"] = to_float(row["compactness_weight"])
        row["preserve_weight"] = to_float(row["preserve_weight"])
        row["edit_phase_encoder_lr_scale"] = to_float(row["edit_phase_encoder_lr_scale"])

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
        r"\[EDIT-PHASE\] entering phase-2 at epoch (\d+) \| retain_recon=([0-9eE+.\-]+) retain_cl=([0-9eE+.\-]+) \| freeze_encoder=([^\s|]+) encoder_lr_scale=([0-9eE+.\-]+) \| task_main=([^\s|]+) edit_w=([0-9eE+.\-]+) \| decoder_inference_only=([^\s|]+)",
        text,
    )
    if m:
        row["phase2_epoch"] = int(m[0])
        row["phase2_retain_recon"] = to_float(m[1])
        row["phase2_retain_cl"] = to_float(m[2])
        row["phase2_freeze_encoder_runtime"] = m[3]
        row["phase2_encoder_lr_scale_runtime"] = to_float(m[4])
        row["phase2_task_main_runtime"] = m[5]
        row["phase2_edit_weight_runtime"] = to_float(m[6])
        row["phase2_decoder_inference_only_runtime"] = m[7]

    m = extract_first(r"\[EDIT\] fixed c0p size = (\d+)", text)
    if m:
        row["fixed_c0p_size"] = int(m[0])

    m = extract_last(
        r"\[BEST VALIDATION\] epoch = (\d+), val_roc = ([0-9eE+.\-]+), val_ap = ([0-9eE+.\-]+)",
        text,
    )
    if m:
        row["best_val_epoch"] = int(m[0])
        row["best_val_roc"] = to_float(m[1])
        row["best_val_ap"] = to_float(m[2])

    m = extract_last(
        r"\[BEST VALIDATION HIT@K\]\s*1=([0-9eE+.\-]+),\s*3=([0-9eE+.\-]+),\s*10=([0-9eE+.\-]+),\s*20=([0-9eE+.\-]+),\s*50=([0-9eE+.\-]+),\s*100=([0-9eE+.\-]+)",
        text,
    )
    if m:
        row["best_val_hit1"] = to_float(m[0])
        row["best_val_hit3"] = to_float(m[1])
        row["best_val_hit10"] = to_float(m[2])
        row["best_val_hit20"] = to_float(m[3])
        row["best_val_hit50"] = to_float(m[4])
        row["best_val_hit100"] = to_float(m[5])

    m = extract_last(
        r"\[BEST VALIDATION META\]\s*radius_before = ([0-9eE+.\-]+|nan),\s*radius_after = ([0-9eE+.\-]+|nan),\s*delta = ([0-9eE+.\-]+|nan),\s*radius_anchor = ([0-9eE+.\-]+|nan),\s*delta_anchor = ([0-9eE+.\-]+|nan),\s*delta_prev_rewrite = ([0-9eE+.\-]+|nan),\s*rewrite_applied = ([0-9eE+.\-]+|nan),\s*edit_recon = ([0-9eE+.\-]+|nan),\s*edit_compact = ([0-9eE+.\-]+|nan),\s*edit_preserve = ([0-9eE+.\-]+|nan)",
        text,
    )
    if m:
        row["best_val_radius_before"] = to_float(m[0])
        row["best_val_radius_after"] = to_float(m[1])
        row["best_val_radius_delta"] = to_float(m[2])
        row["best_val_radius_anchor"] = to_float(m[3])
        row["best_val_delta_anchor"] = to_float(m[4])
        row["best_val_delta_prev_rewrite"] = to_float(m[5])
        row["best_val_rewrite_applied"] = to_float(m[6])
        row["best_val_edit_recon"] = to_float(m[7])
        row["best_val_edit_compact"] = to_float(m[8])
        row["best_val_edit_preserve"] = to_float(m[9])

    if "best_val_epoch" not in row:
        m = extract_last(
            r"\[BEST CHECKPOINT BY VAL ROC\] epoch = (\d+), val_roc = ([0-9eE+.\-]+), val_ap = ([0-9eE+.\-]+)",
            text,
        )
        if m:
            row["best_val_epoch"] = int(m[0])
            row["best_val_roc"] = to_float(m[1])
            row["best_val_ap"] = to_float(m[2])

    m = extract_last(r"\[FINAL TEST\] test_roc = ([0-9eE+.\-]+), test_ap = ([0-9eE+.\-]+)", text)
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

    m = extract_last(
        r"\[SANITY SUMMARY\] best_val_epoch=(\d+) val_roc=([0-9eE+.\-]+) "
        r"radius_before=([0-9eE+.\-]+|nan) radius_after=([0-9eE+.\-]+|nan) "
        r"delta=([0-9eE+.\-]+|nan) radius_anchor=([0-9eE+.\-]+|nan) "
        r"delta_anchor=([0-9eE+.\-]+|nan) delta_prev_rewrite=([0-9eE+.\-]+|nan) "
        r"rewrite_applied=(\d+) edit_recon=([0-9eE+.\-]+|nan) edit_compact=([0-9eE+.\-]+|nan)",
        text,
    )
    if m:
        row["sanity_best_val_epoch"] = int(m[0])
        row["sanity_val_roc"] = to_float(m[1])
        row["sanity_radius_before"] = to_float(m[2])
        row["sanity_radius_after"] = to_float(m[3])
        row["sanity_radius_delta"] = to_float(m[4])
        row["sanity_radius_anchor"] = to_float(m[5])
        row["sanity_delta_anchor"] = to_float(m[6])
        row["sanity_delta_prev_rewrite"] = to_float(m[7])
        row["sanity_rewrite_applied"] = int(m[8])
        row["sanity_edit_recon"] = to_float(m[9])
        row["sanity_edit_compact"] = to_float(m[10])

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
        "best_val_hit1",
        "best_val_hit3",
        "best_val_hit10",
        "best_val_hit20",
        "best_val_hit50",
        "best_val_hit100",
        "final_test_roc",
        "final_test_ap",
        "final_hit1",
        "final_hit3",
        "final_hit10",
        "final_hit20",
        "final_hit50",
        "final_hit100",
        "best_val_radius_before",
        "best_val_radius_after",
        "best_val_radius_delta",
        "training_time_sec",
        "val_acc",
        "test_acc",
    ]
    metrics = [m for m in metrics if m in df.columns]

    rows = []
    for (dataset, exp), g in df.groupby(["dataset", "exp"], dropna=False):
        row = {"dataset": dataset, "exp": exp, "runs": len(g)}
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


def make_comparison_table(summary_df: pd.DataFrame):
    focus_metrics = [
        "best_val_roc_mean",
        "best_val_hit3_mean",
        "best_val_hit10_mean",
        "final_test_roc_mean",
        "final_hit3_mean",
        "final_hit10_mean",
        "training_time_sec_mean",
    ]
    focus_metrics = [m for m in focus_metrics if m in summary_df.columns]
    if not focus_metrics:
        return pd.DataFrame()

    comp = summary_df[["dataset", "exp"] + focus_metrics].copy()
    comp = comp.pivot(index="dataset", columns="exp")
    comp.columns = ["{}_{}".format(metric, exp) for metric, exp in comp.columns]
    comp = comp.reset_index()
    return comp


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("logdir", type=str, help="Root directory containing revgnn_no/ and coeba_v6/ logs")
    parser.add_argument("--pattern", type=str, default="*.log")
    parser.add_argument("--outdir", type=str, default=None)
    args = parser.parse_args()

    logdir = Path(args.logdir)
    outdir = Path(args.outdir) if args.outdir else logdir
    outdir.mkdir(parents=True, exist_ok=True)

    files = sorted(logdir.rglob(args.pattern))
    if not files:
        raise SystemExit(f"No log files found in {logdir} with pattern {args.pattern}")

    rows = []
    for fp in files:
        try:
            rows.append(parse_log(fp))
        except Exception as e:
            rows.append({"file": fp.name, "parse_error": str(e), "exp": infer_exp_from_path(fp)})

    df = pd.DataFrame(rows)
    sort_cols = [c for c in ["dataset", "exp", "seed", "idx"] if c in df.columns]
    if sort_cols:
        df = df.sort_values(by=sort_cols, kind="stable")

    summary_df = summarize(df)
    comparison_df = make_comparison_table(summary_df)

    run_csv = outdir / "run_table.csv"
    summary_csv = outdir / "summary_table.csv"
    compare_csv = outdir / "comparison_table.csv"

    df.to_csv(run_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    comparison_df.to_csv(compare_csv, index=False)

    preview_cols = [c for c in [
        "dataset",
        "exp",
        "seed",
        "best_val_epoch",
        "best_val_roc",
        "best_val_hit3",
        "best_val_hit10",
        "final_test_roc",
        "final_test_ap",
        "final_hit3",
        "final_hit10",
        "best_val_radius_before",
        "best_val_radius_after",
        "best_val_radius_delta",
        "training_time_sec",
    ] if c in df.columns]

    print("\n=== Per-run table preview ===")
    if preview_cols:
        print(df[preview_cols].to_string(index=False))
    else:
        print(df.to_string(index=False))

    print("\n=== Dataset x Experiment summary preview ===")
    print(summary_df.to_string(index=False))

    if not comparison_df.empty:
        print("\n=== Comparison table preview ===")
        print(comparison_df.to_string(index=False))

    print(f"\nSaved run table -> {run_csv}")
    print(f"Saved summary table -> {summary_csv}")
    print(f"Saved comparison table -> {compare_csv}")


if __name__ == "__main__":
    main()
