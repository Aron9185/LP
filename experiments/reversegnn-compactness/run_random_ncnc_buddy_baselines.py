"""Run NCNC and BUDDY on ARON random no-leak splits.

The HeaRT baseline repo has a random-split implementation in
``exist_setting_small_v2``.  This runner pins that implementation to ARON's
cached ``mask_edge/{dataset}_splitseedN_mask_edge.pkl`` files via environment
variables, then parses Hits/AUC/AP from the baseline logs.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
from pathlib import Path

import pandas as pd

from experiment_paths import artifact_path, sweep_log_dir


ROOT = Path("/home/retro")
ARON_ROOT = ROOT / "ARON"
DEFAULT_PYTHON = ROOT / "anaconda3" / "envs" / "pyg" / "bin" / "python"
BASELINE_WORKDIR = ROOT / "baseline" / "HeaRT" / "benchmarking" / "exist_setting_small_v2"
ARON_SPLIT_ROOT = ARON_ROOT / "mask_edge"

METRICS = ("Hits@1", "Hits@3", "Hits@10", "Hits@20", "Hits@50", "Hits@100", "MRR", "AUC", "AP", "AUPRC")
REQUIRED_METRICS = ("Hits@10", "AUC", "AP")


def _buddy_flags(dataset: str) -> list[str]:
    cmd = [
        "main_buddy_CoraCiteseerPubmed.py",
        "--model",
        "BUDDY",
        "--data_name",
        dataset,
        "--hidden_channels",
        "256",
        "--epochs",
        "9999",
        "--kill_cnt",
        "10",
        "--eval_steps",
        "5",
        "--batch_size",
        "1024",
    ]
    if dataset == "cora":
        return cmd + ["--lr", "0.01", "--label_dropout", "0.1", "--feature_dropout", "0.1", "--l2", "1e-4"]
    if dataset == "citeseer":
        return cmd + ["--lr", "0.001", "--label_dropout", "0.3", "--feature_dropout", "0.3", "--l2", "0"]
    raise ValueError(f"Unsupported BUDDY dataset: {dataset}")


def _ncnc_flags(dataset: str) -> list[str]:
    common = [
        "main_ncn_CoraCiteseerPubmed.py",
        "--predictor",
        "incn1cn1",
        "--dataset",
        dataset,
        "--testbs",
        "512",
        "--epochs",
        "9999",
        "--eval_steps",
        "5",
        "--kill_cnt",
        "10",
        "--batch_size",
        "1024",
        "--pt",
        "0.75",
        "--gnnedp",
        "0.0",
        "--ln",
        "--lnnn",
        "--model",
        "puregcn",
        "--maskinput",
        "--jk",
        "--use_xlin",
        "--tailact",
    ]
    if dataset == "cora":
        return common + [
            "--gnnlr",
            "0.01",
            "--prelr",
            "0.01",
            "--l2",
            "1e-4",
            "--predp",
            "0.1",
            "--gnndp",
            "0.1",
            "--mplayers",
            "2",
            "--nnlayers",
            "1",
            "--hiddim",
            "128",
            "--xdp",
            "0.7",
            "--tdp",
            "0.3",
            "--preedp",
            "0.4",
            "--probscale",
            "4.3",
            "--proboffset",
            "2.8",
            "--alpha",
            "1.0",
        ]
    if dataset == "citeseer":
        return common + [
            "--gnnlr",
            "0.001",
            "--prelr",
            "0.001",
            "--l2",
            "1e-7",
            "--predp",
            "0.5",
            "--gnndp",
            "0.5",
            "--mplayers",
            "1",
            "--nnlayers",
            "2",
            "--hiddim",
            "256",
            "--xdp",
            "0.4",
            "--tdp",
            "0.0",
            "--preedp",
            "0.0",
            "--probscale",
            "6.5",
            "--proboffset",
            "4.4",
            "--alpha",
            "0.4",
            "--twolayerlin",
        ]
    raise ValueError(f"Unsupported NCNC dataset: {dataset}")


def _replace_arg(cmd: list[str], flag: str, value: str) -> None:
    try:
        idx = cmd.index(flag)
    except ValueError:
        cmd.extend([flag, value])
        return
    cmd[idx + 1] = value


def build_command(args: argparse.Namespace, model: str, dataset: str, seed: int) -> list[str]:
    if model == "buddy":
        cmd = [str(args.python), *_buddy_flags(dataset)]
    elif model == "ncnc":
        cmd = [str(args.python), *_ncnc_flags(dataset)]
    else:
        raise ValueError(f"Unknown model: {model}")

    cmd += ["--runs", "1", "--seed", str(seed), "--device", str(args.device)]
    if args.epochs is not None:
        _replace_arg(cmd, "--epochs", str(args.epochs))
    if args.eval_steps is not None:
        _replace_arg(cmd, "--eval_steps", str(args.eval_steps))
    if args.kill_cnt is not None:
        _replace_arg(cmd, "--kill_cnt", str(args.kill_cnt))
    return cmd


def log_path(prefix: str, model: str, dataset: str, seed: int) -> Path:
    return sweep_log_dir() / f"{prefix}_{dataset}_{model}_s{seed}.txt"


def parse_summary(log_text: str) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    current_metric: str | None = None
    in_all_runs = False
    stat_re = re.compile(
        r"^(Highest Train|Highest Valid|Final Train|Final Test):\s+"
        r"([0-9.]+)\s+±\s+([0-9A-Za-z.+-]+)"
    )

    for line in log_text.splitlines():
        text = line.strip()
        if text in METRICS:
            current_metric = text
            in_all_runs = False
            continue
        if text == "All runs:" and current_metric:
            in_all_runs = True
            summary[current_metric] = {}
            continue
        if not in_all_runs or not current_metric:
            continue
        match = stat_re.match(text)
        if not match:
            continue
        label, mean, std = match.groups()
        key = label.lower().replace(" ", "_")
        summary[current_metric][f"{key}_mean"] = float(mean)
        summary[current_metric][f"{key}_std"] = float("nan") if std.lower() == "nan" else float(std)
    return summary


def log_is_complete(path: Path) -> bool:
    if not path.exists() or path.stat().st_size <= 1000:
        return False
    text = path.read_text(errors="ignore")
    if "Traceback (most recent call last):" in text or "RuntimeError:" in text:
        return False
    parsed = parse_summary(text)
    return all(metric in parsed for metric in REQUIRED_METRICS)


def rows_from_log(
    path: Path,
    model: str,
    dataset: str,
    seed: int,
    cached: int,
    returncode: int,
    command: list[str],
) -> list[dict[str, object]]:
    text = path.read_text(errors="ignore")
    summary = parse_summary(text)
    rows: list[dict[str, object]] = []
    for metric in METRICS:
        stats = summary.get(metric, {})
        if not stats:
            continue
        row: dict[str, object] = {
            "dataset": dataset,
            "model": model,
            "split_seed": seed,
            "metric": metric,
            "cached": cached,
            "returncode": returncode,
            "log_path": str(path),
            "command": " ".join(command),
        }
        row.update(stats)
        rows.append(row)
    return rows


def run_one(args: argparse.Namespace, model: str, dataset: str, seed: int) -> list[dict[str, object]]:
    out_path = log_path(args.prefix, model, dataset, seed)
    cmd = build_command(args, model, dataset, seed)

    if log_is_complete(out_path) and not args.force:
        print(f"[skip] {dataset} {model} split_seed={seed} -> {out_path.name}", flush=True)
        return rows_from_log(out_path, model, dataset, seed, cached=1, returncode=0, command=cmd)

    split_path = args.split_root / f"{dataset}_splitseed{seed}_mask_edge.pkl"
    if not split_path.exists():
        raise FileNotFoundError(f"Missing ARON random split: {split_path}")

    print(f"[run ] {dataset} {model} split_seed={seed} -> {out_path.name}", flush=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["ARON_RANDOM_SPLIT_ROOT"] = str(args.split_root)
    env["ARON_RANDOM_SPLIT_SEED"] = str(seed)
    with out_path.open("w", encoding="utf-8") as handle:
        handle.write("# Command: " + " ".join(cmd) + "\n")
        handle.write(f"# ARON_RANDOM_SPLIT_ROOT={args.split_root}\n")
        handle.write(f"# ARON_RANDOM_SPLIT_SEED={seed}\n")
        handle.flush()
        proc = subprocess.run(cmd, cwd=args.workdir, stdout=handle, stderr=subprocess.STDOUT, env=env, check=False)

    rows = rows_from_log(out_path, model, dataset, seed, cached=0, returncode=proc.returncode, command=cmd)
    hit10 = next((row for row in rows if row["metric"] == "Hits@10"), None)
    auc = next((row for row in rows if row["metric"] == "AUC"), None)
    ap = next((row for row in rows if row["metric"] == "AP"), None)
    print(
        f"[done] {dataset} {model} s{seed} rc={proc.returncode} "
        f"hit10={hit10.get('final_test_mean') if hit10 else 'NA'} "
        f"auc={auc.get('final_test_mean') if auc else 'NA'} "
        f"ap={ap.get('final_test_mean') if ap else 'NA'}",
        flush=True,
    )
    return rows


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(raw_path: Path, summary_path: Path) -> None:
    raw_df = pd.read_csv(raw_path)
    numeric_cols = [
        col
        for col in raw_df.columns
        if col not in {"dataset", "model", "metric", "log_path", "command"}
        and pd.api.types.is_numeric_dtype(raw_df[col])
    ]
    summary_df = (
        raw_df.groupby(["dataset", "model", "metric"])[numeric_cols]
        .agg(["count", "mean", "std"])
        .round(6)
    )
    summary_df.columns = ["_".join(col).strip("_") for col in summary_df.columns.to_flat_index()]
    summary_df = summary_df.reset_index()
    summary_df.to_csv(summary_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NCNC/BUDDY on ARON random no-leak splits.")
    parser.add_argument("--prefix", default="random_ncnc_buddy_20260604")
    parser.add_argument("--datasets", nargs="+", choices=["cora", "citeseer"], default=["cora", "citeseer"])
    parser.add_argument("--models", nargs="+", choices=["ncnc", "buddy"], default=["ncnc", "buddy"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--epochs", type=int, default=None, help="Override official baseline epochs.")
    parser.add_argument("--eval-steps", type=int, default=None, help="Override official eval cadence.")
    parser.add_argument("--kill-cnt", type=int, default=None, help="Override official early-stop patience.")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--workdir", type=Path, default=BASELINE_WORKDIR)
    parser.add_argument("--split-root", type=Path, default=ARON_SPLIT_ROOT)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.python.exists():
        raise FileNotFoundError(f"Python binary not found: {args.python}")
    if not args.workdir.exists():
        raise FileNotFoundError(f"Baseline workdir not found: {args.workdir}")

    all_rows: list[dict[str, object]] = []
    for dataset in args.datasets:
        for model in args.models:
            for seed in args.seeds:
                cmd = build_command(args, model, dataset, seed)
                if args.dry_run:
                    print(" ".join(cmd))
                    continue
                all_rows.extend(run_one(args, model, dataset, seed))

    if not args.dry_run:
        raw_path = artifact_path(f"{args.prefix}_runs.csv")
        summary_path = artifact_path(f"{args.prefix}_summary.csv")
        write_csv(all_rows, raw_path)
        write_summary(raw_path, summary_path)
        print(f"[csv ] raw={raw_path}")
        print(f"[csv ] summary={summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
