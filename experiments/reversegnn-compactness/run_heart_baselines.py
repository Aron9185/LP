"""Run NCNC and BUDDY on the same HeaRT split files used by ARON.

The older baseline logs in ``exist_setting_small_v2`` use ``*_neg.txt`` with one
negative per positive.  This runner intentionally calls the official
``benchmarking/HeaRT_small`` entrypoints and forwards ARON's dataset directory so
validation/test use ``heart_valid_samples.npy`` and ``heart_test_samples.npy``.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path("/home/retro")
ARON_ROOT = ROOT / "ARON"
EXPERIMENT_ROOT = ARON_ROOT / "experiments" / "reversegnn-compactness"
RESULT_ROOT = EXPERIMENT_ROOT / "results"
LOG_DIR = RESULT_ROOT / "baseline_logs"
HEART_WORKDIR = ROOT / "baseline" / "HeaRT" / "benchmarking" / "exist_setting_small_v2"
ARON_HEART_DATA = ARON_ROOT / "dataset"
DEFAULT_PYTHON = ROOT / "anaconda3" / "envs" / "pyg" / "bin" / "python"

METRICS = ("Hits@1", "Hits@3", "Hits@10", "Hits@100", "MRR")


def _buddy_flags(dataset: str) -> list[str]:
    common = [
        "../HeaRT_small/main_buddy_CoraCiteseerPubmed.py",
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
        return common + [
            "--lr",
            "0.01",
            "--l2",
            "1e-4",
            "--label_dropout",
            "0.1",
            "--feature_dropout",
            "0.1",
        ]
    if dataset == "citeseer":
        return common + [
            "--lr",
            "0.001",
            "--l2",
            "0",
            "--label_dropout",
            "0.5",
            "--feature_dropout",
            "0.5",
        ]
    raise ValueError(f"Unsupported BUDDY dataset: {dataset}")


def _ncnc_flags(dataset: str) -> list[str]:
    common = [
        "../HeaRT_small/main_ncn_CoraCiteseerPubmed.py",
        "--predictor",
        "incn1cn1",
        "--testbs",
        "512",
        "--dataset",
        dataset,
        "--hiddim",
        "256",
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
            "0",
            "--predp",
            "0.1",
            "--gnndp",
            "0.1",
            "--mplayers",
            "2",
            "--nnlayers",
            "2",
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
            "1",
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


def build_command(args: argparse.Namespace, model: str, dataset: str) -> list[str]:
    if model == "buddy":
        cmd = [str(args.python), *_buddy_flags(dataset)]
    elif model == "ncnc":
        cmd = [str(args.python), *_ncnc_flags(dataset)]
    else:
        raise ValueError(f"Unknown model: {model}")

    cmd += [
        "--runs",
        str(args.runs),
        "--seed",
        str(args.seed),
        "--input_dir",
        str(args.input_dir),
        "--filename",
        args.filename,
        "--device",
        str(args.device),
    ]

    if args.epochs is not None:
        _replace_arg(cmd, "--epochs", str(args.epochs))
    if args.eval_steps is not None:
        _replace_arg(cmd, "--eval_steps", str(args.eval_steps))
    if args.kill_cnt is not None:
        _replace_arg(cmd, "--kill_cnt", str(args.kill_cnt))
    return cmd


def _replace_arg(cmd: list[str], flag: str, value: str) -> None:
    try:
        idx = cmd.index(flag)
    except ValueError:
        cmd.extend([flag, value])
        return
    cmd[idx + 1] = value


def log_path(prefix: str, model: str, dataset: str) -> Path:
    return LOG_DIR / f"{prefix}_{dataset}_{model}.log"


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
        summary[current_metric][f"{key}_std"] = float(std)
    return summary


def extract_shape_info(log_text: str) -> dict[str, int]:
    shape_re = re.compile(
        r"train valid_pos valid_neg test_pos test_neg .*?"
        r"torch\.Size\(\[(\d+)\]\).*?"
        r"torch\.Size\(\[(\d+)\]\).*?"
        r"torch\.Size\(\[(\d+),\s*(\d+)\]\).*?"
        r"torch\.Size\(\[(\d+)\]\).*?"
        r"torch\.Size\(\[(\d+),\s*(\d+)\]\)"
    )
    match = shape_re.search(log_text)
    if not match:
        return {}
    train_pred, valid_pred, valid_neg_groups, valid_neg_k, test_pred, test_neg_groups, test_neg_k = (
        int(group) for group in match.groups()
    )
    return {
        "train_pred_len": train_pred,
        "valid_pred_len": valid_pred,
        "valid_neg_groups": valid_neg_groups,
        "valid_neg_k": valid_neg_k,
        "test_pred_len": test_pred,
        "test_neg_groups": test_neg_groups,
        "test_neg_k": test_neg_k,
    }


def log_is_complete(path: Path) -> bool:
    if not path.exists() or path.stat().st_size == 0:
        return False
    text = path.read_text(errors="ignore")
    if "Traceback (most recent call last):" in text or "RuntimeError:" in text:
        return False
    parsed = parse_summary(text)
    return all(metric in parsed for metric in METRICS)


def run_one(args: argparse.Namespace, model: str, dataset: str) -> list[dict[str, object]]:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = log_path(args.prefix, model, dataset)
    cmd = build_command(args, model, dataset)

    if args.dry_run:
        print(" ".join(cmd))
        return []

    if log_is_complete(out_path) and not args.force:
        print(f"[skip] {out_path.name}")
        return rows_from_log(out_path, model, dataset, cached=1, returncode=0, command=cmd)

    print(f"[run ] {dataset} {model} -> {out_path}")
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")
    with out_path.open("w", encoding="utf-8") as handle:
        handle.write("# Command: " + " ".join(cmd) + "\n")
        handle.flush()
        proc = subprocess.run(
            cmd,
            cwd=args.workdir,
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=env,
            check=False,
        )
    rows = rows_from_log(out_path, model, dataset, cached=0, returncode=proc.returncode, command=cmd)
    status = "ok" if proc.returncode == 0 else f"rc={proc.returncode}"
    best = next((row for row in rows if row["metric"] == "MRR"), None)
    if best:
        print(
            f"[done] {dataset} {model} {status} "
            f"MRR test={best.get('final_test_mean')}±{best.get('final_test_std')}"
        )
    else:
        print(f"[done] {dataset} {model} {status}")
    return rows


def rows_from_log(
    path: Path,
    model: str,
    dataset: str,
    cached: int,
    returncode: int,
    command: list[str],
) -> list[dict[str, object]]:
    text = path.read_text(errors="ignore")
    summary = parse_summary(text)
    shape_info = extract_shape_info(text)
    rows: list[dict[str, object]] = []
    for metric in METRICS:
        stats = summary.get(metric, {})
        row: dict[str, object] = {
            "dataset": dataset,
            "model": model,
            "metric": metric,
            "cached": cached,
            "returncode": returncode,
            "log_path": str(path),
            "command": " ".join(command),
        }
        row.update(shape_info)
        row.update(stats)
        rows.append(row)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run NCNC/BUDDY with ARON's HeaRT splits.")
    parser.add_argument("--prefix", default="heart500_baselines")
    parser.add_argument("--datasets", nargs="+", choices=["cora", "citeseer"], default=["cora", "citeseer"])
    parser.add_argument("--models", nargs="+", choices=["ncnc", "buddy"], default=["ncnc", "buddy"])
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=None, help="Override official baseline epochs.")
    parser.add_argument("--eval-steps", type=int, default=None, help="Override official eval cadence.")
    parser.add_argument("--kill-cnt", type=int, default=None, help="Override official early-stop patience.")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--filename", default="samples.npy")
    parser.add_argument("--input-dir", type=Path, default=ARON_HEART_DATA)
    parser.add_argument("--workdir", type=Path, default=HEART_WORKDIR)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summary-csv", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.python.exists():
        raise FileNotFoundError(f"Python binary not found: {args.python}")
    if not args.input_dir.exists():
        raise FileNotFoundError(f"HeaRT input dir not found: {args.input_dir}")
    if not args.workdir.exists():
        raise FileNotFoundError(f"HeaRT workdir not found: {args.workdir}")

    all_rows: list[dict[str, object]] = []
    for dataset in args.datasets:
        for model in args.models:
            all_rows.extend(run_one(args, model, dataset))

    if not args.dry_run:
        summary_path = args.summary_csv or (RESULT_ROOT / f"{args.prefix}_summary.csv")
        write_csv(all_rows, summary_path)
        print(f"[csv ] {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
