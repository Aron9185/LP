"""Run backbone-only link-prediction experiments.

This runner intentionally omits the edited decoder and decoded graph rewriting.
It is meant to answer whether an autoencoder backbone works before the CP/C0p
graph editor is inserted into the pipeline.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import shlex
import subprocess
from pathlib import Path

import pandas as pd

from experiment_paths import artifact_path, experiment_root, repo_root, sweep_log_dir
from run_research_matrix import extract_metrics, flatten_summary_columns, merge_with_existing_runs


BACKBONE_DEFAULTS = {
    "vgnae": [],
    "maskgae": [
        "--maskgae_mask_rate",
        "0.3",
        "--maskgae_feature_weight",
        "0.1",
    ],
    "cimage_full": [
        "--maskgae_mask_rate",
        "0.3",
        "--maskgae_feature_weight",
        "0.0",
        "--cimage_factor_weight",
        "0.1",
        "--cimage_cluster_weight",
        "0.1",
        "--cimage_num_factors",
        "8",
        "--cimage_num_clusters",
        "16",
        "--cimage_pseudo_label_threshold",
        "0.90",
        "--cimage_factor_select_ratio",
        "0.50",
        "--cimage_mrmr_redundancy_weight",
        "0.20",
        "--cimage_cluster_balance_weight",
        "0.05",
        "--cimage_sce_power",
        "2.0",
    ],
}

SCORER_CONFIGS = {
    "dot": {
        "name": "dot",
        "flags": ["--score_source", "dot"],
    },
    "pred": {
        "name": "pred",
        "flags": [
            "--score_source",
            "pred_decoder",
            "--prediction_decoder_type",
            "pair_residual_struct",
            "--prediction_rank_weight",
            "1.0",
            "--prediction_bce_weight",
            "0.1",
            "--prediction_rank_margin",
            "0.2",
            "--prediction_rank_neg_k",
            "16",
            "--prediction_rank_pool_factor",
            "8",
            "--prediction_encoder_weight",
            "0.05",
            "--prediction_joint_start_epoch",
            "100",
            "--mlp_pair_max_rows",
            "16",
            "--no_decoder_normalize_input",
        ],
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run backbone-only ARON link-prediction experiments.")
    parser.add_argument("--prefix", type=str, default="no_editor_backbone")
    parser.add_argument("--datasets", nargs="+", default=["cora", "citeseer"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=700)
    parser.add_argument("--split-mode", choices=["random", "heart"], default="random")
    parser.add_argument("--backbones", nargs="+", choices=sorted(BACKBONE_DEFAULTS), default=["cimage_full"])
    parser.add_argument("--configs", nargs="+", choices=sorted(SCORER_CONFIGS), default=["dot"])
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--eval-log-every", type=int, default=5)
    parser.add_argument("--train-eval-every", type=int, default=1)
    parser.add_argument("--skip-train-acc", action="store_true")
    parser.add_argument("--decoder-diag-every", type=int, default=-1)
    parser.add_argument("--full-matrix-eval", action="store_true", help="Forward --full_matrix_eval instead of the default edge-only eval.")
    parser.add_argument("--heart-eval-every", type=int, default=None)
    parser.add_argument("--heart-val-frac", type=float, default=1.0)
    parser.add_argument("--heart-checkpoint-metric", type=str, default="hit10")
    parser.add_argument("--feat-mask-ratio", type=float, default=0.1)
    parser.add_argument(
        "--extra_flag",
        action="append",
        default=[],
        help="Extra raw flag/token to forward to src/aron_main.py. Repeat once per token.",
    )
    return parser.parse_args()


def log_path(prefix: str, backbone: str, config_name: str, dataset: str, seed: int) -> Path:
    return sweep_log_dir() / f"{prefix}_{backbone}_{config_name}_{dataset}_s{seed}.txt"


def is_complete_log(path: Path) -> bool:
    if not path.exists() or path.stat().st_size <= 1000:
        return False
    text = path.read_text(errors="ignore")
    if "Traceback (most recent call last):" in text or "RuntimeError:" in text or "Run failed" in text:
        return False
    tail = "\n".join(text.splitlines()[-80:])
    return "[FINAL TEST] Hit@K:" in tail


def run_one(task: tuple[int, int, str, dict, str, int], args: argparse.Namespace) -> dict:
    task_idx, task_total, backbone, cfg, dataset, seed = task
    out_path = log_path(args.prefix, backbone, cfg["name"], dataset, seed)
    progress = f"{task_idx}/{task_total}"

    if is_complete_log(out_path) and not args.force:
        metrics = extract_metrics(out_path)
        metrics.update(
            {
                "dataset": dataset,
                "seed": seed,
                "backbone": backbone,
                "config": cfg["name"],
                "cached": 1,
                "returncode": 0,
                "log_path": str(out_path),
            }
        )
        print(f"[skip {progress}] {out_path.name}", flush=True)
        return metrics

    cmd = [
        "python",
        "src/aron_main.py",
        "--sweep_mode",
        "--dataset",
        dataset,
        "--seed",
        str(seed),
        "--epochs",
        str(args.epochs),
        "--split_mode",
        args.split_mode,
        "--eval_log_every",
        str(args.eval_log_every),
        "--train_eval_every",
        str(args.train_eval_every),
        "--decoder_diag_every",
        str(args.decoder_diag_every),
        "--run_tag",
        f"no_editor_{backbone}_{cfg['name']}",
        "--ver",
        "no",
        "--ae_backbone",
        backbone,
        "--feat_mask_ratio",
        str(args.feat_mask_ratio),
        "--skip_oom_epoch",
        *BACKBONE_DEFAULTS[backbone],
        *cfg["flags"],
        *args.extra_flag,
    ]
    if args.skip_train_acc:
        cmd.append("--skip_train_acc")
    if args.full_matrix_eval:
        cmd.append("--full_matrix_eval")
    else:
        cmd.append("--edge_eval")
    if args.split_mode == "heart":
        heart_eval_every = args.heart_eval_every if args.heart_eval_every is not None else args.eval_log_every
        cmd.extend(
            [
                "--heart_val_frac",
                str(args.heart_val_frac),
                "--heart_eval_every",
                str(heart_eval_every),
                "--heart_checkpoint_metric",
                args.heart_checkpoint_metric,
            ]
        )

    bash = (
        f"cd {shlex.quote(str(repo_root()))} && "
        "source /home/retro/anaconda3/etc/profile.d/conda.sh && "
        "conda activate pyg && "
        + " ".join(shlex.quote(part) for part in cmd)
    )
    env = os.environ.copy()
    env["ARON_EXPERIMENT_ROOT"] = str(experiment_root())

    print(f"[run {progress}] {dataset} backbone={backbone} config={cfg['name']} seed={seed}", flush=True)
    with out_path.open("w", encoding="utf-8") as handle:
        proc = subprocess.run(
            ["bash", "-lc", bash],
            stdout=handle,
            stderr=subprocess.STDOUT,
            env=env,
        )

    metrics = extract_metrics(out_path)
    metrics.update(
        {
            "dataset": dataset,
            "seed": seed,
            "backbone": backbone,
            "config": cfg["name"],
            "cached": 0,
            "returncode": proc.returncode,
            "log_path": str(out_path),
        }
    )
    print(
        f"[done {progress}] {dataset} backbone={backbone} config={cfg['name']} s{seed} "
        f"rc={proc.returncode} test_roc={metrics.get('test_roc', float('nan')):.4f} "
        f"test_ap={metrics.get('test_ap', float('nan')):.4f} "
        f"hit10={metrics.get('test_hit10', float('nan')):.4f}",
        flush=True,
    )
    return metrics


def main() -> None:
    args = parse_args()
    configs = [SCORER_CONFIGS[name] for name in args.configs]
    raw_tasks = [
        (backbone, cfg, dataset, seed)
        for backbone in args.backbones
        for cfg in configs
        for dataset in args.datasets
        for seed in args.seeds
    ]
    tasks = [
        (idx, len(raw_tasks), backbone, cfg, dataset, seed)
        for idx, (backbone, cfg, dataset, seed) in enumerate(raw_tasks, start=1)
    ]

    print(
        f"No-editor backbone matrix: {len(tasks)} runs "
        f"(backbones={args.backbones}, configs={args.configs}, split={args.split_mode}, "
        f"datasets={args.datasets}, seeds={args.seeds}, workers={args.max_workers})",
        flush=True,
    )

    rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        for row in executor.map(lambda task: run_one(task, args), tasks):
            rows.append(row)

    raw_path = artifact_path(f"{args.prefix}_runs.csv")
    raw_df = pd.DataFrame(rows).sort_values(["dataset", "backbone", "config", "seed"]).reset_index(drop=True)
    raw_df = merge_with_existing_runs(raw_df, raw_path)
    raw_df.to_csv(raw_path, index=False)

    numeric_cols = [
        col
        for col in raw_df.columns
        if col
        not in {
            "status",
            "dataset",
            "backbone",
            "config",
            "log_path",
        }
        and pd.api.types.is_numeric_dtype(raw_df[col])
    ]
    summary_df = (
        raw_df.groupby(["dataset", "backbone", "config"])[numeric_cols]
        .agg(["mean", "std"])
        .round(6)
    )
    summary_df = flatten_summary_columns(summary_df)
    summary_path = artifact_path(f"{args.prefix}_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print("\n=== No-Editor Backbone Summary ===")
    print(summary_df.to_string(index=False))
    print(f"\nRaw runs saved to {raw_path}")
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
