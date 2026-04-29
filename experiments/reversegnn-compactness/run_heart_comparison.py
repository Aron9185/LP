"""
Run an old-vs-new ReverseGNN comparison matrix.

Primary comparison bundle:
  1. old_frozen_bilinear_recon_radius
  2. old_dynamic_bilinear_recon_radius
  3. old_dynamic_bilinear_hybrid_hybrid_easy
  4. new_dynamic_bilinear_hybrid_hybrid_heart_like
  5. new_dynamic_bilinear_hybrid_radius_heart_like
  6. heart_coeba_v6
"""

import argparse
import concurrent.futures
import os
import shlex
import subprocess
from pathlib import Path

import pandas as pd

from experiment_paths import artifact_path, experiment_root, repo_root, sweep_log_dir
from run_research_matrix import extract_metrics, flatten_summary_columns, merge_with_existing_runs


COMPARE_CONFIGS = [
    {
        "name": "old_frozen_bilinear_recon_radius",
        "family": "old",
        "flags": [
            "--freeze_c0p_at_edit_start",
            "--decoder_type", "bilinear",
            "--decoder_objective", "recon",
            "--compactness_objective", "radius",
        ],
    },
    {
        "name": "ablate_dynamic_bilinear_hybrid_radius_easy",
        "family": "ablate",
        "flags": [
            "--dynamic_c0p_targets",
            "--decoder_type", "bilinear",
            "--decoder_objective", "hybrid",
            "--compactness_objective", "radius",
            "--decoder_rank_strategy", "easy",
        ],
    },
    {
        "name": "ablate_frozen_bilinear_hybrid_radius_heart_like",
        "family": "ablate",
        "flags": [
            "--freeze_c0p_at_edit_start",
            "--decoder_type", "bilinear",
            "--decoder_objective", "hybrid",
            "--compactness_objective", "radius",
            "--decoder_rank_strategy", "heart_like",
        ],
    },
    {
        "name": "ablate_frozen_bilinear_hybrid_radius_easy",
        "family": "ablate",
        "flags": [
            "--freeze_c0p_at_edit_start",
            "--decoder_type", "bilinear",
            "--decoder_objective", "hybrid",
            "--compactness_objective", "radius",
            "--decoder_rank_strategy", "easy",
        ],
    },
    {
        "name": "old_dynamic_bilinear_recon_radius",
        "family": "old",
        "flags": [
            "--dynamic_c0p_targets",
            "--decoder_type", "bilinear",
            "--decoder_objective", "recon",
            "--compactness_objective", "radius",
        ],
    },
    {
        "name": "old_dynamic_bilinear_hybrid_hybrid_easy",
        "family": "old",
        "flags": [
            "--dynamic_c0p_targets",
            "--decoder_type", "bilinear",
            "--decoder_objective", "hybrid",
            "--compactness_objective", "hybrid",
            "--decoder_rank_strategy", "easy",
        ],
    },
    {
        "name": "new_dynamic_bilinear_hybrid_hybrid_heart_like",
        "family": "new",
        "flags": [
            "--dynamic_c0p_targets",
            "--decoder_type", "bilinear",
            "--decoder_objective", "hybrid",
            "--compactness_objective", "hybrid",
            "--decoder_rank_strategy", "heart_like",
        ],
    },
    {
        "name": "ablate_frozen_bilinear_hybrid_hybrid_easy",
        "family": "ablate",
        "flags": [
            "--freeze_c0p_at_edit_start",
            "--decoder_type", "bilinear",
            "--decoder_objective", "hybrid",
            "--compactness_objective", "hybrid",
            "--decoder_rank_strategy", "easy",
        ],
    },
    {
        "name": "ablate_frozen_bilinear_hybrid_hybrid_heart_like",
        "family": "ablate",
        "flags": [
            "--freeze_c0p_at_edit_start",
            "--decoder_type", "bilinear",
            "--decoder_objective", "hybrid",
            "--compactness_objective", "hybrid",
            "--decoder_rank_strategy", "heart_like",
        ],
    },
    {
        "name": "new_dynamic_bilinear_hybrid_radius_heart_like",
        "family": "new",
        "mode": "revgnn",
        "flags": [
            "--dynamic_c0p_targets",
            "--decoder_type", "bilinear",
            "--decoder_objective", "hybrid",
            "--compactness_objective", "radius",
            "--decoder_rank_strategy", "heart_like",
        ],
    },
    {
        "name": "heart_decoder_scorer_bilinear_hybrid_radius",
        "family": "new",
        "mode": "revgnn",
        "flags": [
            "--dynamic_c0p_targets",
            "--decoder_type", "bilinear",
            "--decoder_objective", "hybrid",
            "--compactness_objective", "radius",
            "--decoder_rank_strategy", "heart_like",
            "--score_source", "decoder",
        ],
    },
    {
        "name": "heart_coeba_v6",
        "family": "old",
        "mode": "coeba_v6",
        "flags": [
            "--ver", "v6",
        ],
    },
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run the HeaRT-like old-vs-new comparison matrix.")
    parser.add_argument("--prefix", type=str, default="heart_compare", help="Filename prefix for logs and summary artifacts.")
    parser.add_argument("--datasets", nargs="+", default=["cora", "citeseer"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=700)
    parser.add_argument("--edit-start-epoch", type=int, default=10)
    parser.add_argument("--editor-pull-strength", type=float, default=0.1)
    parser.add_argument("--decoded-add-ratio", type=float, default=0.01)
    parser.add_argument("--decoded-remove-ratio", type=float, default=0.0)
    parser.add_argument("--decoded-graph-aug-bound", type=float, default=-1.0)
    parser.add_argument("--decoder-rank-neg-k", type=int, default=8)
    parser.add_argument("--decoder-rank-pool-factor", type=int, default=4)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--configs",
        nargs="+",
        choices=[cfg["name"] for cfg in COMPARE_CONFIGS],
        default=None,
        help="Optional subset of comparison configs to run.",
    )
    parser.add_argument(
        "--extra_flag",
        action="append",
        default=[],
        help="Extra raw flag/token to forward to src/aron_main.py. Repeat once per token.",
    )
    return parser.parse_args()


def log_path(prefix: str, config_name: str, dataset: str, seed: int) -> Path:
    return sweep_log_dir() / f"{prefix}_{dataset}_{config_name}_s{seed}.txt"


def is_complete_log(path: Path) -> bool:
    if not path.exists() or path.stat().st_size <= 1000:
        return False
    text = path.read_text(errors="ignore")
    if (
        "Traceback (most recent call last):" in text
        or "RuntimeError:" in text
        or "Run failed" in text
    ):
        return False
    tail = "\n".join(text.splitlines()[-40:])
    return "val_acc:[" in tail and "test_acc:[" in tail


def config_subset(names: list[str] | None) -> list[dict]:
    if not names:
        return list(COMPARE_CONFIGS)
    wanted = set(names)
    return [cfg for cfg in COMPARE_CONFIGS if cfg["name"] in wanted]


def run_one(task: tuple[int, int, dict, str, int], args) -> dict:
    task_idx, task_total, cfg, dataset, seed = task
    out_path = log_path(args.prefix, cfg["name"], dataset, seed)
    progress = f"{task_idx}/{task_total}"

    if is_complete_log(out_path) and not args.force:
        metrics = extract_metrics(out_path)
        metrics.update(
            {
                "dataset": dataset,
                "seed": seed,
                "config": cfg["name"],
                "family": cfg["family"],
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
        "--dataset", dataset,
        "--seed", str(seed),
        "--epochs", str(args.epochs),
        "--edit_start_epoch", str(args.edit_start_epoch),
        "--eval_log_every", "5",
        "--run_tag", cfg["name"],
    ]
    mode = cfg.get("mode", "revgnn")
    if mode == "revgnn":
        cmd.extend(
            [
                "--use_edited_decoder",
                "--use_decoded_graph_augment",
                "--pull_mask_scope", "cp",
                "--compactness_mask_scope", "cp",
                "--rewrite_endpoint_scope", "c0p",
                "--decoded_same_cluster_only",
                "--decoded_require_c0p_endpoint",
                "--decoded_temporary_view_only",
                "--decoded_add_ratio", str(args.decoded_add_ratio),
                "--decoded_remove_ratio", str(args.decoded_remove_ratio),
                "--decoded_graph_aug_bound", str(args.decoded_graph_aug_bound),
                "--editor_pull_strength", str(args.editor_pull_strength),
                "--decoder_rank_neg_k", str(args.decoder_rank_neg_k),
                "--decoder_rank_pool_factor", str(args.decoder_rank_pool_factor),
                "--ver", "no",
            ]
        )
    elif mode == "coeba_v6":
        cmd.extend(["--ver", "v6"])
    else:
        raise ValueError(f"Unknown comparison mode: {mode}")

    cmd.extend([*cfg["flags"], *args.extra_flag])

    bash = (
        f'cd {shlex.quote(str(repo_root()))} && '
        "source /home/retro/anaconda3/etc/profile.d/conda.sh && "
        "conda activate pyg && "
        + " ".join(shlex.quote(part) for part in cmd)
    )
    env = os.environ.copy()
    env["ARON_EXPERIMENT_ROOT"] = str(experiment_root())

    print(
        f"[run  {progress}] dataset={dataset} config={cfg['name']} seed={seed} "
        f"log={out_path.name}",
        flush=True,
    )
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
            "config": cfg["name"],
            "family": cfg["family"],
            "cached": 0,
            "returncode": proc.returncode,
            "log_path": str(out_path),
        }
    )
    print(
        f"[done {progress}] {dataset} {cfg['name']} s{seed} "
        f"-> rc={proc.returncode} val_roc={metrics.get('val_roc', float('nan')):.4f} "
        f"hit10={metrics.get('test_hit10', float('nan')):.4f} "
        f"add_rank={metrics.get('edit_add_rank', float('nan')):.4f}",
        flush=True,
    )
    return metrics


def main():
    args = parse_args()
    skip_lastfm_marker = artifact_path(f".{args.prefix}_skip_lastfm")
    if skip_lastfm_marker.exists() and "LastFMAsia" in args.datasets:
        args.datasets = [dataset for dataset in args.datasets if dataset != "LastFMAsia"]
        if not args.datasets:
            print(f"[skip] LastFMAsia skipped because {skip_lastfm_marker} exists")
            return

    configs = config_subset(args.configs)
    task_specs = [(cfg, dataset, seed) for cfg in configs for dataset in args.datasets for seed in args.seeds]
    tasks = [
        (idx, len(task_specs), cfg, dataset, seed)
        for idx, (cfg, dataset, seed) in enumerate(task_specs, start=1)
    ]

    print(
        f"HeaRT-like comparison matrix: {len(tasks)} runs "
        f"(datasets={args.datasets}, seeds={args.seeds}, workers={args.max_workers})",
        flush=True,
    )

    rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        for row in executor.map(lambda task: run_one(task, args), tasks):
            rows.append(row)

    raw_path = artifact_path(f"{args.prefix}_runs.csv")
    raw_df = pd.DataFrame(rows).sort_values(["dataset", "config", "seed"]).reset_index(drop=True)
    raw_df = merge_with_existing_runs(raw_df, raw_path)
    raw_df.to_csv(raw_path, index=False)

    numeric_cols = [
        "best_val_epoch",
        "val_roc",
        "best_val_roc",
        "best_val_ap",
        "best_val_hit1",
        "best_val_hit3",
        "best_val_hit10",
        "best_val_hit20",
        "best_val_hit50",
        "best_val_hit100",
        "test_roc",
        "test_ap",
        "test_hit1",
        "test_hit3",
        "test_hit10",
        "radius_before",
        "radius_after",
        "c0p_radius_before",
        "c0p_radius_after",
        "cp_radius_before",
        "cp_radius_after",
        "added_edges_total",
        "removed_edges_total",
        "edit_add_rank",
        "edit_remove_rank",
        "edit_heart_rank",
        "heart_rank_pairs",
        "decoder_normalize_input",
        "heart_rank_weight",
        "heart_rank_margin",
        "heart_rank_neg_k",
        "diag_dot_val_hit10",
        "diag_decoder_val_hit10",
        "diag_dot_pos_mean",
        "diag_dot_neg_mean",
        "diag_decoder_pos_mean",
        "diag_decoder_neg_mean",
        "diag_dot_decoder_corr",
        "diag_dot_test_hit10",
        "diag_decoder_test_hit10",
        "rewrite_applied",
    ]
    summary_df = (
        raw_df.groupby(["dataset", "family", "config"])[numeric_cols]
        .agg(["mean", "std"])
        .round(6)
    )
    summary_df = flatten_summary_columns(summary_df)
    summary_path = artifact_path(f"{args.prefix}_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print("\n=== HeaRT-like Comparison Summary ===")
    print(summary_df.to_string(index=False))
    print(f"\nRaw runs saved to {raw_path}")
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
