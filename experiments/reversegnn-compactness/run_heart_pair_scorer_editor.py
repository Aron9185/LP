"""Run the two-decoder editor matrix.

This runner supports edit-decoder, prediction-decoder, and shared-scorer
ablation variants trained with HeaRT-style ranking/BCE.
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


PAIR_SCORER_CONFIGS = [
    {
        "name": "one_decoder_dot",
        "family": "one_decoder",
        "score_source": "dot",
        "prediction_decoder_type": "none",
        "use_edit_decoder": True,
        "remove": False,
    },
    {
        "name": "one_decoder_ncnc",
        "family": "one_decoder_ncnc",
        "score_source": "pred_decoder",
        "prediction_decoder_type": "pair_residual_struct_ncnc",
        "use_edit_decoder": False,
        "remove": False,
    },
    {
        "name": "two_decoder_dot",
        "family": "two_decoder",
        "score_source": "dot",
        "prediction_decoder_type": "pair_residual_struct",
        "use_edit_decoder": True,
        "remove": False,
    },
    {
        "name": "two_decoder_pred",
        "family": "two_decoder",
        "score_source": "pred_decoder",
        "prediction_decoder_type": "pair_residual_struct",
        "use_edit_decoder": True,
        "remove": False,
    },
    {
        "name": "two_decoder_ncnc_pred",
        "family": "two_decoder_ncnc",
        "score_source": "pred_decoder",
        "prediction_decoder_type": "pair_residual_struct_ncnc",
        "use_edit_decoder": True,
        "remove": False,
    },
    {
        "name": "two_decoder_ncnc_h3_delta_pred",
        "family": "two_decoder_ncnc_h3_delta",
        "score_source": "pred_decoder",
        "prediction_decoder_type": "pair_residual_struct_ncnc_h3_delta",
        "use_edit_decoder": True,
        "remove": False,
    },
    {
        "name": "two_decoder_ncnc_multi_pred",
        "family": "two_decoder_ncnc_multi",
        "score_source": "pred_decoder",
        "prediction_decoder_type": "pair_residual_struct_ncnc_multi",
        "use_edit_decoder": True,
        "remove": False,
    },
    {
        "name": "two_decoder_ocn_pred",
        "family": "two_decoder_ocn",
        "score_source": "pred_decoder",
        "prediction_decoder_type": "pair_residual_struct_ocn",
        "use_edit_decoder": True,
        "remove": False,
    },
    {
        "name": "one_decoder_dot_remove",
        "family": "one_decoder_remove",
        "score_source": "dot",
        "prediction_decoder_type": "none",
        "use_edit_decoder": True,
        "remove": True,
    },
    {
        "name": "one_decoder_ncnc_remove",
        "family": "one_decoder_ncnc_remove",
        "score_source": "pred_decoder",
        "prediction_decoder_type": "pair_residual_struct_ncnc",
        "use_edit_decoder": False,
        "remove": True,
    },
    {
        "name": "two_decoder_dot_remove",
        "family": "two_decoder_remove",
        "score_source": "dot",
        "prediction_decoder_type": "pair_residual_struct",
        "use_edit_decoder": True,
        "remove": True,
    },
    {
        "name": "two_decoder_pred_remove",
        "family": "two_decoder_remove",
        "score_source": "pred_decoder",
        "prediction_decoder_type": "pair_residual_struct",
        "use_edit_decoder": True,
        "remove": True,
    },
    {
        "name": "two_decoder_ncnc_pred_remove",
        "family": "two_decoder_ncnc_remove",
        "score_source": "pred_decoder",
        "prediction_decoder_type": "pair_residual_struct_ncnc",
        "use_edit_decoder": True,
        "remove": True,
    },
    {
        "name": "two_decoder_ncnc_h3_delta_pred_remove",
        "family": "two_decoder_ncnc_h3_delta_remove",
        "score_source": "pred_decoder",
        "prediction_decoder_type": "pair_residual_struct_ncnc_h3_delta",
        "use_edit_decoder": True,
        "remove": True,
    },
    {
        "name": "two_decoder_ncnc_multi_pred_remove",
        "family": "two_decoder_ncnc_multi_remove",
        "score_source": "pred_decoder",
        "prediction_decoder_type": "pair_residual_struct_ncnc_multi",
        "use_edit_decoder": True,
        "remove": True,
    },
    {
        "name": "two_decoder_ocn_pred_remove",
        "family": "two_decoder_ocn_remove",
        "score_source": "pred_decoder",
        "prediction_decoder_type": "pair_residual_struct_ocn",
        "use_edit_decoder": True,
        "remove": True,
    },
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run pair-scorer editor experiments.")
    parser.add_argument("--prefix", type=str, default="heart_two_decoder_editor_smoke")
    parser.add_argument("--datasets", nargs="+", default=["cora", "citeseer"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=700)
    parser.add_argument(
        "--split-mode",
        choices=["random", "heart", "cimage_paper"],
        default="heart",
        help="Dataset split passed to src/aron_main.py. Use cimage_paper for PyG RandomLinkSplit(0.1 val, 0.05 test).",
    )
    parser.add_argument("--edit-start-epoch", type=int, default=80)
    parser.add_argument("--edit-train-start-epoch", type=int, default=80)
    parser.add_argument("--decoded-rewrite-start-epoch", type=int, default=100)
    parser.add_argument("--decoded-rewrite-every", type=int, default=1)
    parser.add_argument("--editor-pull-strength", type=float, default=1.0)
    parser.add_argument("--pull-mask-scope", choices=["cp", "c0p"], default="cp")
    parser.add_argument("--compactness-mask-scope", choices=["cp", "c0p"], default="cp")
    parser.add_argument("--rewrite-endpoint-scope", choices=["cp", "c0p"], default="c0p")
    parser.add_argument("--decoded-add-ratio", type=float, default=0.01)
    parser.add_argument("--decoded-remove-ratio", type=float, default=0.01)
    parser.add_argument("--decoded-graph-aug-bound", type=float, default=-1.0)
    parser.add_argument("--heart-rank-weight", type=float, default=0.0)
    parser.add_argument("--heart-rank-margin", type=float, default=0.2)
    parser.add_argument("--heart-rank-neg-k", type=int, default=8)
    parser.add_argument("--heart-rank-pool-factor", type=int, default=4)
    parser.add_argument("--decoder-rank-neg-k", type=int, default=8)
    parser.add_argument("--decoder-rank-pool-factor", type=int, default=4)
    parser.add_argument("--mlp-pair-max-rows", type=int, default=16)
    parser.add_argument("--prediction-rank-weight", type=float, default=1.0)
    parser.add_argument("--prediction-bce-weight", type=float, default=0.1)
    parser.add_argument("--prediction-rank-margin", type=float, default=0.2)
    parser.add_argument("--prediction-rank-neg-k", type=int, default=16)
    parser.add_argument("--prediction-rank-pool-factor", type=int, default=8)
    parser.add_argument("--prediction-encoder-weight", type=float, default=0.05)
    parser.add_argument("--prediction-joint-start-epoch", type=int, default=-1)
    parser.add_argument("--prediction-gate-l1-weight", type=float, default=0.0)
    parser.add_argument("--compactness-radius-metric", choices=["cosine", "mahalanobis"], default="cosine")
    parser.add_argument("--decoder-normalize-input", action="store_true", help="Use normalized pair embeddings. Default uses raw pair embeddings.")
    parser.add_argument("--eval-log-every", type=int, default=5)
    parser.add_argument("--train-eval-every", type=int, default=1)
    parser.add_argument("--skip-train-acc", action="store_true")
    parser.add_argument("--decoder-diag-every", type=int, default=-1)
    parser.add_argument("--edit-metric-every", type=int, default=1)
    parser.add_argument("--full-matrix-eval", action="store_true", help="Forward --full_matrix_eval instead of the default edge-only eval.")
    parser.add_argument("--heart-eval-every", type=int, default=5)
    parser.add_argument("--heart-val-frac", type=float, default=1.0)
    parser.add_argument("--heart-checkpoint-metric", type=str, default="hit10")
    parser.add_argument(
        "--random-checkpoint-metric",
        type=str,
        choices=["roc", "ap", "hit1", "hit3", "hit10", "hit20", "hit50", "hit100"],
        default="roc",
        help="Validation metric used by random/cimage-paper split checkpoint selection.",
    )
    parser.add_argument("--max-workers", type=int, default=3)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--configs",
        nargs="+",
        choices=[cfg["name"] for cfg in PAIR_SCORER_CONFIGS],
        default=None,
    )
    parser.add_argument(
        "--extra_flag",
        action="append",
        default=[],
        help="Extra raw flag/token to forward to src/aron_main.py. Repeat once per token. Use --split-mode instead of forwarding --split_mode.",
    )
    return parser.parse_args()


def normalize_forwarded_flags(args) -> None:
    """Consume legacy split-mode forwarding so the child command is unambiguous."""
    cleaned: list[str] = []
    tokens = list(args.extra_flag)
    i = 0
    while i < len(tokens):
        token = tokens[i]
        if token in {"--split_mode", "--split-mode"}:
            if i + 1 >= len(tokens):
                raise ValueError(f"{token} forwarded via --extra_flag requires a value")
            value = tokens[i + 1]
            if value not in {"random", "heart", "cimage_paper"}:
                raise ValueError(f"Unsupported split mode forwarded via --extra_flag: {value}")
            args.split_mode = value
            i += 2
            continue
        if token.startswith("--split_mode=") or token.startswith("--split-mode="):
            value = token.split("=", 1)[1]
            if value not in {"random", "heart", "cimage_paper"}:
                raise ValueError(f"Unsupported split mode forwarded via --extra_flag: {value}")
            args.split_mode = value
            i += 1
            continue
        cleaned.append(token)
        i += 1
    args.extra_flag = cleaned


def config_subset(names: list[str] | None) -> list[dict]:
    if not names:
        return list(PAIR_SCORER_CONFIGS)
    wanted = set(names)
    return [cfg for cfg in PAIR_SCORER_CONFIGS if cfg["name"] in wanted]


def log_path(prefix: str, config_name: str, dataset: str, seed: int) -> Path:
    return sweep_log_dir() / f"{prefix}_{dataset}_{config_name}_s{seed}.txt"


def is_complete_log(path: Path) -> bool:
    if not path.exists() or path.stat().st_size <= 1000:
        return False
    text = path.read_text(errors="ignore")
    if "Traceback (most recent call last):" in text or "RuntimeError:" in text or "Run failed" in text:
        return False
    tail = "\n".join(text.splitlines()[-60:])
    return "[FINAL TEST] Hit@K:" in tail and "[SANITY SUMMARY]" in text


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

    remove_ratio = args.decoded_remove_ratio if cfg.get("remove", False) else 0.0
    cmd = [
        "python",
        "src/aron_main.py",
        "--sweep_mode",
        "--dataset", dataset,
        "--seed", str(seed),
        "--epochs", str(args.epochs),
        "--split_mode", args.split_mode,
        "--random_checkpoint_metric", args.random_checkpoint_metric,
        "--edit_start_epoch", str(args.edit_start_epoch),
        "--edit_train_start_epoch", str(args.edit_train_start_epoch),
        "--decoded_rewrite_start_epoch", str(args.decoded_rewrite_start_epoch),
        "--decoded_rewrite_every", str(args.decoded_rewrite_every),
        "--eval_log_every", str(args.eval_log_every),
        "--train_eval_every", str(args.train_eval_every),
        "--decoder_diag_every", str(args.decoder_diag_every),
        "--edit_metric_every", str(args.edit_metric_every),
        "--run_tag", cfg["name"],
        "--ver", "no",
        "--use_decoded_graph_augment",
        "--dynamic_c0p_targets",
        "--no_decoder_warmup_in_phase1",
        "--decoder_type", "pair_mlp_struct",
        "--decoder_objective", "hybrid",
        "--compactness_objective", "radius",
        "--compactness_radius_metric", args.compactness_radius_metric,
        "--decoder_rank_strategy", "heart_like",
        "--score_source", cfg["score_source"],
        "--pull_mask_scope", args.pull_mask_scope,
        "--compactness_mask_scope", args.compactness_mask_scope,
        "--rewrite_endpoint_scope", args.rewrite_endpoint_scope,
        "--decoded_same_cluster_only",
        "--decoded_require_c0p_endpoint",
        "--decoded_temporary_view_only",
        "--decoded_add_ratio", str(args.decoded_add_ratio),
        "--decoded_remove_ratio", str(remove_ratio),
        "--decoded_graph_aug_bound", str(args.decoded_graph_aug_bound),
        "--editor_pull_strength", str(args.editor_pull_strength),
        "--decoder_rank_neg_k", str(args.decoder_rank_neg_k),
        "--decoder_rank_pool_factor", str(args.decoder_rank_pool_factor),
        "--heart_rank_weight", str(args.heart_rank_weight),
        "--heart_rank_margin", str(args.heart_rank_margin),
        "--heart_rank_neg_k", str(args.heart_rank_neg_k),
        "--heart_rank_pool_factor", str(args.heart_rank_pool_factor),
        "--prediction_decoder_type", str(cfg.get("prediction_decoder_type", "pair_residual_struct")),
        "--prediction_rank_weight", str(args.prediction_rank_weight),
        "--prediction_bce_weight", str(args.prediction_bce_weight),
        "--prediction_rank_margin", str(args.prediction_rank_margin),
        "--prediction_rank_neg_k", str(args.prediction_rank_neg_k),
        "--prediction_rank_pool_factor", str(args.prediction_rank_pool_factor),
        "--prediction_encoder_weight", str(args.prediction_encoder_weight),
        "--prediction_joint_start_epoch", str(args.prediction_joint_start_epoch if args.prediction_joint_start_epoch >= 0 else args.decoded_rewrite_start_epoch),
        "--prediction_gate_l1_weight", str(args.prediction_gate_l1_weight),
        "--mlp_pair_max_rows", str(args.mlp_pair_max_rows),
    ]
    if cfg.get("use_edit_decoder", True):
        cmd.append("--use_edited_decoder")
    if args.skip_train_acc:
        cmd.append("--skip_train_acc")
    if args.full_matrix_eval:
        cmd.append("--full_matrix_eval")
    else:
        cmd.append("--edge_eval")
    if args.split_mode == "heart":
        cmd.extend([
            "--heart_val_frac", str(args.heart_val_frac),
            "--heart_eval_every", str(args.heart_eval_every),
            "--heart_checkpoint_metric", args.heart_checkpoint_metric,
        ])
    if args.decoder_normalize_input:
        cmd.append("--decoder_normalize_input")
    else:
        cmd.append("--no_decoder_normalize_input")
    cmd.extend(args.extra_flag)

    bash = (
        f"cd {shlex.quote(str(repo_root()))} && "
        "source /home/retro/anaconda3/etc/profile.d/conda.sh && "
        "conda activate pyg && "
        + " ".join(shlex.quote(part) for part in cmd)
    )
    env = os.environ.copy()
    env["ARON_EXPERIMENT_ROOT"] = str(experiment_root())

    print(
        f"[run  {progress}] dataset={dataset} config={cfg['name']} seed={seed} "
        f"score={cfg['score_source']} remove={remove_ratio} log={out_path.name}",
        flush=True,
    )
    with out_path.open("w", encoding="utf-8") as handle:
        proc = subprocess.run(["bash", "-lc", bash], stdout=handle, stderr=subprocess.STDOUT, env=env)

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
        f"[done {progress}] {dataset} {cfg['name']} s{seed} -> rc={proc.returncode} "
        f"hit10={metrics.get('test_hit10', float('nan')):.4f} "
        f"dot_diag={metrics.get('diag_dot_test_hit10', float('nan')):.4f} "
        f"decoder_diag={metrics.get('diag_decoder_test_hit10', float('nan')):.4f} "
        f"pred_diag={metrics.get('diag_pred_test_hit10', float('nan')):.4f}",
        flush=True,
    )
    return metrics


def main():
    args = parse_args()
    normalize_forwarded_flags(args)
    configs = config_subset(args.configs)
    task_specs = [(cfg, dataset, seed) for cfg in configs for dataset in args.datasets for seed in args.seeds]
    tasks = [(idx, len(task_specs), cfg, dataset, seed) for idx, (cfg, dataset, seed) in enumerate(task_specs, start=1)]

    print(
        f"Pair-scorer editor matrix: {len(tasks)} runs "
        f"(split={args.split_mode}, datasets={args.datasets}, seeds={args.seeds}, workers={args.max_workers})",
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
        "test_hit20",
        "test_hit50",
        "test_hit100",
        "radius_before",
        "radius_after",
        "c0p_radius_before",
        "c0p_radius_after",
        "cp_radius_before",
        "cp_radius_after",
        "noncompact_radius_before",
        "noncompact_radius_after",
        "noncompact_radius_p90_before",
        "noncompact_radius_p90_after",
        "noncompact_radius_max_before",
        "noncompact_radius_max_after",
        "editor_noncompact_push_strength",
        "editor_noise_push_strength",
        "editor_push_preserve_norm",
        "push_noncompact_count",
        "push_noise_count",
        "push_noncompact_anchor_cosdist_before",
        "push_noncompact_anchor_cosdist_after",
        "push_noise_anchor_cosdist_before",
        "push_noise_anchor_cosdist_after",
        "added_edges_total",
        "removed_edges_total",
        "edit_add_rank",
        "edit_remove_rank",
        "edit_heart_rank",
        "lp_full_graph",
        "maskgae_feature_loss",
        "maskgae_aug_feature_loss",
        "cimage_factor_loss",
        "cimage_cluster_loss",
        "cimage_aug_factor_loss",
        "cimage_aug_cluster_loss",
        "cimage_factor_weight",
        "cimage_cluster_weight",
        "cimage_num_factors",
        "cimage_num_clusters",
        "cimage_pseudo_label_threshold",
        "cimage_factor_select_ratio",
        "cimage_mrmr_redundancy_weight",
        "cimage_cluster_balance_weight",
        "cimage_sce_power",
        "heart_rank_pairs",
        "prediction_rank",
        "prediction_bce",
        "prediction_joint_rank",
        "prediction_joint_bce",
        "prediction_extra_reg",
        "prediction_rank_pairs",
        "prediction_rank_weight",
        "prediction_bce_weight",
        "prediction_encoder_weight",
        "prediction_gate_l1_weight",
        "prediction_h3_gate",
        "decoder_normalize_input",
        "heart_rank_weight",
        "heart_rank_margin",
        "heart_rank_neg_k",
        "selection_score",
        "diag_dot_val_hit10",
        "diag_decoder_val_hit10",
        "diag_pred_val_hit10",
        "diag_dot_pos_mean",
        "diag_dot_neg_mean",
        "diag_decoder_pos_mean",
        "diag_decoder_neg_mean",
        "diag_pred_pos_mean",
        "diag_pred_neg_mean",
        "diag_dot_decoder_corr",
        "diag_dot_pred_corr",
        "diag_decoder_pred_corr",
        "diag_dot_test_hit10",
        "diag_decoder_test_hit10",
        "diag_pred_test_hit10",
        "rewrite_applied",
    ]
    for col in numeric_cols:
        if col not in raw_df.columns:
            raw_df[col] = float("nan")
    summary_df = (
        raw_df.groupby(["dataset", "family", "config"])[numeric_cols]
        .agg(["mean", "std"])
        .round(6)
    )
    summary_df = flatten_summary_columns(summary_df)
    summary_path = artifact_path(f"{args.prefix}_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print("\n=== Pair-Scorer Editor Summary ===")
    print(summary_df.to_string(index=False))
    print(f"\nRaw runs saved to {raw_path}")
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
