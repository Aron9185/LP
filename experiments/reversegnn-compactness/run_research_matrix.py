"""
Run the research-grounded ReverseGNN ablation matrix.

Matrix:
  1. frozen  + bilinear + recon  + radius
  2. dynamic + bilinear + recon  + radius
  3. dynamic + bilinear + hybrid + radius
  4. dynamic + bilinear + hybrid + hybrid compactness
  5. dynamic + mlp_pair + hybrid + hybrid compactness
"""

import argparse
import concurrent.futures
import os
import re
import shlex
import subprocess
from pathlib import Path

import pandas as pd

from experiment_paths import artifact_path, experiment_root, repo_root, sweep_log_dir


RESEARCH_CONFIGS = [
    {
        "name": "frozen_bilinear_recon_radius",
        "flags": [
            "--freeze_c0p_at_edit_start",
            "--decoder_type", "bilinear",
            "--decoder_objective", "recon",
            "--compactness_objective", "radius",
        ],
    },
    {
        "name": "dynamic_bilinear_recon_radius",
        "flags": [
            "--dynamic_c0p_targets",
            "--decoder_type", "bilinear",
            "--decoder_objective", "recon",
            "--compactness_objective", "radius",
        ],
    },
    {
        "name": "dynamic_bilinear_hybrid_radius",
        "flags": [
            "--dynamic_c0p_targets",
            "--decoder_type", "bilinear",
            "--decoder_objective", "hybrid",
            "--compactness_objective", "radius",
        ],
    },
    {
        "name": "dynamic_bilinear_hybrid_hybrid",
        "flags": [
            "--dynamic_c0p_targets",
            "--decoder_type", "bilinear",
            "--decoder_objective", "hybrid",
            "--compactness_objective", "hybrid",
        ],
    },
    {
        "name": "dynamic_mlp_pair_hybrid_hybrid",
        "flags": [
            "--dynamic_c0p_targets",
            "--decoder_type", "mlp_pair",
            "--decoder_objective", "hybrid",
            "--compactness_objective", "hybrid",
        ],
    },
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run the ReverseGNN compactness research matrix.")
    parser.add_argument("--datasets", nargs="+", default=["cora", "citeseer"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--epochs", type=int, default=700)
    parser.add_argument("--edit-start-epoch", type=int, default=10)
    parser.add_argument("--editor-pull-strength", type=float, default=0.1)
    parser.add_argument("--decoded-add-ratio", type=float, default=0.01)
    parser.add_argument("--decoded-remove-ratio", type=float, default=0.0)
    parser.add_argument("--decoded-graph-aug-bound", type=float, default=-1.0)
    parser.add_argument("--compactness-radius-metric", choices=["cosine", "mahalanobis"], default="cosine")
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Run a tiny cora-only, seed-0 smoke pass.")
    parser.add_argument(
        "--configs",
        nargs="+",
        choices=[cfg["name"] for cfg in RESEARCH_CONFIGS],
        default=None,
        help="Optional subset of configs to run.",
    )
    parser.add_argument(
        "--extra_flag",
        action="append",
        default=[],
        help="Extra raw flag/token to forward to src/aron_main.py. Repeat once per token.",
    )
    return parser.parse_args()


def last_matching_line(text: str, prefix: str) -> str | None:
    for line in reversed(text.splitlines()):
        if line.startswith(prefix):
            return line
    return None


def to_float(token: str):
    token = token.strip().rstrip(",")
    if token.lower() == "nan":
        return float("nan")
    try:
        return float(token)
    except Exception:
        return float("nan")


def parse_key_values(line: str | None) -> dict[str, float]:
    if not line:
        return {}
    parsed = {}
    for match in re.finditer(r"([A-Za-z0-9_]+)=([-+0-9.eE]+|nan|inf|-inf)", line):
        parsed[match.group(1)] = to_float(match.group(2))
    return parsed


def parse_final_hits(line: str | None) -> dict[str, float]:
    if not line:
        return {}
    match = re.search(
        r"1=([^,]+), 3=([^,]+), 10=([^,]+), 20=([^,]+), 50=([^,]+), 100=([^,]+)",
        line,
    )
    if not match:
        return {}
    return {
        "test_hit1": to_float(match.group(1)),
        "test_hit3": to_float(match.group(2)),
        "test_hit10": to_float(match.group(3)),
        "test_hit20": to_float(match.group(4)),
        "test_hit50": to_float(match.group(5)),
        "test_hit100": to_float(match.group(6)),
    }


def parse_final_test(line: str | None) -> dict[str, float]:
    if not line:
        return {}
    match = re.search(r"test_roc = ([^,]+), test_ap = ([^\s]+)", line)
    if not match:
        return {}
    return {
        "test_roc": to_float(match.group(1)),
        "test_ap": to_float(match.group(2)),
    }


def parse_best_validation(line: str | None) -> dict[str, float]:
    if not line:
        return {}
    match = re.search(r"epoch = ([^,]+), val_roc = ([^,]+), val_ap = ([^\s]+)", line)
    if not match:
        return {}
    return {
        "best_val_epoch_from_log": to_float(match.group(1)),
        "best_val_roc": to_float(match.group(2)),
        "best_val_ap": to_float(match.group(3)),
    }


def parse_best_validation_hits(line: str | None) -> dict[str, float]:
    if not line:
        return {}
    match = re.search(
        r"1=([^,]+), 3=([^,]+), 10=([^,]+), 20=([^,]+), 50=([^,]+), 100=([^,]+)",
        line,
    )
    if not match:
        return {}
    return {
        "best_val_hit1": to_float(match.group(1)),
        "best_val_hit3": to_float(match.group(2)),
        "best_val_hit10": to_float(match.group(3)),
        "best_val_hit20": to_float(match.group(4)),
        "best_val_hit50": to_float(match.group(5)),
        "best_val_hit100": to_float(match.group(6)),
    }


def extract_metrics(log_path: Path) -> dict:
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return {"status": "missing_log"}

    summary = parse_key_values(last_matching_line(text, "[SANITY SUMMARY]"))
    best_validation = parse_best_validation(last_matching_line(text, "[BEST VALIDATION]"))
    best_validation_hits = parse_best_validation_hits(last_matching_line(text, "[BEST VALIDATION HIT@K]"))
    final_hits = parse_final_hits(last_matching_line(text, "[FINAL TEST] Hit@K:"))
    final_test = parse_final_test(last_matching_line(text, "[FINAL TEST] test_roc ="))
    final_decoder_diag = parse_key_values(last_matching_line(text, "[DECODER-DIAG][FINAL]"))

    edit_graph_rows = re.findall(r"\[EDIT-GRAPH\].*? add=(\d+) remove=(\d+)", text)
    added_total = sum(int(add) for add, _ in edit_graph_rows)
    removed_total = sum(int(rem) for _, rem in edit_graph_rows)

    return {
        "status": "ok" if summary else "parse_incomplete",
        "best_val_epoch": summary.get("best_val_epoch", float("nan")),
        "val_roc": summary.get("val_roc", float("nan")),
        "radius_before": summary.get("radius_before", float("nan")),
        "radius_after": summary.get("radius_after", float("nan")),
        "delta": summary.get("delta", float("nan")),
        "c0p_radius_before": summary.get("c0p_radius_before", float("nan")),
        "c0p_radius_after": summary.get("c0p_radius_after", float("nan")),
        "cp_radius_before": summary.get("cp_radius_before", float("nan")),
        "cp_radius_after": summary.get("cp_radius_after", float("nan")),
        "noncompact_radius_before": summary.get("noncompact_radius_before", float("nan")),
        "noncompact_radius_after": summary.get("noncompact_radius_after", float("nan")),
        "noncompact_radius_p90_before": summary.get("noncompact_radius_p90_before", float("nan")),
        "noncompact_radius_p90_after": summary.get("noncompact_radius_p90_after", float("nan")),
        "noncompact_radius_max_before": summary.get("noncompact_radius_max_before", float("nan")),
        "noncompact_radius_max_after": summary.get("noncompact_radius_max_after", float("nan")),
        "editor_noncompact_push_strength": summary.get("editor_noncompact_push_strength", float("nan")),
        "editor_noise_push_strength": summary.get("editor_noise_push_strength", float("nan")),
        "editor_push_preserve_norm": summary.get("editor_push_preserve_norm", float("nan")),
        "push_noncompact_count": summary.get("push_noncompact_count", float("nan")),
        "push_noise_count": summary.get("push_noise_count", float("nan")),
        "push_noncompact_anchor_cosdist_before": summary.get("push_noncompact_anchor_cosdist_before", float("nan")),
        "push_noncompact_anchor_cosdist_after": summary.get("push_noncompact_anchor_cosdist_after", float("nan")),
        "push_noise_anchor_cosdist_before": summary.get("push_noise_anchor_cosdist_before", float("nan")),
        "push_noise_anchor_cosdist_after": summary.get("push_noise_anchor_cosdist_after", float("nan")),
        "rewrite_applied": summary.get("rewrite_applied", float("nan")),
        "edit_recon": summary.get("edit_recon", float("nan")),
        "edit_keep": summary.get("edit_keep", float("nan")),
        "edit_add_rank": summary.get("edit_add_rank", float("nan")),
        "edit_remove_rank": summary.get("edit_remove_rank", float("nan")),
        "edit_heart_rank": summary.get("edit_heart_rank", float("nan")),
        "lp_full_graph": summary.get("lp_full_graph", float("nan")),
        "maskgae_feature_loss": summary.get("maskgae_feature_loss", float("nan")),
        "maskgae_aug_feature_loss": summary.get("maskgae_aug_feature_loss", float("nan")),
        "cimage_factor_loss": summary.get("cimage_factor_loss", float("nan")),
        "cimage_cluster_loss": summary.get("cimage_cluster_loss", float("nan")),
        "cimage_aug_factor_loss": summary.get("cimage_aug_factor_loss", float("nan")),
        "cimage_aug_cluster_loss": summary.get("cimage_aug_cluster_loss", float("nan")),
        "cimage_factor_weight": summary.get("cimage_factor_weight", float("nan")),
        "cimage_cluster_weight": summary.get("cimage_cluster_weight", float("nan")),
        "cimage_num_factors": summary.get("cimage_num_factors", float("nan")),
        "cimage_num_clusters": summary.get("cimage_num_clusters", float("nan")),
        "cimage_pseudo_label_threshold": summary.get("cimage_pseudo_label_threshold", float("nan")),
        "cimage_factor_select_ratio": summary.get("cimage_factor_select_ratio", float("nan")),
        "cimage_mrmr_redundancy_weight": summary.get("cimage_mrmr_redundancy_weight", float("nan")),
        "cimage_cluster_balance_weight": summary.get("cimage_cluster_balance_weight", float("nan")),
        "cimage_sce_power": summary.get("cimage_sce_power", float("nan")),
        "heart_rank_pairs": summary.get("heart_rank_pairs", float("nan")),
        "prediction_rank": summary.get("prediction_rank", float("nan")),
        "prediction_bce": summary.get("prediction_bce", float("nan")),
        "prediction_joint_rank": summary.get("prediction_joint_rank", float("nan")),
        "prediction_joint_bce": summary.get("prediction_joint_bce", float("nan")),
        "prediction_extra_reg": summary.get("prediction_extra_reg", float("nan")),
        "prediction_rank_pairs": summary.get("prediction_rank_pairs", float("nan")),
        "prediction_rank_weight": summary.get("prediction_rank_weight", final_decoder_diag.get("prediction_rank_weight", float("nan"))),
        "prediction_bce_weight": summary.get("prediction_bce_weight", float("nan")),
        "prediction_encoder_weight": summary.get("prediction_encoder_weight", float("nan")),
        "prediction_gate_l1_weight": summary.get("prediction_gate_l1_weight", float("nan")),
        "prediction_h3_gate_init": summary.get("prediction_h3_gate_init", float("nan")),
        "prediction_h3_gate": summary.get("prediction_h3_gate", final_decoder_diag.get("prediction_h3_gate", float("nan"))),
        "decoder_normalize_input": summary.get("decoder_normalize_input", float("nan")),
        "heart_rank_weight": summary.get("heart_rank_weight", float("nan")),
        "heart_rank_margin": summary.get("heart_rank_margin", float("nan")),
        "heart_rank_neg_k": summary.get("heart_rank_neg_k", float("nan")),
        "diag_dot_val_hit10": summary.get("diag_dot_val_hit10", float("nan")),
        "diag_decoder_val_hit10": summary.get("diag_decoder_val_hit10", float("nan")),
        "diag_pred_val_hit10": summary.get("diag_pred_val_hit10", float("nan")),
        "diag_dot_pos_mean": summary.get("diag_dot_pos_mean", float("nan")),
        "diag_dot_neg_mean": summary.get("diag_dot_neg_mean", float("nan")),
        "diag_decoder_pos_mean": summary.get("diag_decoder_pos_mean", float("nan")),
        "diag_decoder_neg_mean": summary.get("diag_decoder_neg_mean", float("nan")),
        "diag_pred_pos_mean": summary.get("diag_pred_pos_mean", final_decoder_diag.get("pred_pos_mean", float("nan"))),
        "diag_pred_neg_mean": summary.get("diag_pred_neg_mean", final_decoder_diag.get("pred_neg_mean", float("nan"))),
        "diag_dot_decoder_corr": summary.get("diag_dot_decoder_corr", final_decoder_diag.get("dot_decoder_corr", float("nan"))),
        "diag_dot_pred_corr": summary.get("diag_dot_pred_corr", final_decoder_diag.get("dot_pred_corr", float("nan"))),
        "diag_decoder_pred_corr": summary.get("diag_decoder_pred_corr", final_decoder_diag.get("decoder_pred_corr", float("nan"))),
        "diag_dot_test_hit10": final_decoder_diag.get("dot_test_hit10", float("nan")),
        "diag_decoder_test_hit10": final_decoder_diag.get("decoder_test_hit10", float("nan")),
        "diag_pred_test_hit10": final_decoder_diag.get("pred_test_hit10", float("nan")),
        "edit_compact": summary.get("edit_compact", float("nan")),
        "edit_compact_radius": summary.get("edit_compact_radius", float("nan")),
        "edit_compact_proto": summary.get("edit_compact_proto", float("nan")),
        "added_edges_total": added_total,
        "removed_edges_total": removed_total,
        "rewrite_events": len(edit_graph_rows),
        **best_validation,
        **best_validation_hits,
        **final_test,
        **final_hits,
    }


def log_path(config_name: str, dataset: str, seed: int, smoke: bool) -> Path:
    prefix = "research_matrix_smoke" if smoke else "research_matrix"
    return sweep_log_dir() / f"{prefix}_{dataset}_{config_name}_s{seed}.txt"


def config_subset(names: list[str] | None) -> list[dict]:
    if not names:
        return list(RESEARCH_CONFIGS)
    selected = []
    wanted = set(names)
    for cfg in RESEARCH_CONFIGS:
        if cfg["name"] in wanted:
            selected.append(cfg)
    return selected


def run_one(task: tuple[dict, str, int], args) -> dict:
    cfg, dataset, seed = task
    out_path = log_path(cfg["name"], dataset, seed, args.smoke)

    if out_path.exists() and out_path.stat().st_size > 1000 and not args.force:
        metrics = extract_metrics(out_path)
        metrics.update(
            {
                "dataset": dataset,
                "seed": seed,
                "config": cfg["name"],
                "cached": 1,
                "returncode": 0,
                "log_path": str(out_path),
            }
        )
        print(f"[skip] {out_path.name}")
        return metrics

    epochs = min(args.epochs, 3) if args.smoke else args.epochs
    edit_start_epoch = 1 if args.smoke else args.edit_start_epoch
    eval_log_every = 1 if args.smoke else 5

    cmd = [
        "python",
        "src/aron_main.py",
        "--sweep_mode",
        "--dataset", dataset,
        "--seed", str(seed),
        "--epochs", str(epochs),
        "--edit_start_epoch", str(edit_start_epoch),
        "--eval_log_every", str(eval_log_every),
        "--run_tag", cfg["name"],
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
        "--compactness_radius_metric", args.compactness_radius_metric,
        "--ver", "no",
        *cfg["flags"],
        *args.extra_flag,
    ]

    bash = (
        f'cd {shlex.quote(str(repo_root()))} && '
        "source /home/retro/anaconda3/etc/profile.d/conda.sh && "
        "conda activate pyg && "
        + " ".join(shlex.quote(part) for part in cmd)
    )
    env = os.environ.copy()
    env["ARON_EXPERIMENT_ROOT"] = str(experiment_root())

    print(f"[run ] {out_path.name}")
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
            "cached": 0,
            "returncode": proc.returncode,
            "log_path": str(out_path),
        }
    )
    print(
        f"[done] {dataset} {cfg['name']} s{seed} "
        f"-> rc={proc.returncode} val_roc={metrics.get('val_roc', float('nan')):.4f} "
        f"hit10={metrics.get('test_hit10', float('nan')):.4f}"
    )
    return metrics


def flatten_summary_columns(df: pd.DataFrame) -> pd.DataFrame:
    flat_cols = []
    for col in df.columns:
        if isinstance(col, tuple):
            flat_cols.append("_".join(str(part) for part in col if part))
        else:
            flat_cols.append(str(col))
    df.columns = flat_cols
    return df.reset_index()


def merge_with_existing_runs(raw_df: pd.DataFrame, raw_path: Path) -> pd.DataFrame:
    if not raw_path.exists():
        return raw_df

    try:
        existing_df = pd.read_csv(raw_path)
    except Exception:
        return raw_df

    if existing_df.empty:
        return raw_df

    key_cols = ["dataset", "config", "seed"]
    if any(col not in existing_df.columns for col in key_cols):
        return raw_df

    current_keys = set(raw_df[key_cols].itertuples(index=False, name=None))
    keep_mask = ~existing_df[key_cols].apply(tuple, axis=1).isin(current_keys)
    merged_df = pd.concat([existing_df.loc[keep_mask], raw_df], ignore_index=True, sort=False)
    return merged_df.sort_values(key_cols).reset_index(drop=True)


def main():
    args = parse_args()
    if args.smoke:
        args.datasets = ["cora"]
        args.seeds = [0]

    configs = config_subset(args.configs)
    tasks = [(cfg, dataset, seed) for cfg in configs for dataset in args.datasets for seed in args.seeds]

    print(
        f"ReverseGNN research matrix: {len(tasks)} runs "
        f"(datasets={args.datasets}, seeds={args.seeds}, workers={args.max_workers}, smoke={int(args.smoke)})"
    )

    rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        for row in executor.map(lambda task: run_one(task, args), tasks):
            rows.append(row)

    raw_path = artifact_path("research_matrix_smoke_runs.csv" if args.smoke else "research_matrix_runs.csv")
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
        "noncompact_radius_before",
        "noncompact_radius_after",
        "noncompact_radius_p90_before",
        "noncompact_radius_p90_after",
        "noncompact_radius_max_before",
        "noncompact_radius_max_after",
        "added_edges_total",
        "removed_edges_total",
        "edit_heart_rank",
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
        "prediction_h3_gate_init",
        "prediction_h3_gate",
        "decoder_normalize_input",
        "heart_rank_weight",
        "heart_rank_margin",
        "heart_rank_neg_k",
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
    ]
    summary_df = (
        raw_df.groupby(["dataset", "config"])[numeric_cols]
        .agg(["mean", "std"])
        .round(6)
    )
    summary_df = flatten_summary_columns(summary_df)
    summary_path = artifact_path("research_matrix_smoke_summary.csv" if args.smoke else "research_matrix_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print("\n=== Research Matrix Summary ===")
    print(summary_df.to_string(index=False))
    print(f"\nRaw runs saved to {raw_path}")
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
