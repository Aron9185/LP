"""Auto-launch a full h3-delta run after seed-3 gate ablations finish."""

from __future__ import annotations

import argparse
import csv
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = REPO_ROOT / "experiments" / "reversegnn-compactness" / "results"

GATE_RUNS = [
    ("initm3", -3.0, "random_h3_s3_initm3_l10_20260607"),
    ("initm1", -1.0, "random_h3_s3_initm1_l10_20260607"),
    ("init0", 0.0, "random_h3_s3_init0_l10_20260607"),
]


def _to_float(value: object, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _read_single_run(prefix: str) -> dict[str, str] | None:
    path = RESULT_DIR / f"{prefix}_runs.csv"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return None
    return rows[0]


def _wait_for_gate_runs(poll_seconds: int) -> list[dict[str, object]]:
    while True:
        ready_rows: list[dict[str, object]] = []
        missing = []
        bad = []
        for tag, gate_init, prefix in GATE_RUNS:
            row = _read_single_run(prefix)
            if row is None:
                missing.append(prefix)
                continue
            returncode = int(_to_float(row.get("returncode"), default=-1))
            status = str(row.get("status", ""))
            if status != "ok" or returncode != 0:
                bad.append((prefix, status, returncode))
                continue
            row = dict(row)
            row["gate_tag"] = tag
            row["gate_init"] = gate_init
            row["prefix"] = prefix
            ready_rows.append(row)

        if len(ready_rows) == len(GATE_RUNS):
            return ready_rows

        print(
            "[watch] waiting for h3 gate ablations | "
            f"ready={len(ready_rows)}/{len(GATE_RUNS)} missing={len(missing)} bad={bad}",
            flush=True,
        )
        time.sleep(max(5, int(poll_seconds)))


def _write_selection(rows: list[dict[str, object]], selected: dict[str, object], out_path: Path) -> None:
    fieldnames = [
        "selected",
        "gate_tag",
        "gate_init",
        "best_val_hit10",
        "test_hit10",
        "test_roc",
        "test_ap",
        "test_hit100",
        "prediction_h3_gate",
        "prediction_joint_rank",
        "prediction_joint_bce",
        "log_path",
    ]
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "selected": int(row is selected),
                    "gate_tag": row["gate_tag"],
                    "gate_init": row["gate_init"],
                    "best_val_hit10": row.get("best_val_hit10", ""),
                    "test_hit10": row.get("test_hit10", ""),
                    "test_roc": row.get("test_roc", ""),
                    "test_ap": row.get("test_ap", ""),
                    "test_hit100": row.get("test_hit100", ""),
                    "prediction_h3_gate": row.get("prediction_h3_gate", ""),
                    "prediction_joint_rank": row.get("prediction_joint_rank", ""),
                    "prediction_joint_bce": row.get("prediction_joint_bce", ""),
                    "log_path": row.get("log_path", ""),
                }
            )


def _full_run_command(gate_tag: str, gate_init: float, prefix: str) -> list[str]:
    return [
        sys.executable,
        "experiments/reversegnn-compactness/run_heart_pair_scorer_editor.py",
        "--prefix",
        prefix,
        "--split-mode",
        "random",
        "--datasets",
        "cora",
        "citeseer",
        "--seeds",
        "0",
        "1",
        "2",
        "3",
        "4",
        "--epochs",
        "700",
        "--max-workers",
        "1",
        "--mlp-pair-max-rows",
        "16",
        "--configs",
        "two_decoder_ncnc_h3_delta_pred",
        "--random-checkpoint-metric",
        "hit10",
        "--prediction-gate-l1-weight",
        "0.0",
        "--prediction-h3-gate-init",
        str(gate_init),
        "--decoded-remove-ratio",
        "0.0",
        "--editor-pull-strength",
        "0.25",
        "--decoded-add-ratio",
        "0.20",
        "--decoded-graph-aug-bound",
        "0.10",
        "--eval-log-every",
        "50",
        "--decoder-diag-every",
        "50",
        "--edit-metric-every",
        "0",
        "--skip-train-acc",
        "--extra_flag=--skip_oom_epoch",
        "--extra_flag=--feat_mask_ratio",
        "--extra_flag=0.1",
        "--extra_flag=--decoded_require_c0p_noncompact_endpoint",
        "--extra_flag=--decoded_add_degree_target",
        "--extra_flag=1",
        "--extra_flag=--decoded_add_degree_target_scope",
        "--extra_flag=intra_cluster",
        "--extra_flag=--decoded_add_degree_target_nodes",
        "--extra_flag=cp",
        "--extra_flag=--decoded_guarantee_degree_target",
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument(
        "--next-prefix",
        type=str,
        default="",
        help="Optional explicit prefix for the follow-up full run.",
    )
    args = parser.parse_args()

    rows = _wait_for_gate_runs(args.poll_seconds)
    selected = max(
        rows,
        key=lambda row: (
            _to_float(row.get("best_val_hit10")),
            _to_float(row.get("test_hit10")),
        ),
    )
    gate_tag = str(selected["gate_tag"])
    gate_init = float(selected["gate_init"])
    next_prefix = args.next_prefix or f"random_h3_bestgate_{gate_tag}_l10_cora_citeseer5_autofollow_20260607"

    selection_path = RESULT_DIR / "random_h3_s3_gate_ablation_autoselect_20260607.csv"
    _write_selection(rows, selected, selection_path)
    print(f"[watch] selected gate={gate_tag} init={gate_init} by best_val_hit10", flush=True)
    print(f"[watch] selection table: {selection_path}", flush=True)

    cmd = _full_run_command(gate_tag, gate_init, next_prefix)
    env = os.environ.copy()
    env["PYTORCH_CUDA_ALLOC_CONF"] = env.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    print(f"[watch] launching follow-up: {shlex.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env)
    print(f"[watch] follow-up finished rc={proc.returncode}", flush=True)
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
