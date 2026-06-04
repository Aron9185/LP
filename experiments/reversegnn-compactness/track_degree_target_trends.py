#!/usr/bin/env python3
"""Extract live trend rows from degree-target / decoded-rewrite sweep logs.

The training logs already contain the raw signals we care about, but they are
spread across [EDIT], [EDIT-GRAPH], [DECODED-DEG], and future [DEGREE] lines.
This script turns those lines into a flat CSV that can be refreshed while a
tmux experiment is still running.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

from experiment_paths import artifact_path, sweep_log_dir


EPOCH_RE = re.compile(r"\[E(\d{4,})\]")
KEYVAL_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)=([^\s]+)")


def _coerce(value: str):
    value = value.rstrip(",")
    if value in {"True", "False"}:
        return int(value == "True")
    if value in {"None", "nan"}:
        return float("nan")
    try:
        if any(ch in value for ch in ".eE"):
            return float(value)
        return int(value)
    except ValueError:
        return value


def _keyvals(line: str) -> dict:
    return {key: _coerce(value) for key, value in KEYVAL_RE.findall(line)}


def _log_identity(path: Path, prefix: str | None) -> dict:
    stem = path.stem
    if prefix and stem.startswith(prefix + "_"):
        suffix = stem[len(prefix) + 1 :]
    else:
        suffix = stem
    seed = None
    if "_s" in suffix:
        suffix, seed_text = suffix.rsplit("_s", 1)
        try:
            seed = int(seed_text)
        except ValueError:
            seed = None
    parts = suffix.split("_")
    dataset = parts[0] if parts else ""
    config = "_".join(parts[1:]) if len(parts) > 1 else ""
    return {
        "log": path.name,
        "dataset": dataset,
        "config": config,
        "seed": seed,
    }


def _parse_log(path: Path, prefix: str | None) -> list[dict]:
    ident = _log_identity(path, prefix)
    rows: list[dict] = []
    pending_degree_rows: list[dict] = []
    last_epoch: int | None = None

    for line_no, line in enumerate(path.read_text(errors="ignore").splitlines(), start=1):
        epoch_match = EPOCH_RE.search(line)
        if epoch_match:
            last_epoch = int(epoch_match.group(1))
            for pending in pending_degree_rows:
                if pending.get("epoch") is None:
                    pending["epoch"] = last_epoch
            pending_degree_rows.clear()

        row_type = None
        if "[EDIT][" in line:
            row_type = "edit"
        elif "[EDIT-GRAPH]" in line:
            row_type = "edit_graph"
        elif "[GRAPH-DIFF]" in line:
            row_type = "graph_diff"
        elif "[DECODED-DEG]" in line:
            row_type = "decoded_degree"
        elif "[DEGREE-CLUSTER]" in line:
            row_type = "degree_cluster"
        elif "[DEGREE]" in line:
            row_type = "degree"
        elif "[DECODER-DIAG]" in line:
            row_type = "decoder_diag"
        else:
            continue

        row = {
            **ident,
            "line": line_no,
            "row_type": row_type,
            "epoch": last_epoch,
        }
        row.update(_keyvals(line))
        if row_type == "decoded_degree" and row.get("epoch") is None:
            pending_degree_rows.append(row)
        rows.append(row)
    return rows


def _write_csv(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    preferred = [
        "log",
        "dataset",
        "config",
        "seed",
        "row_type",
        "epoch",
        "line",
        "target",
        "scope",
        "target_nodes",
        "target_mask_nodes",
        "guarantee_target",
        "repair_added",
        "need_before",
        "need_after",
        "unrepaired",
        "min_target_deg",
        "add_limit",
        "repair_limit",
        "cluster_bad_before",
        "cluster_bad_after",
        "cluster_worst_min_before",
        "cluster_worst_min_after",
        "add",
        "remove",
        "added_this_epoch",
        "removed_this_epoch",
        "min_degree",
        "mean_degree",
        "c0p_min_degree",
        "c0p_mean_degree",
        "cp_min_degree",
        "cp_mean_degree",
        "cp_clusters",
        "cp_worst_min",
        "cp_mean_min",
        "cp_need_nodes",
        "cp_bad_clusters",
        "c0p_clusters",
        "c0p_worst_min",
        "c0p_mean_min",
        "c0p_need_nodes",
        "c0p_bad_clusters",
        "base_edges",
        "view_edges",
        "added_vs_base",
        "removed_vs_base",
        "symdiff_edges",
        "edge_jaccard",
        "diff_frac_base",
        "add_frac_base",
        "remove_frac_base",
        "same_cluster_added",
        "same_cluster_removed",
        "cross_cluster_added",
        "cross_cluster_removed",
        "target_touch_added",
        "target_touch_removed",
        "add_budget_ratio",
        "remove_budget_ratio",
        "accumulate_base",
        "radius_before",
        "radius_after",
        "delta",
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
        "rewrite_applied",
        "rewrite_nodes",
        "add_selected",
        "remove_selected",
        "val_hit10",
        "test_hit10",
    ]
    keys = set()
    for row in rows:
        keys.update(row.keys())
    fieldnames = preferred + sorted(keys - set(preferred))
    with out_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _latest_summary(rows: list[dict]) -> list[dict]:
    latest: dict[tuple, dict] = {}
    for row in rows:
        key = (row.get("dataset"), row.get("config"), row.get("seed"), row.get("row_type"))
        latest[key] = row
    return list(latest.values())


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract live trend CSVs from sweep logs.")
    parser.add_argument("--prefix", type=str, help="Sweep log prefix to parse.")
    parser.add_argument("--log", action="append", default=[], help="Specific log file to parse. Repeatable.")
    parser.add_argument("--out", type=Path, default=None, help="Output CSV path.")
    parser.add_argument("--latest-out", type=Path, default=None, help="Optional latest-row CSV path.")
    args = parser.parse_args()

    if not args.prefix and not args.log:
        parser.error("provide --prefix or at least one --log")

    logs = [Path(p) for p in args.log]
    if args.prefix:
        logs.extend(sorted(sweep_log_dir().glob(f"{args.prefix}_*.txt")))
    logs = sorted(set(logs))

    rows: list[dict] = []
    for log_path in logs:
        if log_path.exists():
            rows.extend(_parse_log(log_path, args.prefix))

    out_path = args.out or artifact_path("trends", f"{args.prefix or 'selected_logs'}_trend.csv")
    _write_csv(rows, out_path)
    print(f"[trend] wrote {len(rows)} rows from {len(logs)} logs -> {out_path}")

    latest_out = args.latest_out or artifact_path("trends", f"{args.prefix or 'selected_logs'}_latest.csv")
    latest_rows = _latest_summary(rows)
    _write_csv(latest_rows, latest_out)
    print(f"[trend] wrote {len(latest_rows)} latest rows -> {latest_out}")


if __name__ == "__main__":
    main()
