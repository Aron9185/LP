"""Parse decoded rewrite audit lines from ARON sweep logs.

Example:
  python experiments/reversegnn-compactness/parse_rewrite_audit.py \
    --prefix random_rewrite_quality_current_20260607
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from experiment_paths import artifact_path, sweep_log_dir


AUDIT_RE = re.compile(r"\[REWRITE-AUDIT\]\[E(?P<epoch>\d+)\]\s+(?P<body>.*)")
PAIR_RE = re.compile(r"(?P<key>[A-Za-z0-9_]+)=(?P<value>[^\s]+)")


def _to_float(value: str) -> float | str:
    try:
        return float(value)
    except ValueError:
        return value


def _metadata_from_name(path: Path, prefix: str | None) -> dict[str, object]:
    stem = path.stem
    rest = stem
    if prefix and stem.startswith(prefix + "_"):
        rest = stem[len(prefix) + 1 :]

    seed = None
    seed_match = re.search(r"_s(\d+)$", rest)
    if seed_match:
        seed = int(seed_match.group(1))
        rest = rest[: seed_match.start()]

    dataset = None
    config = rest
    for candidate in ("citeseer", "cora", "Cora_ML", "LastFMAsia", "wisconsin", "amazon_photo"):
        marker = f"{candidate}_"
        if rest.startswith(marker):
            dataset = candidate
            config = rest[len(marker) :]
            break

    return {
        "log_path": str(path),
        "log_name": path.name,
        "dataset": dataset,
        "config": config,
        "seed": seed,
    }


def parse_log(path: Path, prefix: str | None) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    meta = _metadata_from_name(path, prefix)
    for line in path.read_text(errors="ignore").splitlines():
        match = AUDIT_RE.search(line)
        if not match:
            continue
        row = dict(meta)
        row["epoch"] = int(match.group("epoch"))
        for pair in PAIR_RE.finditer(match.group("body")):
            row[pair.group("key")] = _to_float(pair.group("value"))
        rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", default="", help="Sweep log filename prefix to parse.")
    parser.add_argument("--logs", nargs="*", default=None, help="Explicit log files. Defaults to sweep_logs/<prefix>*.txt.")
    parser.add_argument("--out", default="", help="Output CSV path. Defaults to results/<prefix>_rewrite_audit.csv.")
    args = parser.parse_args()

    if args.logs:
        paths = [Path(p) for p in args.logs]
    elif args.prefix:
        paths = sorted(sweep_log_dir().glob(f"{args.prefix}*.txt"))
    else:
        paths = sorted(sweep_log_dir().glob("*.txt"))

    rows: list[dict[str, object]] = []
    for path in paths:
        if path.exists():
            rows.extend(parse_log(path, args.prefix or None))

    out_path = Path(args.out) if args.out else artifact_path(f"{args.prefix or 'all'}_rewrite_audit.csv")
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)

    print(f"parsed_logs={len(paths)} rows={len(df)} out={out_path}")
    if not df.empty:
        key_cols = [
            "dataset",
            "config",
            "seed",
            "add_count",
            "add_test_pos",
            "add_test_neg",
            "add_cn_mean",
            "add_score_mean",
            "add_dot_mean",
        ]
        keep = [c for c in key_cols if c in df.columns]
        summary = df.groupby(["dataset", "config", "seed"], dropna=False)[keep[3:]].mean().reset_index()
        print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
