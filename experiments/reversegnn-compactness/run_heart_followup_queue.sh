#!/usr/bin/env bash
set -euo pipefail

ROOT="/home/retro/ARON"
PY="/home/retro/anaconda3/envs/pyg/bin/python"
LOG_DIR="${ROOT}/experiments/reversegnn-compactness/results"
LOG_FILE="${LOG_DIR}/heart_followup_queue_tmux.log"
WAIT_SESSION="${WAIT_SESSION:-heart_editstart_diag_10seed}"
POLL_SECONDS="${POLL_SECONDS:-300}"
EPOCHS="${EPOCHS:-700}"
MAX_WORKERS="${MAX_WORKERS:-1}"

mkdir -p "${LOG_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

cd "${ROOT}"

echo "[followup-queue] started $(date -Is)"
echo "[followup-queue] waiting_for=${WAIT_SESSION}"
echo "[followup-queue] poll_seconds=${POLL_SECONDS}"
echo "[followup-queue] epochs=${EPOCHS}"
echo "[followup-queue] max_workers=${MAX_WORKERS}"

while tmux has-session -t "${WAIT_SESSION}" 2>/dev/null; do
  echo "[followup-queue] $(date -Is) still waiting for ${WAIT_SESSION}"
  sleep "${POLL_SECONDS}"
done

echo "[followup-queue] ${WAIT_SESSION} finished or not present at $(date -Is)"

BEST_EDIT_START="$("${PY}" - <<'PY'
import glob
import pathlib
import re

import pandas as pd

rows = []
root = pathlib.Path("/home/retro/ARON/experiments/reversegnn-compactness/results")
for path in sorted(root.glob("heart_editstart_es*_summary.csv")):
    m = re.search(r"heart_editstart_es(?P<es>\d+)_p(?P<pull>\d+p\d+)_a(?P<add>\d+p\d+)_summary", path.name)
    if not m:
        continue
    run_path = path.with_name(path.name.replace("_summary.csv", "_runs.csv"))
    if not run_path.exists():
        continue
    try:
        if len(pd.read_csv(run_path)) != 20:
            continue
        summary = pd.read_csv(path)
    except Exception:
        continue
    for _, row in summary.iterrows():
        rows.append(
            {
                "edit_start": int(m.group("es")),
                "add": float(m.group("add").replace("p", ".")),
                "dataset": row["dataset"],
                "hit10": float(row["test_hit10_mean"]),
            }
        )

if not rows:
    print(100)
    raise SystemExit

df = pd.DataFrame(rows)
# Pick the edit-start whose best add setting per dataset has the best average.
best_by_dataset = (
    df.groupby(["edit_start", "dataset"], as_index=False)["hit10"]
    .max()
)
score = (
    best_by_dataset.groupby("edit_start", as_index=False)["hit10"]
    .mean()
    .sort_values(["hit10", "edit_start"], ascending=[False, True])
)
print(int(score.iloc[0]["edit_start"]))
PY
)"

echo "[followup-queue] selected_edit_start=${BEST_EDIT_START}"
echo "[followup-queue] starting rank grid at $(date -Is)"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  EPOCHS="${EPOCHS}" \
  EDIT_START="${BEST_EDIT_START}" \
  MAX_WORKERS="${MAX_WORKERS}" \
  bash "${ROOT}/experiments/reversegnn-compactness/run_heart_rank_diag_grid.sh"

echo "[followup-queue] rank grid finished at $(date -Is)"
echo "[followup-queue] starting threshold-remove grid at $(date -Is)"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" \
  EPOCHS="${EPOCHS}" \
  EDIT_START="${BEST_EDIT_START}" \
  MAX_WORKERS="${MAX_WORKERS}" \
  bash "${ROOT}/experiments/reversegnn-compactness/run_heart_threshold_remove_diag_grid.sh"

echo "[followup-queue] threshold-remove grid finished at $(date -Is)"
echo "[followup-queue] finished $(date -Is)"
