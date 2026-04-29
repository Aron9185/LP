# HeaRT-Aligned Experiment Checkpoint

Date: 2026-04-29
Branch: experiments/reversegnn-compactness

## Protocol

Runs use the HeaRT `samples.npy` split files for Cora/Citeseer, full validation during training (`heart_val_frac=1.0`), Hit@10 checkpoint selection, 700 epochs, and 10 seeds. The final queued threshold-remove grid finished at `2026-04-29T07:10:50+08:00`.

## Best Results

| Method family | Cora Hit@10 | Citeseer Hit@10 | Read |
| --- | ---: | ---: | --- |
| Add-only best | 26.57 +/- 1.06 | 38.35 +/- 0.82 | Stable but weak. |
| Ratio-remove best | 27.32 +/- 1.08 | 38.79 +/- 1.02 | Best current ARON/ReverseGNN setting. |
| Edit-start sweep best | 27.06 +/- 1.15 | 38.00 +/- 1.08 | Delaying edits does not close the gap. |
| Rank-hardening sweep best | 26.85 +/- 1.02 | 37.67 +/- 1.14 | Harder decoder ranking did not help. |
| Threshold-remove sweep best | 26.83 +/- 0.96 | 37.65 +/- 0.93 | Threshold removal selected zero removals in all completed summaries. |
| BUDDY baseline | 30.34 +/- 1.02 | 48.61 +/- 1.32 | Same HeaRT split files. |
| NCNC baseline | 36.66 +/- 1.09 | 52.79 +/- 1.03 | Same HeaRT split files. |

## Final Sweep Notes

- Ratio-remove is the only follow-up with a real positive signal, but the gain over add-only is small and within roughly one standard deviation.
- Threshold-remove tried remove thresholds 0.05, 0.10, and 0.20 with max 50 removals per edit round. Every final summary reports `removed_edges_total_mean=0`, so this path did not exercise actual edge removal.
- Edit-start and rank-hardening sweeps mostly reproduce the add-only band rather than moving toward BUDDY/NCNC.
- The performance gap is therefore unlikely to be a measurement mismatch. The checkpoint preserves the measurement alignment work, but the next architecture should change the scoring/training interface.

## Next Architecture Direction

Start from a fresh commit/branch around `heart-decoder-scorer`. The main hypothesis is to make the learned decoder the final scorer, train it directly against HeaRT ranking pressure, and stop relying on dot-product decoding as the final evaluation surface.

## Local Sources

- `results/heart_hit10_diag_*_summary.csv`
- `results/heart_hit10_remove_diag_*_summary.csv`
- `results/heart_editstart_*_summary.csv`
- `results/heart_rank_*_summary.csv`
- `results/heart_remove_thr_*_summary.csv`
- `results/heart500_baselines_summary.csv`
- `results/heart_threshold_remove_diag_grid_tmux.log`
