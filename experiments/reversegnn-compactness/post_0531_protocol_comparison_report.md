# Protocol Comparison Report, 2026-05-31

This report separates the ReverseGNN, CIMAGE, and MaskGAE comparisons by evaluation protocol. The main result should be read under the ARON no-leak setting. Full-graph results are useful diagnostics, but they should not be mixed into the same leaderboard as no-leak edge prediction.

## Protocols

| Protocol | Train graph | Split/eval source | Leakage status | Main use |
| --- | --- | --- | --- | --- |
| ARON no-leak | Held-out validation/test positive edges removed | ARON cached random split, 5% validation positives and 10% test positives, cached negatives | No edge leakage | Main fair comparison |
| ARON split with full-graph training | Full original graph visible to encoder/training | Same ARON held-out edges and negatives | Leaky for link prediction | Diagnostic for protocol sensitivity |
| CIMAGE paper/default reference | Public CIMAGE code uses PyG `RandomLinkSplit(num_val=0.1, num_test=0.05)` and full-graph training | PyG split, not ARON cached split | Leaky under the public code behavior | Reference to explain WSDM/public-code comparability |

All reported values below are percentages. AP is the same metric we have also called AUPRC in prior notes.

## Main Result: ARON No-Leak

Matched 3-seed comparison using seeds `0,1,2`.

| Method | Dataset | AUROC | AP / AUPRC | Hit@10 |
| --- | ---: | ---: | ---: | ---: |
| ReverseGNN | Cora | 95.93 +/- 0.59 | 96.51 +/- 0.38 | 74.70 +/- 2.78 |
| CIMAGE authors code | Cora | 86.42 +/- 0.27 | 88.52 +/- 0.24 | 47.00 +/- 2.86 |
| MaskGAE Edge authors code | Cora | 96.83 +/- 0.14 | 97.10 +/- 0.18 | 75.71 +/- 4.68 |
| ReverseGNN | Citeseer | 95.98 +/- 0.12 | 96.65 +/- 0.10 | 75.31 +/- 1.11 |
| CIMAGE authors code | Citeseer | 89.94 +/- 1.41 | 92.10 +/- 1.05 | 61.98 +/- 3.96 |
| MaskGAE Edge authors code | Citeseer | 96.87 +/- 0.15 | 97.33 +/- 0.09 | 78.24 +/- 0.95 |

Read:

- Under the fair ARON no-leak split, ReverseGNN is much stronger than CIMAGE on both datasets.
- ReverseGNN is close to official MaskGAE Edge, especially on Hit@10. MaskGAE Edge is slightly higher overall in this matched 3-seed table.
- The older ReverseGNN baseline we quoted was a 5-seed result: Cora `95.99 +/- 0.51` AUROC, `96.40 +/- 0.46` AP, `71.92 +/- 5.61` Hit@10; Citeseer `96.43 +/- 0.64` AUROC, `96.96 +/- 0.43` AP, `76.79 +/- 2.18` Hit@10. The table above uses 3 seeds only to match the official CIMAGE/MaskGAE ARON runs.

## Diagnostic: ARON Split With Full-Graph Leakage

Same ARON held-out edges and negatives, but the encoder/training graph includes the full graph. This means held-out positive edges are visible structurally.

| Method | Dataset | AUROC | AP / AUPRC | Hit@10 |
| --- | ---: | ---: | ---: | ---: |
| ReverseGNN | Cora | 99.96 +/- 0.03 | 99.95 +/- 0.03 | 100.00 +/- 0.00 |
| CIMAGE authors code | Cora | 97.23 +/- 0.38 | 97.80 +/- 0.31 | 87.48 +/- 1.74 |
| MaskGAE Edge authors code | Cora | 99.85 +/- 0.06 | 99.82 +/- 0.08 | 99.68 +/- 0.55 |
| ReverseGNN | Citeseer | 99.94 +/- 0.01 | 99.93 +/- 0.01 | 100.00 +/- 0.00 |
| CIMAGE authors code | Citeseer | 99.21 +/- 0.23 | 99.29 +/- 0.17 | 93.63 +/- 1.01 |
| MaskGAE Edge authors code | Citeseer | 99.87 +/- 0.01 | 99.81 +/- 0.03 | 100.00 +/- 0.00 |

Read:

- Full-graph visibility makes the task almost saturated for ReverseGNN and MaskGAE Edge.
- CIMAGE also improves sharply under full-graph visibility, but still trails on Hit@10, especially on Cora.
- This table should be described as a leakage diagnostic, not as the main fair comparison.

## CIMAGE WSDM Paper Context

The CIMAGE WSDM paper reports link-prediction results on Cora and Citeseer as follows:

| Source | Dataset | AUROC | AP |
| --- | ---: | ---: | ---: |
| CIMAGE WSDM Table 2 | Cora | 96.93 +/- 0.16 | 96.76 +/- 0.15 |
| CIMAGE WSDM Table 2 | Citeseer | 97.90 +/- 0.52 | 97.96 +/- 0.60 |

Important caveat:

- The paper text says link prediction uses learned node representations with 85% of edges and an equal number of sampled non-existing edges for AUC/AP evaluation.
- The public CIMAGE link-prediction code uses PyG `RandomLinkSplit(num_val=0.1, num_test=0.05)` and trains on the full `data` graph, not the split train graph.
- Therefore, our ARON full-graph leakage table is not a reproduction of the WSDM paper setting. It is an ARON-split diagnostic.
- The public-code/default numbers below should be called a protocol reference, not ARON no-leak.

## Reference: CIMAGE Public-Code/Default Protocol

This setting uses the CIMAGE public-code-style split/training behavior: PyG `RandomLinkSplit(num_val=0.1, num_test=0.05)` plus full-graph training visibility. It is useful for explaining why public-code/default results are much higher, but it is leaky for link prediction and should not be mixed with the ARON no-leak leaderboard.

Matched 3-seed comparison using seeds `0,1,2` where available.

| Method | Dataset | Runs | AUROC | AP / AUPRC | Hit@10 | Read |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| ReverseGNN | Cora | 3 | 99.97 +/- 0.03 | 99.97 +/- 0.04 | 100.00 +/- 0.00 | our method under CIMAGE public-code/default protocol |
| CIMAGE authors code | Cora | 3 | 96.73 +/- 1.16 | 97.37 +/- 0.93 | 86.19 +/- 4.17 | public-code/default reference |
| ReverseGNN | Citeseer | 3 | 99.75 +/- 0.28 | 99.60 +/- 0.52 | 100.00 +/- 0.00 | our method under CIMAGE public-code/default protocol |
| CIMAGE authors code | Citeseer | 3 | 99.43 +/- 0.13 | 99.49 +/- 0.09 | 96.62 +/- 0.67 | public-code/default reference |

Read:

- ReverseGNN also saturates under the CIMAGE public-code/default protocol, as expected when the training graph has full-graph visibility.
- CIMAGE authors code gets close to the WSDM/public-code regime on Citeseer, but Cora remains lower than the single best-run value we first saw from the default script.
- MaskGAE is not listed in this exact table yet because we have not run a matched 3-seed MaskGAE Edge job under this CIMAGE public-code/default protocol. The existing MaskGAE author-default clean run is useful background, but it is not the same matched 3-seed comparison.
- This table should answer protocol comparability, not replace the ARON no-leak table as the fair comparison.

## Backbone Replacement Status

We have already tested replacing the current VGNAE backbone inside our ReverseGNN/editor pipeline with both MaskGAE and CIMAGE-full style backbones.

| Backbone inside our editor | Dataset | Seeds | AUROC | AP / AUPRC | Hit@10 | Read |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| VGNAE CP target-1 repair | Cora | 5/5 | 95.99 | 96.40 | 71.92 +/- 5.61 | current main baseline |
| VGNAE CP target-1 repair | Citeseer | 5/5 | 96.43 | 96.96 | 76.79 +/- 2.18 | current main baseline |
| MaskGAE backbone | Cora | 5/5 | 94.46 | 95.41 | 67.59 +/- 2.60 | underperforms VGNAE |
| MaskGAE backbone | Citeseer | 5/5 | 94.80 | 95.72 | 71.52 +/- 3.62 | underperforms VGNAE |
| CIMAGE-full backbone | Cora | 3/3 | 91.91 | 92.79 | 60.22 +/- 8.68 | clear underperformance |
| CIMAGE-full backbone | Citeseer | 3/3 | 89.54 | 92.32 | 63.22 +/- 3.74 | clear underperformance |

Read:

- Yes, we already replaced VGNAE with CIMAGE-full inside our pipeline and completed the Cora seed-0 rerun after fixing the prototype compactness target-index bug.
- We also already tried a MaskGAE backbone inside the editor pipeline. It underperformed the current VGNAE baseline, even though official MaskGAE Edge is strong when run as its own standalone link-prediction method under ARON no-leak.
- This suggests the issue is not that MaskGAE is weak. The issue is likely integration: the current editor, CP/C0p selection, GMM clusters, and pairwise link-ranking objective are better aligned with the VGNAE embedding/scoring path.

## Final Takeaway

The final report should make the ARON no-leak comparison the main claim:

> Under the fair ARON no-leak split, ReverseGNN substantially outperforms CIMAGE and is competitive with official MaskGAE Edge. Under full-graph leakage, ReverseGNN and MaskGAE Edge become nearly saturated, showing that full-graph visibility strongly inflates link-prediction results. CIMAGE WSDM/public-code results should be reported as a protocol reference, not mixed with ARON no-leak results.

For the next backbone experiment, the most useful direction is not a blind replacement with MaskGAE again. We should either:

1. keep VGNAE as the main backbone and continue endpoint/repulsion experiments; or
2. run a focused MaskGAE-integration ablation that asks why standalone official MaskGAE Edge is strong but our integrated MaskGAE backbone is weaker.

## Source Files

- ReverseGNN ARON no-leak: `experiments/reversegnn-compactness/results/random_two_decoder_cp_target1_repair_c0p_noncompact_soft025_cap010_cp_dtarget1_guarantee_addr020_20260520_runs.csv`
- ReverseGNN ARON full-graph leakage: `experiments/reversegnn-compactness/results/random_two_decoder_fullgraph_leakage_current_c0p_noncompact_soft025_cap010_cp_dtarget1_guarantee_addr020_20260529_summary.csv`
- ReverseGNN CIMAGE paper/default reference: `experiments/reversegnn-compactness/results/random_two_decoder_cimage_paper_current_c0p_noncompact_soft025_cap010_cp_dtarget1_guarantee_addr020_20260530_summary.csv`
- Official CIMAGE/MaskGAE ARON no-leak: `official_baselines/runs/20260528_aron_split/summary.md`
- Official CIMAGE/MaskGAE ARON split full-graph leakage: `official_baselines/runs/20260529_aron_split_fullgraph_leakage/`
- Official CIMAGE public-code/default: `/home/retro/official_baselines/runs/20260531_cimage_paper_multiseed_fixed/`
- Integrated backbone comparison: `experiments/reversegnn-compactness/post_0521_experiment_report.md`
- CIMAGE WSDM/arXiv reference: `https://ar5iv.org/html/2503.07852v1`
