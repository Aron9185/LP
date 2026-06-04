# Post-0521 Experiment Plan

This report starts the follow-up plan after the May 21 checkpoint. The focus is no longer only whether the target-1 large-budget setting improves metrics; the next step is to verify that the cluster/orphan signal and the minimum-degree repair logic are correct.

## Next Week Plan

### GMM And Cluster-Orphan Correctness

First, validate that the GMM membership and orphan diagnostics mean what we think they mean.

Checks:

1. Test `gmm_labels` directly on synthetic embeddings:
   - clean separated clusters should be assigned confidently;
   - low-confidence boundary nodes should become `-1`;
   - graph isolation should not relabel a node as noise after GMM assignment.
2. Add or run a cluster-orphan diagnostic for each dataset/seed:
   - per-cluster CP size;
   - CP orphan count and ratio, where orphan means intra-cluster degree `0`;
   - C0p orphan count and ratio;
   - p50/p90/max orphan ratio across clusters;
   - clusters with suspiciously high orphan ratio.
3. Treat high orphan ratios as a GMM/cluster-quality warning:
   - if many nodes in the same predicted cluster have no same-cluster graph neighbors, the cluster may be too embedding-driven and not graph-consistent;
   - if only a few orphans exist, the repair algorithm should be able to handle them.
4. Add true node-level intra-degree averages to future logs:
   - `cp_intra_mean_degree`;
   - `c0p_intra_mean_degree`;
   - possibly p10/p50/p90 intra-degree for CP and C0p.

### GMM And Cluster-Orphan Diagnostic Result

Implemented and ran the reusable diagnostic:

- Script: `experiments/reversegnn-compactness/diagnose_gmm_cluster_orphans.py`

Self-test command:

```bash
/home/retro/anaconda3/envs/pyg/bin/python experiments/reversegnn-compactness/diagnose_gmm_cluster_orphans.py --self-test-gmm --datasets cora citeseer --variants no v6 --tag 20260522_cora_citeseer_no_v6
```

Self-test result:

- Clean separated embedding clusters are assigned confidently by `gmm_labels`.
- A low-confidence boundary point is assigned `-1` when the posterior threshold is high enough.
- Graph isolation cannot directly change a GMM label, because `gmm_labels(Z, K, tau, metric)` takes embeddings only and no graph/adjacency argument.

Current-method original-graph diagnostic:

- Summary CSV: `experiments/reversegnn-compactness/results/gmm_orphan_diagnostics/gmm_orphan_summary_20260522_cp_target1_repair_original_graph.csv`
- Cluster CSV: `experiments/reversegnn-compactness/results/gmm_orphan_diagnostics/gmm_orphan_clusters_20260522_cp_target1_repair_original_graph.csv`
- Aggregate CSV: `experiments/reversegnn-compactness/results/gmm_orphan_diagnostics/gmm_orphan_aggregate_20260522_cp_target1_repair_original_graph.csv`

Command:

```bash
/home/retro/anaconda3/envs/pyg/bin/python experiments/reversegnn-compactness/diagnose_gmm_cluster_orphans.py --datasets cora citeseer --variants no --seeds 0 1 2 3 4 --tag 20260522_cp_target1_repair_original_graph
```

Scope note: this diagnostic uses the latest saved seed-0-to-4 `final_gmm_labels.npy` and `final_core_mask.npy` cache artifacts after the CP target-1 repair run, measured against the original dataset graph. It does not diagnose the exact edited graph, because the training run did not save per-run edited graph snapshots in the t-SNE cache.

Aggregate current-method orphan result:

| Dataset | Runs | Global Degree-0 Nodes | CP Intra-Orphan | C0p Intra-Orphan | CP Intra Avg | C0p Intra Avg | Suspicious CP Clusters | Suspicious C0p Clusters |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Cora | 5 | 0 / 2708 = 0.00% | 8.76% | 5.93% | 2.91 | 3.22 | 3 total | 2 total |
| Citeseer | 5 | 48 / 3327 = 1.44% | 10.85% | 8.00% | 2.28 | 2.51 | 2 total | 1 total |

Per-cluster distribution:

| Dataset | Scope | Mean | Median | P75 | P90 | Max | >25% Clusters |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Cora | CP | 7.12% | 5.16% | 7.58% | 17.07% | 30.79% | 3 / 80 |
| Cora | C0p | 4.47% | 2.72% | 4.75% | 10.03% | 30.17% | 2 / 80 |
| Citeseer | CP | 8.91% | 7.70% | 12.16% | 16.93% | 29.58% | 2 / 80 |
| Citeseer | C0p | 6.18% | 4.58% | 8.15% | 12.93% | 28.43% | 1 / 80 |

Here, an intra-orphan means same-cluster intra-degree `0` under the predicted non-noise GMM cluster. Suspicious clusters use orphan ratio `>= 25%` with cluster size at least `5`.

Real-data visualization, seed 0:

```bash
/home/retro/anaconda3/envs/pyg/bin/python experiments/reversegnn-compactness/diagnose_gmm_cluster_orphans.py --plot-real-orphans --datasets cora citeseer --variants no --seeds 0 --tag 20260525_real_seed0 --plot-method tsne --max-edge-lines 6000 --max-cross-edge-lines 1000
```

Generated plots:

- ![Cora seed0 GMM orphan t-SNE](results/gmm_orphan_diagnostics/real_orphan_cora_no_seed0_tsne_20260525_real_seed0.png)
- ![Citeseer seed0 GMM orphan t-SNE](results/gmm_orphan_diagnostics/real_orphan_citeseer_no_seed0_tsne_20260525_real_seed0.png)

Seed-0 snapshot:

| Dataset | CP Intra-Orphan | C0p Intra-Orphan | Global Degree-0 | CP Intra Avg | C0p Intra Avg | Read |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Cora | 10.20% = 275 / 2697 | 7.12% = 154 / 2163 | 0.00% | 2.83 | 3.16 | orphans are not globally isolated; they are mostly nodes without same-cluster graph neighbors |
| Citeseer | 10.59% = 350 / 3306 | 7.62% = 202 / 2651 | 1.44% | 2.27 | 2.51 | same pattern, plus true global degree-0 nodes |

The t-SNE position comes from the saved embedding artifact, while the gray/orange line overlays come from the original dataset graph. Gray lines are sampled same-cluster edges; orange lines are sampled cross-cluster edges touching an intra-orphan. The pictures make the key diagnosis more concrete: many intra-orphans are not graph-isolated, they are graph-connected primarily outside their predicted GMM cluster.

Original-vs-repair visualization, seed 0:

Because the historical run used `--decoded_temporary_view_only`, the exact edited adjacency was not saved. Also, the old best checkpoints were written under a generic `no_1_best.pt` name and can be overwritten across runs. So the comparison below is a reconstructed target-1 repair view: it uses the saved final embeddings/GMM labels/C0p mask, the same CP target-1 guarantee constraints, and pulled-dot scores as the confidence tie-breaker. It should be read as a repair-mechanism visualization, not as the exact historical decoder-score graph.

```bash
/home/retro/anaconda3/envs/pyg/bin/python experiments/reversegnn-compactness/diagnose_gmm_cluster_orphans.py --plot-repair-comparison --datasets cora citeseer --variants no --seeds 0 --tag 20260525_repair_compare_seed0 --plot-method tsne --max-edge-lines 6000 --max-cross-edge-lines 1000 --max-added-edge-lines 2500 --repair-add-ratio 0.20 --repair-per-node-cap 0.10 --repair-pull-strength 0.25 --save-repair-adj
```

Generated comparison plots:

- ![Cora seed0 original vs reconstructed repair](results/gmm_orphan_diagnostics/repair_compare_cora_no_seed0_tsne_20260525_repair_compare_seed0.png)
- ![Citeseer seed0 original vs reconstructed repair](results/gmm_orphan_diagnostics/repair_compare_citeseer_no_seed0_tsne_20260525_repair_compare_seed0.png)

Artifacts:

- CSV: `experiments/reversegnn-compactness/results/gmm_orphan_diagnostics/repair_comparison_20260525_repair_compare_seed0.csv`
- Cora reconstructed adjacency: `experiments/reversegnn-compactness/results/gmm_orphan_diagnostics/repair_reconstructed_adj_cora_no_seed0_20260525_repair_compare_seed0.npz`
- Citeseer reconstructed adjacency: `experiments/reversegnn-compactness/results/gmm_orphan_diagnostics/repair_reconstructed_adj_citeseer_no_seed0_20260525_repair_compare_seed0.npz`

| Dataset | Original CP Orphan | Repaired CP Orphan | Original C0p Orphan | Repaired C0p Orphan | Original Global Degree-0 | Repaired Global Degree-0 | Added Edges | Same-Cluster Added | Removed Edges |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Cora | 10.20% | 0.00% | 7.12% | 0.00% | 0.00% | 0.00% | 645 | 645 | 0 |
| Citeseer | 10.59% | 0.00% | 7.62% | 0.00% | 1.44% | 0.03% | 799 | 799 | 0 |

Read: the reconstructed target-1 repair view removes all CP/C0p intra-orphans by adding same-cluster edges only. Citeseer still has a tiny global degree-0 remainder because target nodes are CP nodes, while some globally isolated nodes are outside CP/noise and are not repair targets.

Tail-cluster inspection:

We added a focused tail-cluster plot mode to inspect the clusters with the highest orphan ratios. The plot zooms into the selected cluster plus the orphan nodes' outside neighbors. Blue nodes are non-orphans in the focus cluster, red nodes are intra-orphans, black squares are C0p orphans, orange edges are cross-cluster edges from orphan nodes, and green edges are reconstructed target-1 repair additions.

Commands:

```bash
/home/retro/anaconda3/envs/pyg/bin/python experiments/reversegnn-compactness/diagnose_gmm_cluster_orphans.py --plot-tail-clusters --tail-include-repair --datasets cora citeseer --variants no --seeds 0 --tag 20260525_tail_seed0 --plot-method tsne --tail-top-k 4 --tail-min-cluster-size 5 --tail-max-neighbor-nodes 250 --repair-add-ratio 0.20 --repair-per-node-cap 0.10 --repair-pull-strength 0.25
/home/retro/anaconda3/envs/pyg/bin/python experiments/reversegnn-compactness/diagnose_gmm_cluster_orphans.py --plot-tail-clusters --tail-include-repair --datasets citeseer --variants no --seeds 2 --tag 20260525_tail_citeseer_seed2 --plot-method tsne --tail-top-k 4 --tail-min-cluster-size 5 --tail-max-neighbor-nodes 250 --repair-add-ratio 0.20 --repair-per-node-cap 0.10 --repair-pull-strength 0.25
```

Generated plots:

- ![Cora seed0 tail clusters](results/gmm_orphan_diagnostics/tail_clusters_cora_no_seed0_tsne_20260525_tail_seed0.png)
- ![Citeseer seed0 tail clusters](results/gmm_orphan_diagnostics/tail_clusters_citeseer_no_seed0_tsne_20260525_tail_seed0.png)
- ![Citeseer seed2 tail clusters](results/gmm_orphan_diagnostics/tail_clusters_citeseer_no_seed2_tsne_20260525_tail_citeseer_seed2.png)

Tail summary artifacts:

- `experiments/reversegnn-compactness/results/gmm_orphan_diagnostics/tail_cluster_summary_20260525_tail_seed0.csv`
- `experiments/reversegnn-compactness/results/gmm_orphan_diagnostics/tail_cluster_orphan_nodes_20260525_tail_seed0.csv`
- `experiments/reversegnn-compactness/results/gmm_orphan_diagnostics/tail_cluster_summary_20260525_tail_citeseer_seed2.csv`
- `experiments/reversegnn-compactness/results/gmm_orphan_diagnostics/tail_cluster_orphan_nodes_20260525_tail_citeseer_seed2.csv`

Highest-tail examples:

| Dataset | Seed | Cluster | CP Orphans | C0p Orphans | Orphan Global-0 | Orphan Cross-Degree Mean | Top Outside Neighbor Clusters | Read |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Cora | 0 | 1 | 30.79% = 93 / 302 | 30.17% = 73 / 242 | 0 | 1.84 | 6, 13, 7, 14, 0 | graph-connected elsewhere, not isolated |
| Cora | 0 | 6 | 24.33% = 64 / 263 | 16.59% = 35 / 211 | 0 | 1.97 | 2, 1, 11, 14, 13 | same cross-cluster-neighbor pattern |
| Citeseer | 2 | 6 | 29.58% = 113 / 382 | 28.43% = 87 / 306 | 15 | 1.07 | 9, 2, 3, noise, 4 | mixed sparse/isolation plus cross-cluster mismatch |
| Citeseer | 0 | 11 | 21.56% = 108 / 501 | 16.46% = 66 / 401 | 7 | 1.17 | 0, 1, 12, 5, 8 | mostly cross-cluster-neighbor mismatch, with some true isolates |

Diagnosis: the Cora tail is not driven by globally isolated nodes at all; the orphans have cross-cluster graph edges but no same-cluster graph edges. Citeseer is more mixed: the worst clusters include some truly global degree-0 nodes, but most orphan nodes still have outside-cluster neighbors. This supports the current interpretation that the tail clusters are embedding-consistent but graph-neighborhood-misaligned, and that target-1 repair is bridging that mismatch by adding same-cluster edges.

After-edit repair-log diagnostic:

The orphan table above is the before-edit/original-graph view. For the edited graph view, the current CP target-1 repair run logs `[DECODED-DEG]` after every repair check. In the cap-off `addr020` run, all 10 runs and all 6600 repair checks ended with `need_after=0`, `unrepaired=0`, and `cluster_bad_after=0`.

Here, a repair check is one call to the decoded-edge repair routine, not one independent experiment and not one final saved graph. Each dataset has 5 seeds. Each seed has 660 logged repair checks, giving 3300 checks per dataset. These checks cover epoch-level repair feasibility checks after `rewrite_start=100`, plus the materialized rewrite views.

| Dataset | Repair Checks | Target Nodes Avg | Need Before Avg | Need Before Rate | Need After Avg | Unrepaired After |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Cora | 3300 | 2695.9 | 328.5 | 12.19% | 0.0 | 0 |
| Citeseer | 3300 | 3302.1 | 519.5 | 15.73% | 0.0 | 0 |

Interpretation: under the temporary edited graph used by the repair pass, the CP target-1 guarantee is satisfying the intra-cluster minimum-degree condition. So after editing, the target-1 intra-orphan rate is effectively `0%` for the checked CP target mask. We still do not have a saved edited graph snapshot for later per-cluster distribution analysis, so this is a log-based after-edit guarantee check rather than an artifact-level recomputation.

Current CP target-1 edge/performance/radius comparison:

`added_edges_total` is the sum of edge additions across temporary rewrite views, not a final persistent edge count. These runs use 60 materialized rewrite views, so `mean added/view` is the easier number to interpret.

| Variant | Dataset | AUROC | AP | Hit@10 | Added Edge-Events | Mean Added/View | CP Radius | C0p Radius | Noncompact Radius | Noncompact P90 | Noncompact Max |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- | --- | --- |
| `cap010_addr020` | Cora | 95.99 | 96.40 | 71.92 +/- 5.61 | 46720.4 | 778.7 | 0.3804 -> 0.3681 | 0.3340 -> 0.3311 | 0.5151 -> 0.4751 | 0.7532 -> 0.7234 | 1.1330 -> 1.0631 |
| `capoff_addr020` | Cora | 96.02 | 96.39 | 71.46 +/- 6.08 | 53880.0 | 898.0 | 0.3758 -> 0.3657 | 0.3285 -> 0.3254 | 0.5148 -> 0.4838 | 0.7412 -> 0.7155 | 1.0680 -> 1.0128 |
| `cap010_addr020` | Citeseer | 96.43 | 96.96 | 76.79 +/- 2.18 | 48746.4 | 812.4 | 0.4335 -> 0.4196 | 0.3894 -> 0.3852 | 0.5563 -> 0.5143 | 0.7845 -> 0.7401 | 1.0946 -> 1.0428 |
| `capoff_addr020` | Citeseer | 96.48 | 96.94 | 76.00 +/- 3.93 | 49113.6 | 818.6 | 0.4329 -> 0.4221 | 0.3892 -> 0.3846 | 0.5543 -> 0.5262 | 0.8002 -> 0.7731 | 1.1200 -> 1.0773 |

Read: both current CP target-1 variants repair the target-1 intra-cluster deficits. The capped version is slightly better on Hit@10 and usually gives stronger noncompact-radius improvement, while cap-off adds more edge-events on Cora without improving final quality.

Historical cache-wide background result:

The earlier `no` vs `v6` table was a cache-wide sanity check across older saved artifacts. It is useful as background, but it should not be treated as the primary current-method result.

| Dataset | Variant | Runs | Noise Ratio | CP Orphan Ratio | C0p Orphan Ratio | CP Intra Avg | C0p Intra Avg | Suspicious CP Clusters | Suspicious C0p Clusters |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Cora | no | 15 | 0.30% | 10.01% | 7.16% | 2.84 | 3.14 | 13 total | 10 total |
| Cora | v6 | 10 | 0.25% | 11.30% | 7.98% | 2.80 | 3.11 | 12 total | 7 total |
| Citeseer | no | 15 | 0.32% | 11.03% | 8.06% | 2.27 | 2.52 | 10 total | 7 total |
| Citeseer | v6 | 10 | 0.28% | 11.60% | 8.43% | 2.26 | 2.50 | 5 total | 2 total |

Interpretation:

- The GMM calculation itself passed the direct correctness sanity checks.
- The current-method original-graph orphan signal is real but not catastrophic at the whole-run level: roughly 9-11% of CP nodes and 6-8% of C0p nodes are same-cluster orphans.
- These are not mostly global isolated nodes. Cora has 0 global degree-0 nodes; Citeseer has 48 global degree-0 nodes, about 1.44%.
- The tail is still important. Some current-method clusters reach about 28-31% intra-orphans, so a few predicted clusters are embedding-consistent but graph-sparse.
- C0p is consistently less orphan-heavy than full CP, which is good. It suggests the core selection filters out some weak graph-consistency cases, but not all of them.

Next diagnostic improvement:

- Save or log this orphan table inside target-1 runs after editing, so we can compare original graph vs edited graph directly.
- Add node-level `cp_intra_mean_degree` and `c0p_intra_mean_degree` to normal training logs; the existing `[DEGREE-CLUSTER]` mean-min values are cluster-minimum averages, not true node-level intra-degree averages.

### Synthetic Clear-Cluster Orphan Smoke Test, 2026-05-23

Implemented a synthetic smoke mode in `experiments/reversegnn-compactness/diagnose_gmm_cluster_orphans.py`.

Command:

```bash
/home/retro/anaconda3/envs/pyg/bin/python experiments/reversegnn-compactness/diagnose_gmm_cluster_orphans.py --self-test-orphans --datasets none --variants none --tag 20260523_synthetic_clear_clusters
```

Output CSV:

- `experiments/reversegnn-compactness/results/gmm_orphan_diagnostics/synthetic_orphan_smoke_20260523_synthetic_clear_clusters.csv`

Visualization:

- `experiments/reversegnn-compactness/results/gmm_orphan_diagnostics/synthetic_orphan_examples_20260525_synthetic_examples_3case.png`

Smoke result:

| Synthetic Graph | GMM Purity | Global Orphan | CP Intra-Orphan | C0p Intra-Orphan | CP Intra Avg | Read |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| connected ring inside true clusters | 100.00% | 0.00% | 0.00% | 0.00% | 2.00 | perfect cluster/graph alignment |
| ER inside true clusters, mean intra-degree 8 | 100.00% | 0.01% | 0.01% | 0.00% | 7.98 | dense enough, no orphan issue |
| ER inside true clusters, mean intra-degree 2 | 100.00% | 13.99% | 13.99% | 13.71% | 1.96 | sparse graph alone creates high intra-orphans |

Interpretation:

- A high intra-cluster orphan rate does not by itself prove GMM/CP is wrong.
- With expected intra-cluster degree near `2`, the expected zero-intra-neighbor probability is about `exp(-2) ~= 13.5%`, even when clusters are perfectly recovered.
- Our observed original-graph rates, roughly Cora `8.76%` CP and Citeseer `10.85%` CP on the latest seed-0-to-4 caches, are therefore plausible for sparse graphs.
- The stronger warning is not the overall 8-11% rate; it is the per-cluster tail where some clusters reach 30-40%+ intra-orphans.

### Minimum-Degree Guarantee Correctness

Second, audit the guarantee repair logic itself.

Current guarantee path:

- target deficits are computed from the selected degree scope, e.g. `intra_cluster`;
- candidate edges are filtered by the active endpoint policy;
- candidate priority is deficit first:
  - larger endpoint deficit;
  - larger pair deficit sum;
  - decoded graph confidence as the tie-breaker;
- if `--decoded_guarantee_degree_target` is enabled, the repair pass can exceed `add_ratio` and per-node caps, but it still respects the candidate/endpoint constraints unless those constraints are explicitly changed.

Correctness questions:

1. If `target_nodes=cp`, should the repair pass still obey `C0p-to-noncompact CP` endpoint constraints?
   - Current behavior can fail to repair full CP membership even when same-cluster CP-to-CP candidates exist.
   - We should test a repair-only relaxation: for guarantee repair, allow same-cluster CP-to-CP edges; for leftover augmentation, keep the C0p-to-noncompact endpoint policy.
2. Does the deficit-first ordering ever starve a node under limited budget or endpoint constraints?
   - Unit-test toy clusters where multiple target nodes compete for the same high-score partner.
   - Compare current greedy ordering against a round-robin deficit repair or simple bipartite/matching-style repair.
3. Are unrepaired nodes explainable?
   - Log counters for `no_valid_candidate`, `blocked_by_endpoint_rule`, `blocked_by_max_add`, `blocked_by_cap`, and `already_satisfied`.
   - Report unrepaired nodes by cluster, target scope, and endpoint type.

Expected deliverable: a small correctness test suite plus a per-run orphan/repair diagnostic table. Only after this should we decide whether the endpoint rule, target scope, or tie-breaker should change.

## Autoencoder Backbone Comparison, 2026-05-27

After the CP target-1 repair and GMM/orphan diagnostics, we tested whether replacing the VGNAE backbone with MaskGAE or CIMAGE-style masked autoencoding improves the same graph-editing recipe.

Backbone method differences:

| Backbone | Training signal | Cluster/factor signal | Relation to our editor |
| --- | --- | --- | --- |
| VGNAE current baseline | Variational graph autoencoder reconstruction with KL regularization; APPNP-style propagation; dot/prediction decoder trained for link ranking | No extra pseudo-label or masked-feature objective | Best aligned with our existing decoded-edge scorer and GMM CP/C0p repair pipeline |
| MaskGAE | Deterministic masked-feature autoencoder; feature masking plus structure reconstruction; KL disabled | No explicit cluster/factor objective | Adds denoising but can blur the pairwise link-ranking signal |
| CIMAGE-full | Factorized masked autoencoder with edge masking, modularity-style soft pseudo-label clustering, and factor reconstruction; KL disabled | Explicit latent factors plus pseudo-label/modularity signal | Encourages community/factor structure, but this is not yet aligned with the GMM CP/C0p clusters used by graph editing |

Performance summary:

| Backbone | Dataset | Seeds | AUROC | AUPRC/AP | Hit@10 | Read |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| VGNAE CP target-1 repair | Cora | 5/5 | 95.99 | 96.40 | 71.92 +/- 5.61 | strongest current baseline |
| VGNAE CP target-1 repair | Citeseer | 5/5 | 96.43 | 96.96 | 76.79 +/- 2.18 | strongest current baseline |
| MaskGAE backbone | Cora | 5/5 | 94.46 | 95.41 | 67.59 +/- 2.60 | moderate drop from VGNAE |
| MaskGAE backbone | Citeseer | 5/5 | 94.80 | 95.72 | 71.52 +/- 3.62 | moderate drop from VGNAE |
| CIMAGE-full backbone | Cora | 3/3 | 91.91 | 92.79 | 60.22 +/- 8.68 | completed rerun; clear underperformance |
| CIMAGE-full backbone | Citeseer | 3/3 | 89.54 | 92.32 | 63.22 +/- 3.74 | clear underperformance |

Important CIMAGE caveat:

- Cora seed 0 in the first full CIMAGE run crashed at epoch 591 with a CUDA `nll_loss` target-range assert.
- The cause was a prototype compactness target-index bug: skipped small clusters left gaps in class ids passed to `cross_entropy`.
- Fixed in `src/aron_train_edit_decoder.py` by assigning prototype targets from the compacted prototype list index.
- Rerun completed on May 27:
  - tmux session: `cimage_full_cora_s0_rerun_20260527`
  - prefix: `random_two_decoder_cimage_full_backbone_c0p_noncompact_soft025_cap010_cp_dtarget1_guarantee_addr020_cora_s0_rerun_20260527`
  - scope: Cora seed 0, 700 epochs
  - result: AUROC 92.05, AUPRC/AP 92.28, Hit@10 53.51

CIMAGE-full Cora seed breakdown:

| Seed | AUROC | AUPRC/AP | Hit@10 | Best Val Epoch |
| ---: | ---: | ---: | ---: | ---: |
| 0 rerun | 92.05 | 92.28 | 53.51 | 139 |
| 1 | 92.20 | 93.91 | 70.02 | 195 |
| 2 | 91.47 | 92.20 | 57.12 | 161 |
| Mean | 91.91 | 92.79 | 60.22 +/- 8.68 | - |

Read:

- MaskGAE helps test masked-denoising as a backbone, but in this pipeline it does not improve link prediction. It loses about 4.3 Hit@10 on Cora and 5.3 Hit@10 on Citeseer relative to the VGNAE CP target-1 repair baseline.
- CIMAGE-full contributes a stronger cluster/factor prior, but the current version is worse than both VGNAE and MaskGAE. After the Cora seed-0 rerun, it trails the VGNAE CP target-1 repair baseline by about 11.7 Hit@10 on Cora and 13.6 Hit@10 on Citeseer. The main mismatch is likely objective alignment: CIMAGE optimizes modularity-style pseudo-label/factor structure, while our editor still chooses CP/C0p using GMM clusters and evaluates edge ranking.
- The graph repair mechanism still works under CIMAGE: the repair logs reach target-1 intra-cluster degree with `need_after=0` and `unrepaired=0`. The performance drop is therefore more likely from the embedding/scoring backbone than from the CP target-1 repair logic.
- For now, VGNAE remains the main backbone. MaskGAE and CIMAGE should be treated as ablations, not replacements, until a cluster-objective ablation shows otherwise.

Immediate follow-up:

1. Run `cimage_cluster_weight=0` while keeping factor reconstruction on, to test whether the modularity pseudo-label clustering is helping or hurting.
2. If CIMAGE still underperforms, stop backbone replacement work and return to the VGNAE path for endpoint/repulsion experiments.
