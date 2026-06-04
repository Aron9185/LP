# Post-5/14 Experiment Report

This report tracks the work after the May 14 checkpoint. It assumes the earlier post-4/30 report as background and focuses on the continuation: radius diagnostics, the no-add/no-compact confirmation, and the next augmentation/pulling redesign.

## Starting Point On 5/14

The locked random-split default before this continuation was:

- config: `two_decoder_pred`
- edit decoder: `pair_mlp_struct`
- prediction head: `pair_residual_struct`
- score source: `pred_decoder`
- decoded add ratio: `0.01`
- compactness weight: `0.2`
- removal: disabled

Fair 10-seed baseline:

| Dataset | ROC-AUC | AP | Hit@10 |
|---|---:|---:|---:|
| Cora | 95.95 +/- 0.40 | 96.32 +/- 0.45 | 70.49 +/- 5.11 |
| Citeseer | 96.16 +/- 0.55 | 96.59 +/- 0.46 | 73.56 +/- 2.73 |

The pre-5/14 diagnosis was:

1. The separate structure-aware prediction head is the main gain.
2. Removal should stay disabled.
3. Decoded additions and compactness are questionable.
4. `add000_compact000` is the candidate to confirm before changing the default.

Metric note: in these reports, `ROC-AUC` is AUROC. `AP` is average precision, i.e. the AUPRC-style area/summary used by the existing evaluation CSVs.

## Work Since 5/14

### Radius Diagnostics

We added and exercised Mahalanobis radius accounting through `--compactness_radius_metric=mahalanobis`, while also logging compact and noncompact radius summaries.

The 1-epoch Cora smoke passed with the new radius metric, confirming that parser wiring and runtime logging worked.

### Mahalanobis Add/Compact Diagnostic

We reran the current add+compact setting and the `add000_compact000` candidate on seeds `0-4` for Cora and Citeseer.

Setup:

- split: random
- config: `two_decoder_pred`
- edit decoder: `pair_mlp_struct`
- prediction head: `pair_residual_struct`
- score source: `pred_decoder`
- radius metric: `mahalanobis`

| Dataset | Variant | Add Ratio | Compact W | ROC-AUC | AP | Hit@10 | Delta vs Current | Edit-Decoder Test Hit@10 | Added Edges |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Cora | current | 0.01 | 0.2 | 95.86 +/- 0.47 | 96.15 +/- 0.56 | 69.64 +/- 6.57 | +0.00 | 30.93 +/- 2.69 | 2700 |
| Cora | `add000_compact000` | 0.00 | 0.0 | 95.89 +/- 0.44 | 96.19 +/- 0.53 | 69.87 +/- 5.04 | +0.23 | 63.30 +/- 8.23 | 0 |
| Citeseer | current | 0.01 | 0.2 | 96.22 +/- 0.55 | 96.67 +/- 0.40 | 72.84 +/- 2.98 | +0.00 | 14.07 +/- 0.66 | 2340 |
| Citeseer | `add000_compact000` | 0.00 | 0.0 | 96.22 +/- 0.60 | 96.67 +/- 0.40 | 73.98 +/- 1.86 | +1.14 | 60.92 +/- 2.46 | 0 |

Conclusion: Mahalanobis accounting does not rescue the current add+compact setting. The no-add/no-compact candidate remains tied or better on ROC-AUC/AP/Hit@10, and the edit-decoder diagnostic is much healthier without decoded additions.

This is still a 5-seed diagnostic, not a replacement for the 10-seed confirmation.

Source artifacts:

- `results/random_two_decoder_mahalanobis_current_20260515b_summary.csv`
- `results/random_two_decoder_mahalanobis_add000_compact000_20260515b_summary.csv`
- `results/mahalanobis_diag_20260515b_driver.log`

## 10-Seed No-Add/No-Compact Confirmation

The 10-seed confirmation for `add000_compact000` finished successfully on May 16.

Setup:

- datasets: Cora, Citeseer
- seeds: `0-9`
- split: random
- config: `two_decoder_pred`
- decoded add ratio: `0.0`
- compactness weight: `0.0`
- removal: disabled
- workers: `1`

| Dataset | Method | ROC-AUC | AP | Hit@10 | Delta vs Current |
|---|---|---:|---:|---:|---:|
| Cora | current add+compact | 95.95 +/- 0.40 | 96.32 +/- 0.45 | 70.49 +/- 5.11 | +0.00 |
| Cora | `add000_compact000` | 95.96 +/- 0.37 | 96.30 +/- 0.41 | 69.92 +/- 5.11 | -0.57 |
| Citeseer | current add+compact | 96.16 +/- 0.55 | 96.59 +/- 0.46 | 73.56 +/- 2.73 | +0.00 |
| Citeseer | `add000_compact000` | 96.13 +/- 0.79 | 96.54 +/- 0.62 | 72.18 +/- 3.56 | -1.38 |

Conclusion: `add000_compact000` does not hold as a 10-seed replacement. The 5-seed signal was useful as a diagnostic, but the locked add+compact default remains better on final Hit@10, especially on Citeseer.

The no-add/no-compact run still gives an important control: the edit-decoder diagnostic looks much healthier without augmentation, but final prediction quality is worse. That means the current augmentation/pulling path may be noisy, yet it is not safe to remove it outright.

Source artifacts:

- `results/random_two_decoder_add000_compact000_confirm10_20260515_summary.csv`
- `results/random_two_decoder_add000_compact000_confirm10_20260515_runs.csv`
- `results/add000_compact000_confirm10_20260515_driver.log`
- `run_random_two_decoder_add000_compact000_confirm10_20260515.sh`

## Current Diagnosis

The best reading after 5/14 is not "augmentation and pulling are useless." The 10-seed confirmation makes the diagnosis narrower:

1. The current add+compact recipe should stay as the locked default for now.
2. Plainly removing decoded additions and compactness is too simple; it loses final Hit@10 at 10 seeds.
3. The edit-decoder diagnostic is not enough: `add000_compact000` improves decoder-looking behavior but loses final prediction quality.
4. The next improvement should redesign pulling and endpoint selection, not just delete augmentation or increase compactness.
5. We should keep the no-add/no-compact run as a clean control while testing better augmentation/pulling mechanisms.

## Next Work

1. Improve augmentation as a separate mechanism:

   Measure proposal quality, selected-edge score distributions, endpoint types, cluster membership, per-node caps, and whether selected edges overlap with held-out positives when that is valid to inspect.

2. Improve pulling as a separate mechanism:

   Test pull target choice, pull mask scope, pull schedule, encoder coupling, and whether noncompact nodes are being moved in the intended direction.

3. Prioritize soft-pull and endpoint-rule variants:

   The strongest next hypothesis is that current pull/endpoint behavior is too crude. Test softer pull strengths, C0p-to-noncompact endpoint rules, and per-node caps before changing the prediction head.

4. Only promote an augmentation or pulling change if it improves final ROC-AUC/AP/Hit@10, not just edit-decoder diagnostic Hit@10 or radius movement.

## Pull/Endpoint Grid Started

We added a new endpoint rule:

- `--decoded_require_c0p_noncompact_endpoint`

This keeps the current same-cluster filtering but requires each decoded edge to connect one C0p anchor with one non-C0p CP partner. The goal is to test anchor-to-uncertain-node augmentation instead of simply adding more C0p/core-adjacent edges.

The one-epoch Cora smoke passed with the new endpoint rule.

Planned 5-seed grid:

| Variant | Pull Strength | Endpoint Rule | Per-Node Cap |
|---|---:|---|---:|
| `current_pull100` | 1.00 | current same-cluster + C0p endpoint | disabled |
| `softpull025` | 0.25 | current same-cluster + C0p endpoint | disabled |
| `softpull010` | 0.10 | current same-cluster + C0p endpoint | disabled |
| `c0p_noncompact_soft025` | 0.25 | same-cluster C0p-to-noncompact CP | disabled |
| `c0p_noncompact_soft025_cap010` | 0.25 | same-cluster C0p-to-noncompact CP | 0.10 |

Launcher:

- `run_random_two_decoder_pull_endpoint_grid_20260516.sh`

## Pull/Endpoint Grid Result

The pull/endpoint grid finished on May 17.

This grid actually answers two separate questions, so the results are split below.

### Pull Strength Sweep

This comparison holds the endpoint rule fixed at same-cluster + C0p endpoint and keeps the per-node cap disabled.

| Variant | Pull Strength | Cora AUROC | Cora AUPRC/AP | Cora Hit@10 | Citeseer AUROC | Citeseer AUPRC/AP | Citeseer Hit@10 | Read |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `current_pull100` | 1.00 | 96.04 | 96.28 | 69.60 +/- 6.38 | 96.30 | 96.73 | 73.41 +/- 2.01 | local baseline |
| `softpull025` | 0.25 | 96.06 | 96.33 | 68.39 +/- 7.73 | 96.42 | 96.79 | 71.74 +/- 2.31 | worse |
| `softpull010` | 0.10 | 96.33 | 96.59 | 71.46 +/- 7.84 | 96.53 | 96.76 | 70.90 +/- 0.94 | Cora-only gain |

Softer pull does not give a clean cross-dataset win under the old endpoint rule. Pull `0.10` helps Cora, but Citeseer drops.

### Endpoint And Cap Sweep

This comparison holds pull strength at `0.25` and uses `softpull025` as the bridge baseline.

| Variant | Endpoint Rule | Per-Node Cap | Cora AUROC | Cora AUPRC/AP | Cora Hit@10 | Citeseer AUROC | Citeseer AUPRC/AP | Citeseer Hit@10 | Read |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| `softpull025` | same-cluster + C0p endpoint | disabled | 96.06 | 96.33 | 68.39 +/- 7.73 | 96.42 | 96.79 | 71.74 +/- 2.31 | bridge baseline |
| `c0p_noncompact_soft025` | same-cluster C0p-to-noncompact CP | disabled | 96.07 | 96.35 | 69.15 +/- 7.57 | 96.59 | 96.99 | 75.03 +/- 2.46 | Citeseer gain, Cora partial recovery |
| `c0p_noncompact_soft025_cap010` | same-cluster C0p-to-noncompact CP | 0.10 | 96.17 | 96.45 | 70.06 +/- 7.19 | 96.53 | 96.96 | 75.08 +/- 2.31 | best balanced 5-seed result |

The capped C0p-to-noncompact endpoint variant is the best balanced candidate so far. It improves Hit@10 on both datasets relative to `current_pull100`, improves more clearly relative to the `softpull025` bridge baseline, and uses fewer added edges than the uncapped local baseline:

- Cora: `1851.6` added edges versus `2700`
- Citeseer: `1428.8` added edges versus `2340`

This supports the diagnosis that the old decoded additions were too broad. Pull strength alone is not the main answer; the endpoint redesign makes the decoder diagnostic much healthier, while the per-node cap helps prevent the Cora drop seen in the uncapped version.

Source artifacts:

- `results/random_two_decoder_pull_endpoint_current_pull100_20260516_summary.csv`
- `results/random_two_decoder_pull_endpoint_softpull010_20260516_summary.csv`
- `results/random_two_decoder_pull_endpoint_softpull025_20260516_summary.csv`
- `results/random_two_decoder_pull_endpoint_c0p_noncompact_soft025_20260516_summary.csv`
- `results/random_two_decoder_pull_endpoint_c0p_noncompact_soft025_cap010_20260516_summary.csv`
- `results/pull_endpoint_grid_20260516_driver.log`

## Degree-Targeted Decoded Additions

We added an opt-in decoded addition policy:

- `--decoded_add_degree_target K`

When enabled, decoded additions first spend their add budget on valid candidate edges touching rewrite-mask nodes whose current degree is below `K`. Candidate priority is deficit-first, then decoder-score tie-break. This adapts the old `ver=aron_desc` "fill degree deficit under budget" idea to the decoded graph rewrite path without changing the default behavior.

The first planned test is to apply this to the current best endpoint setting:

- pull strength: `0.25`
- endpoint rule: C0p-to-noncompact CP
- per-node cap: `0.10`
- degree targets: `2`, `3`, `4`

Launcher:

- `run_random_two_decoder_degree_target_grid_20260517.sh`

Clarification after inspecting the first degree-target logs: the real target should be minimum **intra-cluster** degree, not just total node degree. A node can have enough total edges while still being isolated inside its GMM cluster. The next clean variant should therefore use:

- `--decoded_add_degree_target_scope intra_cluster`

This keeps the current total-degree behavior available for comparison while letting the degree-target selector prioritize same-cluster induced degree. We also added cluster-min diagnostics (`[DEGREE-CLUSTER]`) for worst cluster minimum degree, mean cluster minimum degree, below-target nodes, and below-target clusters.

## Intra-Cluster Budget Grid Plan

The first degree-target run shows that a 1% add budget is too small to make minimum degree move meaningfully. For a cleaner comparison, the next budget grid isolates one target and varies only the topology budget:

| Variant | Degree Scope | Target | Add Ratio | Feature Mask |
|---|---|---:|---:|---:|
| `intra_dtarget2_addr001` | intra-cluster | 2 | 0.01 | 0.10 |
| `intra_dtarget2_addr003` | intra-cluster | 2 | 0.03 | 0.10 |
| `intra_dtarget2_addr005` | intra-cluster | 2 | 0.05 | 0.10 |

Launcher:

- `run_random_two_decoder_intra_degree_budget_grid_20260518.sh`

The new `[GRAPH-DIFF]` diagnostic logs how far each generated augmentation view moves from the pre-rewrite graph: edge Jaccard, symmetric-difference edges, add/remove fraction of the base graph, same-cluster vs cross-cluster changed edges, and target-touching changed edges. For accumulated decoded rewrites, it also logs the cumulative difference from the edit-start graph.

Feature note: the decoded graph editor changes topology. However, the current runner inherits `aron_main.py`'s default `--feat_mask_ratio 0.1`, so the contrastive view also uses 10% feature dropout unless we explicitly pass `--feat_mask_ratio 0.0`. The new budget grid passes the feature-mask ratio explicitly so this is visible in the command and easy to ablate later.

## Target-1 Large-Budget Feasibility Test

To test whether temporary augmentation can satisfy a very weak intra-cluster degree target without accumulating edits, run target `1` with a much larger add budget:

| Variant | Degree Scope | Target | Add Ratio | Per-Node Cap | Purpose |
|---|---|---:|---:|---:|---|
| `intra_dtarget1_addr010` | intra-cluster | 1 | 0.10 | 0.10 | moderate large budget with known-good cap |
| `intra_dtarget1_addr020` | intra-cluster | 1 | 0.20 | 0.10 | larger budget with known-good cap |
| `intra_dtarget1_addr020_capoff` | intra-cluster | 1 | 0.20 | disabled | test whether per-node cap blocks repair |

Launcher:

- `run_random_two_decoder_intra_target1_large_budget_20260518.sh`

Decision rule: if `target=1` still leaves `c0p_worst_min=0` and many `c0p_need_nodes` with 20% budget, then the endpoint/candidate/cap constraints are the blocker. If target=1 succeeds but Hit@10 drops, we need a softer schedule or a repair-only budget that stops once deficits are filled.

## Target-1 Large-Budget Result

The old target-1 large-budget sweep finished on May 20.

| Variant | Add Ratio | Per-Node Cap | Cora ROC-AUC | Cora AP | Cora Hit@10 | Citeseer ROC-AUC | Citeseer AP | Citeseer Hit@10 | Read |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `addr010_cap010` | 0.10 | 0.10 | 96.00 | 96.30 | 70.78 +/- 7.05 | 96.45 | 96.92 | 75.16 +/- 5.28 | moderate budget helps compactness but still leaves deficits |
| `addr020_cap010` | 0.20 | 0.10 | 96.12 | 96.46 | 72.83 +/- 6.47 | 96.53 | 97.01 | 76.35 +/- 2.53 | best old setting |
| `addr020_capoff` | 0.20 | disabled | 95.97 | 96.31 | 70.59 +/- 4.23 | 96.65 | 97.04 | 75.65 +/- 4.24 | repairs immediate rewrite target but adds too many edges and hurts Hit@10 |

Compactness moved in the intended direction for the best old setting:

- Cora noncompact radius: `0.518598 -> 0.484043`; max noncompact radius: `1.066666 -> 1.009950`
- Citeseer noncompact radius: `0.555253 -> 0.512984`; max noncompact radius: `1.118891 -> 1.073038`

Additional radius and edge-edit diagnostics:

`added_edges_total` is edge-addition pressure summed across the temporary decoded views, not a final persistent graph edge count. Each run logs 60 rewrite views, so `mean added/view` is the easier number to compare.

| Variant | Dataset | Added Edge-Events | Mean Added/View | CP Radius Delta | C0p Radius Delta | Noncompact Radius Delta | Noncompact Max Delta | Compact Radius Loss |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `addr010_cap010` | Cora | 26940.0 | 449.0 | -0.007861 | -0.001131 | -0.025545 | -0.051295 | 0.416473 |
| `addr010_cap010` | Citeseer | 23220.0 | 387.0 | -0.006952 | -0.001392 | -0.022020 | -0.039121 | 0.419951 |
| `addr020_cap010` | Cora | 31840.8 | 530.7 | -0.009883 | -0.001014 | -0.034555 | -0.056716 | 0.408676 |
| `addr020_cap010` | Citeseer | 38728.0 | 645.5 | -0.012529 | -0.001697 | -0.042269 | -0.045853 | 0.429238 |
| `addr020_capoff` | Cora | 53880.0 | 898.0 | -0.007521 | -0.002406 | -0.021083 | -0.013683 | 0.417447 |
| `addr020_capoff` | Citeseer | 46440.0 | 774.0 | -0.007708 | -0.003026 | -0.020013 | -0.044457 | 0.419216 |

Interpretation: `addr020_cap010` gives the best noncompact-radius improvement and the best Hit@10. Removing the per-node cap forces the full 20% add budget every view, but it does not improve noncompact compactness and hurts Cora.

Degree diagnostics show why this is not the final repair setting:

- `addr020_cap010` repaired the immediate `rewrite` target on Cora (`need_after=0`) but still left CP cluster deficits (`cp_need_nodes` mean `75.8` on Cora and `235.6` on Citeseer).
- `addr020_capoff` reached `need_after=0` for the immediate decoded-degree target on both datasets, but still left broad CP cluster deficits because it was repairing `rewrite` nodes, not all CP cluster members.

Latest per-seed cluster-degree snapshots make the same point. In this table, `Need Nodes` means nodes whose intra-cluster degree is still below target `1`; `Bad Clusters` means clusters that still contain at least one such node. `CP` is the full non-noise GMM cluster membership, while `C0p` is the selected rewrite/repair target subset.

| Variant | Dataset | CP Need Nodes | CP Bad Clusters | C0p Need Nodes | C0p Bad Clusters |
|---|---|---:|---:|---:|---:|
| `addr010_cap010` | Cora | 97.0 +/- 21.0 | 12.8 +/- 1.1 | 62.6 +/- 19.7 | 1.4 +/- 0.5 |
| `addr010_cap010` | Citeseer | 324.0 +/- 87.1 | 15.2 +/- 0.8 | 252.8 +/- 90.7 | 2.4 +/- 1.5 |
| `addr020_cap010` | Cora | 75.8 +/- 32.4 | 1.2 +/- 0.4 | 75.8 +/- 32.4 | 1.2 +/- 0.4 |
| `addr020_cap010` | Citeseer | 235.6 +/- 32.4 | 2.6 +/- 0.9 | 235.6 +/- 32.4 | 2.6 +/- 0.9 |
| `addr020_capoff` | Cora | 86.6 +/- 11.5 | 14.4 +/- 0.9 | 0.0 +/- 0.0 | 0.0 +/- 0.0 |
| `addr020_capoff` | Citeseer | 149.2 +/- 20.4 | 15.8 +/- 0.4 | 0.0 +/- 0.0 | 0.0 +/- 0.0 |

Minimum-degree snapshots clarify the mismatch between repaired C0p nodes and unrepaired CP membership. The table below reports the latest snapshot from each seed. For CP/C0p we report intra-cluster degree diagnostics, because the target is intra-cluster degree. `Intra Cluster-Min Avg` is the average, across clusters, of each cluster's minimum intra-degree. The old logs did not record node-level average intra-cluster degree.

| Variant | Dataset | Global Degree Min | Global Degree Avg | C0p Intra Min | C0p Intra Cluster-Min Avg | CP Intra Min | CP Intra Cluster-Min Avg |
|---|---|---:|---:|---:|---:|---:|---:|
| `addr010_cap010` | Cora | 0.0 | 3.65 +/- 0.00 | 0.0 | 0.91 +/- 0.03 | 0.0 | 0.20 +/- 0.07 |
| `addr010_cap010` | Citeseer | 0.0 | 2.56 +/- 0.00 | 0.0 | 0.85 +/- 0.09 | 0.0 | 0.05 +/- 0.05 |
| `addr020_cap010` | Cora | 0.0 | 3.71 +/- 0.00 | 0.0 | 0.93 +/- 0.03 | 0.0 | 0.93 +/- 0.03 |
| `addr020_cap010` | Citeseer | 0.0 | 2.72 +/- 0.00 | 0.0 | 0.84 +/- 0.06 | 0.0 | 0.84 +/- 0.06 |
| `addr020_capoff` | Cora | 0.0 | 3.98 +/- 0.00 | 1.0 | 1.12 +/- 0.08 | 0.0 | 0.23 +/- 0.11 |
| `addr020_capoff` | Citeseer | 0.0 | 2.79 +/- 0.00 | 1.0 | 1.00 +/- 0.00 | 0.0 | 0.01 +/- 0.03 |

So the old target-1 method can raise the selected C0p/rewrite subset to minimum degree 1 when the cap is disabled, but the global graph and full CP membership still contain degree-0 nodes. `addr020_cap010` is better for final quality and compactness, but it still does not guarantee the target.

Conclusion: the old method was the right feasibility baseline. Larger budget with the known-good cap is useful, but it does not guarantee maximum minimum intra-cluster degree over CP membership. Cap-off is not attractive because it adds many more edges and lowers final Hit@10.

Source artifacts:

- `results/random_two_decoder_intra_target1_large_budget_c0p_noncompact_soft025_cap010_intra_dtarget1_addr010_20260518_summary.csv`
- `results/random_two_decoder_intra_target1_large_budget_c0p_noncompact_soft025_cap010_intra_dtarget1_addr020_20260518_summary.csv`
- `results/random_two_decoder_intra_target1_large_budget_c0p_noncompact_soft025_capoff_intra_dtarget1_addr020_20260518_summary.csv`
- `results/intra_target1_large_budget_20260518_driver.log`

## Cluster-Deficit Repair Setting

One important correction: GMM cluster labels should define membership, but graph-isolated members inside a cluster should not be relabeled as noise. They should become repair targets.

The decoded graph editor now has an explicit target-node selector:

- `--decoded_add_degree_target_nodes rewrite`: original behavior; repair only rewrite-mask nodes.
- `--decoded_add_degree_target_nodes cp`: repair all non-noise GMM cluster members, with deficits computed by the selected degree scope.
- `--decoded_add_degree_target_nodes cluster_deficit`: repair only non-noise cluster members currently below the target.
- `--decoded_add_degree_target_nodes rewrite_or_cluster_deficit`: repair rewrite-mask nodes plus any non-noise cluster member below the target.

For the current hypothesis, the clean setting is:

- `--decoded_add_degree_target 1`
- `--decoded_add_degree_target_scope intra_cluster`
- `--decoded_add_degree_target_nodes cp`
- `--decoded_guarantee_degree_target`

This keeps the pulled-latent decoder as the edge scorer, keeps the same-cluster candidate constraint, and first runs a repair pass that can exceed `add_ratio` and the per-node cap until target nodes reach the requested intra-cluster degree whenever valid candidates exist. Any leftover add budget is then spent on normal high-score compactness edges. The new `[DECODED-DEG]` logs include `target_nodes`, `target_mask_nodes`, `guarantee_target`, `repair_added`, `repair_limit`, and `unrepaired` so we can see whether the guarantee pass actually repairs the full CP cluster membership.

Launcher:

- `run_random_two_decoder_cp_target1_repair_20260519.sh`

## CP Guarantee Run Started

On May 20, we cleaned up the runner arguments and started the corrected CP repair experiment with a fresh stamp:

- tmux session: `cp_target1_repair_20260520`
- driver log: `results/cp_target1_repair_20260520_driver.log`
- first prefix: `random_two_decoder_cp_target1_repair_c0p_noncompact_soft025_cap010_cp_dtarget1_guarantee_addr010_20260520`

The child command is now unambiguous:

- `--split_mode random`
- `--decoded_add_degree_target 1`
- `--decoded_add_degree_target_scope intra_cluster`
- `--decoded_add_degree_target_nodes cp`
- `--decoded_guarantee_degree_target`
- `--mlp_pair_max_rows 16`

Early progress when this section was added:

- Cora seed `0`: Hit@10 `0.7154`
- Cora seed `1`: Hit@10 `0.7571`
- Cora seed `2`: running

The main decision signal for this run is not only final Hit@10. The primary structural check is whether `[DECODED-DEG]` reports `guarantee_target=1`, `need_after=0`, and `unrepaired=0` across CP target nodes, plus whether `[DEGREE-CLUSTER]` confirms the CP cluster deficits are actually removed.

## Split-Mode Runner Cleanup

The earlier runner command was confusing because `run_heart_pair_scorer_editor.py` hardcoded:

- `--split_mode heart`

and several random-split launchers appended:

- `--extra_flag=--split_mode`
- `--extra_flag=random`

`argparse` used the last value, so the actual runs were random split, but the command line looked contradictory.

We fixed this by adding an explicit runner option:

- `--split-mode {random,heart}`

The runner now passes only one `--split_mode` to `src/aron_main.py`. It also consumes legacy forwarded `--split_mode` tokens from `--extra_flag`, so older launchers remain compatible while new logs are unambiguous. The random-split validation log message was also relabeled from the misleading `[HeaRT-EVAL] full validation used during training` to:

- `[EVAL] random split full validation/test used during training`

Files updated:

- `run_heart_pair_scorer_editor.py`
- `run_random_two_decoder_cp_target1_repair_20260519.sh`
- `src/aron_train_edit_decoder.py`

## TODO Status

Original TODOs and current status:

| TODO | Status | Answer |
|---|---|---|
| `先比AUC來看效果` | answered | AUROC and AUPRC/AP were compared throughout the ablations. They are mostly stable and less discriminative than Hit@10. We should continue reporting AUROC/AUPRC, but method selection cannot rely on AUC alone. |
| `Maximum minimum node degree in c0p: not guarantee. use old method first` | old method answered; guarantee run active | The old budget-limited method was tested first. `addr020_cap010` is the best old setting, but it does not guarantee CP cluster repair. `addr020_capoff` repairs the immediate rewrite target but hurts Hit@10 and still misses the full CP target. The CP guarantee run is now the direct test. |
| `(fix remove edge)` | answered negatively | Removal was tested earlier. Normal removal is near-inactive, and relaxed/forced removal hurts ROC-AUC/AP/Hit@10. Current best experiments keep `decoded_remove_ratio=0.0`. |
| `(change radius calculation to align with gmm)` | implemented and tested | We added `--compactness_radius_metric=mahalanobis` and CP/C0p/noncompact radius diagnostics. Mahalanobis accounting improved diagnosis but did not rescue the old add+compact recipe. |
| `(when pulling, do we push the non-compact portion too? 不compact的拉遠)` | still open | We tested softer pull strengths and C0p-to-noncompact endpoint selection, but we have not implemented an explicit repulsion/push-away objective for noncompact nodes. Current pull still mainly pulls selected nodes toward cluster centers. |

Open follow-up from the TODO list: design a true noncompact repulsion variant only after the CP guarantee run tells us whether structural repair alone is enough.
