# ReverseGNN Research Organization Draft

Date: 2026-06-01

This is a Notion-ready organization draft for the ReverseGNN / ARON research thread. The Notion links were not readable from this environment, so the content below is grounded in the local reports, result CSVs, launch scripts, and git history under `ARON/experiments/reversegnn-compactness` plus `official_baselines`.

## One-Line Story

We started with compactness-driven ReverseGNN graph editing, found that radius movement and decoder-only diagnostics were not enough, moved to a two-decoder design with a structure-aware prediction head, then refined augmentation into C0p-to-noncompact CP target-1 repair, and finally separated fair no-leak claims from leaky/full-graph protocol references against CIMAGE and MaskGAE.

## Current Main Claim

Main protocol: ARON random edge split with held-out validation/test positive edges removed from training, cached negatives, and no full-graph leakage.

Under this fair ARON no-leak setting:

| Method / Setting | Dataset | Seeds | AUROC | AP | Hit@10 | Read |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| ReverseGNN, VGNAE + CP target-1 repair | Cora | 5 | 95.99 | 96.40 | 71.92 +/- 5.61 | Current main baseline |
| ReverseGNN, VGNAE + CP target-1 repair | Citeseer | 5 | 96.43 | 96.96 | 76.79 +/- 2.18 | Current main baseline |
| CIMAGE authors code, ARON no-leak wrapper | Cora | 3 | 86.42 | 88.52 | 47.00 +/- 2.86 | Much weaker under no-leak |
| CIMAGE authors code, ARON no-leak wrapper | Citeseer | 3 | 89.94 | 92.10 | 61.98 +/- 3.96 | Much weaker under no-leak |
| MaskGAE Edge authors code, ARON no-leak wrapper | Cora | 3 | 96.83 | 97.10 | 75.71 +/- 4.68 | Slightly above ReverseGNN in this 3-seed slice |
| MaskGAE Edge authors code, ARON no-leak wrapper | Citeseer | 3 | 96.87 | 97.33 | 78.24 +/- 0.95 | Slightly above ReverseGNN in this 3-seed slice |

Protocol warning: full-graph training makes ReverseGNN and MaskGAE nearly saturated on the held-out ARON edges. Those numbers are useful for diagnosing leakage/protocol sensitivity, but should not be mixed into the main leaderboard.

Primary source:

- `post_0531_protocol_comparison_report.md`
- `results/random_two_decoder_cp_target1_repair_c0p_noncompact_soft025_cap010_cp_dtarget1_guarantee_addr020_20260520_summary.csv`
- `/home/retro/official_baselines/runs/20260528_aron_split/summary.md`

## Current Method Snapshot

Recommended current method family:

- Encoder/backbone: VGNAE, still best aligned with the editor.
- Cluster/core logic: dynamic GMM labels, CP/C0p target masks.
- Edit decoder: `pair_mlp_struct`.
- Prediction head: `pair_residual_struct`.
- Final score source: `pred_decoder`, not dot product and not the edit decoder.
- Augmentation: add-only temporary decoded graph view.
- Endpoint policy: same-cluster C0p-to-noncompact CP.
- Degree repair: CP intra-cluster target-1 guarantee.
- Add budget: large enough for repair, with per-node cap used by the best current baseline.
- Removal: disabled.

## Chronological Timeline

### 2026-03-26 and 2026-04-02: Compactness / ReverseGNN Foundation

Role in the story:

- Established the ReverseGNN compactness/editing question.
- Explored frozen vs dynamic cluster targets, radius objectives, add ratios, pull strengths, and basic bilinear/MLP decoder variants.
- Found early evidence that moving radius alone does not guarantee better link-prediction Hit@10.

What we learned:

- Frozen baselines stayed surprisingly strong on Hit@10.
- Dynamic compactness could shrink radii but did not reliably improve ranking.
- Add/remove/pull sweeps needed to be judged on final AUROC/AP/Hit@10, not radius alone.

Useful artifacts:

- `results/research_matrix_summary.csv`
- `results/stageA_corrected_summary.csv`
- `results/stageB_results.csv`
- `results/stage2_pull_sweep_results.csv`
- `results/stage3a_addratio_sweep.csv`

### 2026-04-16: Research Matrix and Deep Dynamic Comparison

Role in the story:

- Compared old frozen/dynamic baselines against a new dynamic bilinear hybrid method with a HeaRT-like ranking signal.

What we learned:

- The new dynamic method improved over prior dynamic baselines on Cora and Citeseer.
- It still did not beat the strongest frozen baseline on Hit@10.
- The HeaRT-like decoder objective was mechanically active, but the method still needed a better scoring/training interface.

Useful artifact:

- `results/heart_compare_deep_report.md`

### 2026-04-23 to 2026-04-29: HeaRT-Aligned Measurement

Role in the story:

- Aligned experiments with HeaRT `samples.npy`, full validation, Hit@10 checkpointing, and baselines such as BUDDY and NCNC.

What we learned:

- Best ReverseGNN HeaRT-samples Hit@10 stayed well below BUDDY/NCNC.
- Ratio removal had only a small signal.
- Edit-start sweeps, harder decoder ranking, and threshold removal did not close the gap.
- The next architecture needed to separate edit scoring from final link prediction.

Useful artifact:

- `heart_aligned_result_note.md`

### 2026-04-30: Structural Decoder and Two-Decoder Turn

Role in the story:

- Architectural pivot from a single bilinear/dot scoring path toward a two-decoder setup.

What changed:

- Added a richer pair scorer and structural pair features.
- Separated edit scoring from final prediction.
- Introduced `pair_mlp_struct` for edit decoding.
- Introduced `pair_residual_struct` as the prediction head.
- Made `pred_decoder` the final score source.

What we learned:

- The separate structure-aware prediction head became the main gain.
- Decoder-as-final-scorer and pure dot scoring were weaker.

Useful artifacts:

- `post_0430_commit_experiment_report.md`
- `results/random_fair10_report.md`

### 2026-05-14: Random-Split Fair 10-Seed Result and Ablations

Role in the story:

- Locked the first fair random-split 10-seed win against old ReverseGNN and CoEBA.

Main result:

| Dataset | New two-decoder Hit@10 | Old ReverseGNN Hit@10 | CoEBA Hit@10 |
| --- | ---: | ---: | ---: |
| Cora | 70.49 +/- 5.11 | 68.56 +/- 2.56 | 67.78 +/- 3.71 |
| Citeseer | 73.56 +/- 2.73 | 70.81 +/- 2.23 | 68.92 +/- 2.86 |

Decision outcomes:

- Keep `pair_residual_struct` prediction head.
- Keep `pair_mlp_struct` as the shared edit decoder.
- Disable removal by default.
- Do not promote scalar tuning variants like `bce005` or Cora-only `enc010`.
- Treat decoded additions and compactness as questionable, but not yet removable.

Useful artifacts:

- `post_0430_commit_experiment_report.md`
- `results/random_fair10_report.md`
- `results/random_two_decoder_capped_remove_diag_report.md`
- `results/random_two_decoder_decoder_ablation_report.md`
- `results/random_two_decoder_tune_confirm10_report.md`
- `results/random_two_decoder_aug_compact_ablation_report.md`

### 2026-05-14 to 2026-05-21: No-Add Control, Endpoint Redesign, and Degree Repair

Role in the story:

- Tested whether augmentation/compactness could simply be deleted.
- When the 10-seed no-add/no-compact confirmation failed, redesigned augmentation instead of dropping it.

What we learned:

- `add000_compact000` looked good at 5 seeds but did not hold at 10 seeds.
- The old decoded additions were too broad.
- A C0p-to-noncompact CP endpoint rule plus per-node cap was the first balanced improvement.
- CP target-1 intra-cluster repair became the best current augmentation mechanism.

Key current result:

| Variant | Dataset | AUROC | AP | Hit@10 | Read |
| --- | --- | ---: | ---: | ---: | --- |
| CP target-1 repair, capped | Cora | 95.99 | 96.40 | 71.92 +/- 5.61 | Current main baseline |
| CP target-1 repair, capped | Citeseer | 96.43 | 96.96 | 76.79 +/- 2.18 | Current main baseline |

Useful artifacts:

- `post_0514_experiment_report.md`
- `post_0521_experiment_report.md`
- `results/random_two_decoder_add000_compact000_confirm10_20260515_summary.csv`
- `results/random_two_decoder_cp_target1_repair_c0p_noncompact_soft025_cap010_cp_dtarget1_guarantee_addr020_20260520_summary.csv`

### 2026-05-21 to 2026-05-25: GMM and Cluster-Orphan Correctness

Role in the story:

- Validated whether the GMM labels, CP/C0p masks, and orphan diagnostics meant what we thought.

What we learned:

- `gmm_labels` passed synthetic correctness checks.
- Graph isolation does not relabel GMM noise, because GMM uses embeddings only.
- Original-graph CP intra-orphans were real but plausible for sparse graphs:
  - Cora CP intra-orphan: 8.76%
  - Citeseer CP intra-orphan: 10.85%
  - Cora global degree-0: 0.00%
  - Citeseer global degree-0: 1.44%
- Tail clusters still matter: some clusters reached roughly 28-31% intra-orphans.
- Reconstructed target-1 repair eliminated CP/C0p intra-orphans in the seed-0 comparison using same-cluster additions.
- Repair logs showed `need_after=0`, `unrepaired=0`, and `cluster_bad_after=0` for all logged repair checks in the cap-off run.

Useful artifacts:

- `post_0521_experiment_report.md`
- `diagnose_gmm_cluster_orphans.py`
- `results/gmm_orphan_diagnostics/`

### 2026-05-27 to 2026-05-28: Integrated Backbone Replacement

Role in the story:

- Tested whether replacing VGNAE inside our editor with MaskGAE or CIMAGE-style masked autoencoding improves the same graph-editing recipe.

What we learned:

| Backbone inside our editor | Dataset | Seeds | AUROC | AP | Hit@10 | Read |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| VGNAE CP target-1 repair | Cora | 5 | 95.99 | 96.40 | 71.92 +/- 5.61 | Best aligned with editor |
| VGNAE CP target-1 repair | Citeseer | 5 | 96.43 | 96.96 | 76.79 +/- 2.18 | Best aligned with editor |
| MaskGAE backbone | Cora | 5 | 94.46 | 95.41 | 67.59 +/- 2.60 | Underperforms VGNAE |
| MaskGAE backbone | Citeseer | 5 | 94.80 | 95.72 | 71.52 +/- 3.62 | Underperforms VGNAE |
| CIMAGE-full backbone | Cora | 3 | 91.91 | 92.79 | 60.22 +/- 8.68 | Clear underperformance |
| CIMAGE-full backbone | Citeseer | 3 | 89.54 | 92.32 | 63.22 +/- 3.74 | Clear underperformance |

Decision:

- Keep VGNAE as the main backbone.
- Treat MaskGAE and CIMAGE integration as ablations unless a focused integration study is needed.

Useful artifacts:

- `post_0521_experiment_report.md`
- `results/random_two_decoder_maskgae_backbone_*_summary.csv`
- `results/random_two_decoder_cimage_full_backbone_*_summary.csv`

### 2026-05-28 to 2026-06-01: Official Baselines and Protocol Separation

Role in the story:

- Compared official CIMAGE and MaskGAE code against ReverseGNN under multiple protocols.

Protocol decision:

- Main claim must use ARON no-leak.
- Full-graph ARON and CIMAGE public-code/default are diagnostics/references, not fair leaderboard rows.

Key reads:

- CIMAGE drops sharply under ARON no-leak.
- Official MaskGAE Edge is strong under ARON no-leak.
- Full-graph visibility makes ReverseGNN and MaskGAE nearly saturated.
- CIMAGE public-code/default numbers help explain paper/public-code comparability but are leaky for our edge-prediction claim.

Useful artifacts:

- `post_0531_protocol_comparison_report.md`
- `/home/retro/official_baselines/runs/20260528_aron_split/summary.md`
- `/home/retro/official_baselines/runs/20260529_aron_split_fullgraph_leakage/`
- `/home/retro/official_baselines/runs/20260531_cimage_paper_multiseed_fixed/`

## Decision Register

### Keep as Mainline

| Decision | Why |
| --- | --- |
| Use ARON no-leak as the main protocol | Prevents held-out positive edges from being visible to training |
| Keep VGNAE in the editor pipeline | MaskGAE/CIMAGE integrated backbones underperformed despite standalone MaskGAE strength |
| Keep separate prediction head | Structure-aware prediction head gave the biggest gain over dot scoring |
| Keep `pair_mlp_struct` edit decoder | Best shared edit decoder across Cora/Citeseer |
| Keep removal disabled | Normal removal was inactive/slightly worse; forced removal clearly hurt |
| Keep CP target-1 repair as current augmentation | Repairs intra-cluster deficits and improved the current baseline |

### Rejected or Deprioritized

| Idea | Reason |
| --- | --- |
| Decoder-as-final-scorer | Weaker than separate prediction head |
| Dot-only final scoring | Lost to `pred_decoder` |
| Bilinear edit decoder as default | Worse overall, especially Citeseer |
| Stronger/frozen compactness | Radius movement did not translate to final metrics |
| Simple no-add/no-compact default | 5-seed signal did not hold at 10 seeds |
| Forced/removal-heavy editing | Harmed AUROC/AP/Hit@10 |
| Blind MaskGAE/CIMAGE backbone replacement | Integrated versions underperformed VGNAE |
| Full-graph protocol as main leaderboard | Leaks held-out positives and saturates results |

### Tentative / Needs More Evidence

| Thread | Status |
| --- | --- |
| Pull-mask scope: CP vs C0p | Early 2026-05-31 3-seed grid suggests CP pull remains safer than C0p pull-only, especially on Citeseer |
| Noncompact/noise push | Current available rows are smoke/forced-rewrite checks, not mature full runs |
| CIMAGE cluster/factor ablations | Useful only if we want to explain integration failure, not needed for the main VGNAE path |
| Edited-graph orphan snapshots | Still needed for artifact-level after-edit orphan diagnostics |

## Suggested Notion Structure

Create one master page with the following sections:

1. Executive Status
   - Current main claim
   - Current method snapshot
   - What is fair vs diagnostic

2. Timeline
   - 2026-03-26 and 2026-04-02: Compactness foundation
   - 2026-04-16: Research matrix
   - 2026-04-23/29: HeaRT alignment
   - Structural Decoder
   - 2026-05-14: Random split and ablations
   - 2026-05-21: Endpoint/degree repair
   - GMM and cluster orphan correctness
   - 2026-05-28: Official baselines
   - 2026-06-01: Backbone/protocol comparison

3. Method State
   - Current default
   - Main alternatives tried
   - Why each alternative was accepted/rejected

4. Protocol Ledger
   - ARON no-leak
   - ARON split with full-graph training
   - CIMAGE public-code/default
   - HeaRT samples
   - Random split legacy comparisons

5. Evidence Index
   - Report path
   - CSV path
   - Datasets/seeds
   - Status: accepted, rejected, diagnostic, tentative

6. Next Experiments
   - VGNAE endpoint/repulsion work
   - Save edited graph snapshots
   - Node-level intra-degree logging
   - Optional MaskGAE integration ablation

## Page Mapping for the Provided Links

| Notion page title | Best role in master organization |
| --- | --- |
| Experiments 3-26 | Early compactness/GMM scaffold and initial dynamic-vs-frozen questions |
| Experiments 4-2 | Pull/add/radius sweeps and early compactness diagnostics |
| Experiments 4-16 | Research matrix and dynamic method comparison |
| Experiments 4-23 | HeaRT-aligned measurement and baseline gap |
| Structural Decoder | Two-decoder architecture pivot |
| Experiments 5-14 | Random-split fair 10-seed win and ablations |
| Experiments 5-21 | Endpoint redesign, target-1 repair, next correctness plan |
| GMM And Cluster Orphan Correctness | GMM/orphan validation and repair-mechanism diagnostics |
| Experiments 5-28 | Official CIMAGE/MaskGAE baselines and no-leak comparisons |
| Backbone Comparison 6-1 | Protocol/backbone synthesis and final claim framing |

## Next Work

Most useful next steps:

1. Keep the main paper claim centered on ARON no-leak results.
2. Extend/confirm the current VGNAE CP target-1 baseline where needed for matched seed counts.
3. Finish a real full-epoch noncompact/noise push experiment if repulsion is still a target hypothesis.
4. Save edited adjacency snapshots for after-edit orphan diagnostics, not only reconstructed views/log counters.
5. Add normal training logs for node-level `cp_intra_mean_degree`, `c0p_intra_mean_degree`, and relevant percentiles.
6. Only run deeper MaskGAE/CIMAGE integration studies if the paper needs an explanation of why standalone MaskGAE is strong but integrated MaskGAE is not.

