# Paper-Ready Ablation Tables

日期：2026-06-03

這份文件把主結果以外的 ablation studies 整理成接近論文可直接使用的表格形式。除非特別註明，所有數字都是 ARON random no-leak split、百分比、`mean +/- std`。主讀法以 Hit@10 為主；AUROC/AP 放在需要支持 completeness 的表格中。

建議論文放置方式：

| Table | 建議位置 | 主要 claim |
| --- | --- | --- |
| Table 1 | Main paper | 分離 edit decoder 與 prediction head 是有效架構變更 |
| Table 2 | Main paper | `pair_residual_struct` 是最大單點增益 |
| Table 3 | Main paper 或 appendix | Edit decoder diagnostic 高不等於 final LP 好 |
| Table 4 | Appendix | 直接刪 augmentation/compactness 不穩 |
| Table 5 | Main paper 或 appendix | C0p-to-noncompact endpoint 與 CP target-1 repair 是有效 structural redesign |
| Table 6 | Appendix | Removal 是 negative control，不採用 |
| Table 7 | Appendix | Push/repulsion 是 mixed control，不採用主線 |
| Table 8 | Main paper 或 appendix | Integrated MaskGAE/CIMAGE backbone 不適合目前 editor pipeline |
| Table 9 | Appendix | Scalar tuning 沒有穩定 cross-dataset gain |
| Table 10 | Appendix / qualitative analysis | GMM orphan 診斷支持 target-1 repair 的結構動機 |

## Table 1. Architecture Ablation

Caption draft：Effect of decoupling graph editing and final link prediction on the ARON random no-leak split.

| Method | Seeds | Cora AUROC | Cora AP | Cora Hit@10 | Citeseer AUROC | Citeseer AP | Citeseer Hit@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Old ReverseGNN | 10 | 95.62 +/- 0.22 | 95.94 +/- 0.24 | 68.56 +/- 2.56 | 96.06 +/- 0.29 | 96.48 +/- 0.22 | 70.81 +/- 2.23 |
| CoEBA | 10 | 95.56 +/- 0.18 | 95.85 +/- 0.22 | 67.78 +/- 3.71 | 95.55 +/- 0.48 | 96.14 +/- 0.28 | 68.92 +/- 2.86 |
| Two-decoder, final `pair_residual_struct` scorer | 10 | 95.95 +/- 0.40 | 96.32 +/- 0.45 | 70.49 +/- 5.11 | 96.16 +/- 0.55 | 96.59 +/- 0.46 | 73.56 +/- 2.73 |

Paper takeaway：The two-decoder design improves Hit@10 over the old ReverseGNN by `+1.94` on Cora and `+2.75` on Citeseer, while also outperforming CoEBA.

Source：`results/random_fair10_new_vs_old_baselines.csv`

## Table 2. Final Scorer Ablation

Caption draft：Replacing dot-product scoring with the structure-aware prediction head.

| Dataset | Seeds | Dot scorer Hit@10 | `pair_residual_struct` Hit@10 | Gain |
| --- | ---: | ---: | ---: | ---: |
| Cora | 3 | 63.50 +/- 2.90 | 70.71 +/- 3.35 | +7.21 |
| Citeseer | 3 | 70.84 +/- 1.99 | 73.99 +/- 3.84 | +3.15 |

Paper takeaway：The structure-aware residual scorer is the strongest isolated component gain. This supports keeping `pair_residual_struct` as the final prediction head even if the editor is simplified later.

Source：`post_0430_commit_experiment_report.md`

## Table 3. Edit Decoder Ablation

Caption draft：Ablating the graph-edit decoder while keeping the final prediction head fixed.

| Dataset | Edit decoder | Seeds | AUROC | AP | Hit@10 | Delta vs `pair_mlp_struct` | Edit-decoder diag Hit@10 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Cora | `pair_mlp_struct` | 5 | 95.93 +/- 0.43 | 96.19 +/- 0.53 | 69.15 +/- 7.07 | +0.00 | 30.82 |
| Cora | `mlp_pair` | 5 | 96.16 +/- 0.61 | 96.27 +/- 0.65 | 69.22 +/- 7.91 | +0.08 | 44.74 |
| Cora | `bilinear` | 5 | 95.93 +/- 0.57 | 96.01 +/- 0.66 | 68.80 +/- 9.54 | -0.34 | 54.91 |
| Citeseer | `pair_mlp_struct` | 5 | 96.15 +/- 0.51 | 96.60 +/- 0.40 | 73.85 +/- 2.74 | +0.00 | 15.47 |
| Citeseer | `mlp_pair` | 5 | 96.26 +/- 0.36 | 96.63 +/- 0.38 | 72.22 +/- 4.10 | -1.63 | 55.16 |
| Citeseer | `bilinear` | 5 | 95.98 +/- 0.27 | 96.40 +/- 0.19 | 70.46 +/- 1.89 | -3.38 | 55.96 |

Paper takeaway：A high edit-decoder diagnostic score does not guarantee better final link prediction. `pair_mlp_struct` is the safest shared edit decoder because it avoids the Citeseer degradation seen with `mlp_pair` and `bilinear`.

Source：`results/random_two_decoder_decoder_ablation_report.md`

## Table 4. Augmentation And Compactness Ablation

Caption draft：Testing whether decoded additions and compactness can be removed.

| Dataset | Variant | Seeds | AUROC | AP | Hit@10 | Delta |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Cora | Current add+compact | 10 | 95.95 +/- 0.40 | 96.32 +/- 0.45 | 70.49 +/- 5.11 | +0.00 |
| Cora | No additions, no compactness | 10 | 95.96 +/- 0.37 | 96.30 +/- 0.41 | 69.92 +/- 5.11 | -0.57 |
| Citeseer | Current add+compact | 10 | 96.16 +/- 0.55 | 96.59 +/- 0.46 | 73.56 +/- 2.73 | +0.00 |
| Citeseer | No additions, no compactness | 10 | 96.13 +/- 0.79 | 96.54 +/- 0.62 | 72.18 +/- 3.56 | -1.38 |

Paper takeaway：A 5-seed screen suggested removing additions/compactness might help, but the 10-seed confirmation did not hold. The right path is to redesign the augmentation, not delete it outright.

Source：`results/random_two_decoder_add000_compact000_confirm10_20260515_summary.csv`, `results/random_fair10_new_vs_old_baselines.csv`

## Table 5. Structural Augmentation Redesign

Caption draft：Endpoint selection and CP intra-cluster target-1 repair.

| Variant | Structural change | Seeds | Cora Hit@10 | Citeseer Hit@10 | Read |
| --- | --- | ---: | ---: | ---: | --- |
| Older two-decoder default | broad decoded additions | 10 | 70.49 +/- 5.11 | 73.56 +/- 2.73 | stable baseline |
| `current_pull100` | same-cluster + C0p endpoint, pull 1.00 | 5 | 69.60 +/- 6.38 | 73.41 +/- 2.01 | local endpoint baseline |
| `softpull025` | softer pull only | 5 | 68.39 +/- 7.73 | 71.74 +/- 2.31 | pull strength alone is worse |
| `c0p_noncompact_soft025_cap010` | C0p-to-noncompact endpoint + per-node cap | 5 | 70.06 +/- 7.19 | 75.08 +/- 2.31 | better endpoint rule |
| old target-1 `addr020_cap010` | intra-cluster target-1 budget, capped | 5 | 72.83 +/- 6.47 | 76.35 +/- 2.53 | strong metric screen, but still left CP deficits |
| CP target-1 guarantee `addr020` | repair all CP intra-cluster degree deficits to target 1 | 5 | 71.92 +/- 5.61 | 76.79 +/- 2.18 | current structural repair default |

Paper takeaway：Endpoint selection matters more than simply changing pull strength. The corrected CP target-1 guarantee is the cleanest structural version because it repairs the intended CP membership, even though the earlier target-1 screen had a slightly higher Cora mean.

Source：`post_0514_experiment_report.md`, `results/random_two_decoder_cp_target1_repair_c0p_noncompact_soft025_cap010_cp_dtarget1_guarantee_addr020_20260520_summary.csv`

## Table 6. Removal Ablation

Caption draft：Edge removal as a negative control.

| Setting | Seeds | Cora Hit@10 | Citeseer Hit@10 | Removed edges | Read |
| --- | ---: | ---: | ---: | ---: | --- |
| No removal | 10 | 70.49 +/- 5.11 | 73.56 +/- 2.73 | 0 | baseline |
| Normal removal | 10 | 70.34 +/- 4.63 | 72.90 +/- 2.64 | near-inactive | small drop |
| Forced removal, cap 10 | 5 | 68.43 +/- 5.38 | 68.26 +/- 2.75 | 600 | harmful |
| Forced removal, cap 20 | 5 | 67.86 +/- 6.42 | 67.47 +/- 3.39 | 1200 | harmful |
| Forced removal, cap 50 | 5 | 67.17 +/- 5.60 | 67.16 +/- 5.31 | 3000 | harmful |

Paper takeaway：Removal is not merely inactive. When forced to remove edges, it consistently damages Hit@10, especially on Citeseer. Current method should remain add-only.

Source：`post_0430_commit_experiment_report.md`, `results/random_two_decoder_capped_remove_diag_report.md`

## Table 7. Push / Repulsion Ablation

Caption draft：Full-epoch noncompact/noise push under the current structural repair setting.

| Variant | Push strength | Seeds | Cora AUROC | Cora AP | Cora Hit@10 | Citeseer AUROC | Citeseer AP | Citeseer Hit@10 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `baseline_cp_pull` | none | 3 | 95.96 +/- 0.44 | 96.53 +/- 0.24 | 74.45 +/- 2.88 | 96.49 +/- 0.22 | 97.00 +/- 0.16 | 76.70 +/- 2.16 |
| `c0p_pull_only` | none | 3 | 95.88 +/- 0.49 | 96.47 +/- 0.27 | 74.38 +/- 1.19 | 96.04 +/- 0.43 | 96.61 +/- 0.36 | 74.80 +/- 0.83 |
| `c0p_pull_push_weak` | noncompact 0.05, noise 0.02 | 3 | 96.01 +/- 0.76 | 96.58 +/- 0.58 | 74.64 +/- 3.51 | 96.25 +/- 0.32 | 96.77 +/- 0.39 | 75.82 +/- 5.57 |
| `c0p_pull_push_mid` | noncompact 0.10, noise 0.05 | 3 | 96.00 +/- 0.54 | 96.56 +/- 0.34 | 73.43 +/- 2.95 | 96.47 +/- 0.12 | 96.98 +/- 0.13 | 76.92 +/- 1.72 |

Paper takeaway：Push is active but mixed. Weak push slightly improves Cora but drops Citeseer relative to the baseline; mid push recovers Citeseer but hurts Cora. It should stay as an ablation, not the main method.

Source：`results/random_two_decoder_pull_push_*_20260531_summary.csv`

## Table 8. Integrated Backbone Ablation

Caption draft：Replacing the entire ARON autoencoder backbone inside the editor pipeline.

| Backbone inside ARON editor pipeline | Seeds | Cora AUROC | Cora AP | Cora Hit@10 | Citeseer AUROC | Citeseer AP | Citeseer Hit@10 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| VGNAE + CP target-1 repair | 5 | 95.99 +/- 0.51 | 96.40 +/- 0.46 | 71.92 +/- 5.61 | 96.43 +/- 0.64 | 96.96 +/- 0.43 | 76.79 +/- 2.18 |
| Integrated MaskGAE backbone | 5 | 94.46 +/- 1.14 | 95.41 +/- 1.02 | 67.59 +/- 2.60 | 94.80 +/- 0.27 | 95.72 +/- 0.38 | 71.52 +/- 3.62 |
| Integrated CIMAGE-full backbone | 3 | 91.91 +/- 0.39 | 92.79 +/- 0.96 | 60.22 +/- 8.68 | 89.54 +/- 1.03 | 92.32 +/- 0.84 | 63.22 +/- 3.74 |

Paper takeaway：The MaskGAE/CIMAGE replacement is not an editor-only backbone swap. It changes the embeddings used by reconstruction/contrastive learning, GMM CP/C0p, the edit decoder, and the prediction head. Standalone MaskGAE is strong, but the integrated MaskGAE/CIMAGE objectives are not aligned with our current GMM-guided editor pipeline.

Note：CIMAGE Cora uses the rerun-adjusted seed set from `post_0521_experiment_report.md`: seed `0` rerun plus original seeds `1-2`.

Source：`post_0521_experiment_report.md`, `results/random_two_decoder_maskgae_backbone_*_summary.csv`, `results/random_two_decoder_cimage_full_backbone_*_summary.csv`

## Table 9. Scalar Tuning Ablation

Caption draft：Tuning prediction-head loss weights.

| Dataset | Variant | Setting | Seeds | AUROC | AP | Hit@10 | Delta | Paired wins/losses |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Cora | current | default | 10 | 95.95 +/- 0.40 | 96.32 +/- 0.45 | 70.49 +/- 5.11 | +0.00 | -- |
| Cora | `bce005` | `prediction_bce_weight=0.05` | 10 | 95.97 +/- 0.39 | 96.33 +/- 0.41 | 70.17 +/- 4.34 | -0.32 | 5 / 5 |
| Cora | `enc010` | `prediction_encoder_weight=0.10` | 10 | 95.94 +/- 0.38 | 96.34 +/- 0.38 | 71.14 +/- 2.71 | +0.65 | 4 / 6 |
| Citeseer | current | default | 10 | 96.16 +/- 0.55 | 96.59 +/- 0.46 | 73.56 +/- 2.73 | +0.00 | -- |
| Citeseer | `bce005` | `prediction_bce_weight=0.05` | 10 | 96.16 +/- 0.66 | 96.58 +/- 0.56 | 73.69 +/- 3.91 | +0.13 | 7 / 3 |
| Citeseer | `enc010` | `prediction_encoder_weight=0.10` | 5 | 96.18 +/- 0.49 | 96.67 +/- 0.35 | 73.45 +/- 2.91 | -0.40 | 1 / 4 |

Paper takeaway：`bce005` is not a shared upgrade, and `enc010` is a Cora-only signal with weak paired evidence. Keep default scalar weights.

Source：`results/random_two_decoder_tune_confirm10_report.md`

## Table 10. GMM Orphan And Repair Correctness

Caption draft：Cluster-orphan diagnosis and reconstructed target-1 repair.

| Dataset | Runs | Global degree-0 | CP intra-orphan | C0p intra-orphan | CP intra avg | C0p intra avg |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Cora | 5 | 0.00% | 8.76% | 5.93% | 2.91 | 3.22 |
| Citeseer | 5 | 1.44% | 10.85% | 8.00% | 2.28 | 2.51 |

Repair visualization, seed 0：

| Dataset | Original CP orphan | Repaired CP orphan | Original C0p orphan | Repaired C0p orphan | Added edges | Removed edges |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Cora | 10.20% | 0.00% | 7.12% | 0.00% | 645 | 0 |
| Citeseer | 10.59% | 0.00% | 7.62% | 0.00% | 799 | 0 |

Paper takeaway：The orphan signal is not mostly global isolation. It is a mismatch between GMM embedding clusters and original graph neighborhoods. The reconstructed target-1 repair removes same-cluster intra-orphans by adding same-cluster edges only, which supports the structural motivation for CP target-1 repair.

Important caveat：The repair visualization uses reconstructed target-1 repair views because the historical temporary edited adjacency was not saved. The training logs nevertheless show the CP target-1 guarantee pass reached `need_after=0`, `unrepaired=0`, and `cluster_bad_after=0` across logged repair checks.

Source：`post_0521_experiment_report.md`, `results/gmm_orphan_diagnostics/`

## Recommended Ablation Narrative

1. The largest isolated improvement is the final `pair_residual_struct` prediction head.
2. The edit decoder still matters, but its own diagnostic score is not the target; final prediction quality is.
3. Simple removal of augmentation/compactness does not survive 10-seed confirmation.
4. The productive redesign is structural: C0p-to-noncompact endpoint selection plus CP target-1 intra-cluster repair.
5. Removal is a negative control and should stay disabled.
6. Push is active but mixed, so it remains diagnostic.
7. Integrated MaskGAE/CIMAGE backbones underperform VGNAE in this editor pipeline, even though standalone MaskGAE is strong.
8. Current limitations for the ablation story: CP target-1 repair random split is still 5 seeds, VGNAE-only / no-editor baseline is still pending, and edited-graph snapshots were not saved for post-hoc orphan recomputation.
