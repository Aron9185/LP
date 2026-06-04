# Post-4/30 Experiment Report

This report summarizes the work done after the 4/30 experiment note.

For the continuation after May 14, use `post_0514_experiment_report.md`. This file remains the historical rollup from the 4/30 diagnosis through the May 14 checkpoint.

Important evaluation note: the 4/30 note focused on HeaRT `samples.npy` alignment, where Hit@10 numbers were in the `20-50` range. Most results in this report are the later random-split setting, so the absolute Hit@10 numbers are not directly comparable to the 4/30 HeaRT-samples table. Within each table below, methods are compared under the same seed/split budget.

## Starting Point After 4/30

The 4/30 diagnosis was:

1. The bilinear edit decoder was not learning the HeaRT ranking surface well enough.
2. Decoder-as-final-scorer was weaker than embedding dot product.
3. Removal had a small signal, but it was not reliable enough.
4. The next method should separate edit scoring from final link prediction.

The planned next steps were:

1. Replace normalized bilinear with a richer pair scorer.
2. Add structural pair features.
3. Train with HeaRT positive-vs-hard-negative ranking.
4. Separate the edit scorer from the final prediction scorer.
5. Run clean ablations for edit/no-edit and dot/pair-scorer behavior.

## Current Method After This Commit

The method evolved into a two-decoder setup:

1. **Encoder**

   VGNAE produces node embeddings `Z`.

2. **Dynamic GMM/C0p targets**

   The current main runner uses dynamic C0p targets with `--dynamic_c0p_targets`.

3. **Edit decoder**

   The default edit decoder is now `pair_mlp_struct`, a structure-aware pair MLP. It uses pair embedding features plus graph/cluster structural context.

4. **Prediction head**

   Final scoring uses a separate `pair_residual_struct` prediction decoder, trained with HeaRT-style ranking and sampled BCE.

5. **Temporary decoded graph view**

   The locked fair-10 default used temporary decoded additions:

   - add-only
   - no removal
   - same-cluster only
   - require C0p endpoint
   - temporary view only, not accumulated into the base graph

6. **Final scoring**

   The final score source is `pred_decoder`, not dot product and not the edit decoder.

## 1. Fair 10-Seed Random-Split Result

We completed a fair random-split comparison on Cora and Citeseer using seeds `0-9` for each method.

| Dataset | Method | Config / Source | ROC-AUC | AP | Hit@10 |
|---|---|---|---:|---:|---:|
| Cora | New two-decoder | `two_decoder_pred` | 95.95 +/- 0.40 | 96.32 +/- 0.45 | **70.49 +/- 5.11** |
| Cora | ReverseGNN | `revgnn_no` | 95.62 +/- 0.22 | 95.94 +/- 0.24 | 68.56 +/- 2.56 |
| Cora | CoEBA | `coeba_v6` | 95.56 +/- 0.18 | 95.85 +/- 0.22 | 67.78 +/- 3.71 |
| Citeseer | New two-decoder | `two_decoder_pred` | 96.16 +/- 0.55 | 96.59 +/- 0.46 | **73.56 +/- 2.73** |
| Citeseer | ReverseGNN | `revgnn_no` | 96.06 +/- 0.29 | 96.48 +/- 0.22 | 70.81 +/- 2.23 |
| Citeseer | CoEBA | `coeba_v6` | 95.55 +/- 0.48 | 96.14 +/- 0.28 | 68.92 +/- 2.86 |

Hit@10 deltas:

| Dataset | vs ReverseGNN | vs CoEBA |
|---|---:|---:|
| Cora | +1.94 | +2.71 |
| Citeseer | +2.75 | +4.64 |

Conclusion: under the equal-round random-split setting, the new method is the best of the compared methods on both datasets.

Source artifacts:

- `results/random_two_decoder_editor_20260501_runs.csv`
- `results/random_old_baselines_seed0_9_runs.csv`
- `results/random_fair10_report.md`

## 2. Scorer And Prediction-Head Effect

The largest clear gain came from the structure-aware prediction head.

Matched seeds where both dot scorer and prediction-head scorer were run:

| Dataset | Seeds | Dot Scorer | Structure-Aware Prediction Head | Gain |
|---|---:|---:|---:|---:|
| Cora | `0-2` | 63.50 +/- 2.90 | 70.71 +/- 3.35 | **+7.21** |
| Citeseer | `0-2` | 70.84 +/- 1.99 | 73.99 +/- 3.84 | **+3.15** |

Conclusion: the `pair_residual_struct` prediction head is the strongest positive change in this commit. The old issue from 4/30, where the bilinear decoder was not a good final scorer, is addressed by separating edit decoding from final link prediction.

## 3. Removal: Normal And Forced

The normal 10-seed remove variant did not beat no-removal:

| Dataset | Variant | ROC-AUC | AP | Hit@10 | Removed Edges |
|---|---|---:|---:|---:|---:|
| Cora | no removal | 95.95 +/- 0.40 | 96.32 +/- 0.45 | 70.49 +/- 5.11 | 0 |
| Cora | removal | 95.92 +/- 0.43 | 96.25 +/- 0.49 | 70.34 +/- 4.63 | 4.1 |
| Citeseer | no removal | 96.16 +/- 0.56 | 96.59 +/- 0.46 | 73.56 +/- 2.73 | 0 |
| Citeseer | removal | 96.17 +/- 0.63 | 96.57 +/- 0.52 | 72.90 +/- 2.64 | 0.0 |

Because normal removal barely removed edges, we ran a capped-removal diagnostic with relaxed constraints.

Setup:

- seeds `0-4`
- `two_decoder_pred_remove`
- add ratio `0.01`
- remove ratio `0.05`
- caps of `10`, `20`, and `50` removals per rewrite round
- `decoded_degree_floor=0`
- cross-cluster allowed
- no C0p endpoint requirement

| Remove Cap | Dataset | ROC-AUC | AP | Hit@10 | Baseline Hit@10 | Delta | Added Edges | Removed Edges | Radius Delta |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | Cora | 95.42 +/- 0.87 | 95.74 +/- 0.75 | 68.43 +/- 5.38 | 69.15 | -0.72 | 2700 | 600 | -0.000049 |
| 10 | Citeseer | 95.22 +/- 0.51 | 95.78 +/- 0.32 | 68.26 +/- 2.75 | 73.85 | -5.58 | 2340 | 600 | -0.000067 |
| 20 | Cora | 95.60 +/- 0.84 | 95.88 +/- 0.80 | 67.86 +/- 6.42 | 69.15 | -1.29 | 2700 | 1200 | -0.000271 |
| 20 | Citeseer | 95.35 +/- 0.73 | 95.77 +/- 0.60 | 67.47 +/- 3.39 | 73.85 | -6.38 | 2340 | 1200 | -0.000243 |
| 50 | Cora | 95.61 +/- 0.90 | 95.77 +/- 1.07 | 67.17 +/- 5.60 | 69.15 | -1.98 | 2700 | 3000 | -0.000889 |
| 50 | Citeseer | 95.12 +/- 0.54 | 95.68 +/- 0.40 | 67.16 +/- 5.31 | 73.85 | -6.69 | 2340 | 3000 | -0.000427 |

Conclusion: removal should not be part of the default method. Normal removal is near-inactive, and relaxed forced removal hurts ROC-AUC/AP/Hit@10.

Source artifact:

- `results/random_two_decoder_capped_remove_diag_report.md`

## 4. 5-Seed Tuning Screen And 10-Seed Confirmation

We screened seven small tuning variants on seeds `0-4`.

| Variant | Setting | Dataset | Hit@10 | Delta |
|---|---|---|---:|---:|
| `bce005` | `prediction_bce_weight=0.05` | Cora | 69.60 +/- 5.54 | +0.46 |
| `bce005` | `prediction_bce_weight=0.05` | Citeseer | 74.20 +/- 3.18 | +0.35 |
| `bce020` | `prediction_bce_weight=0.20` | Cora | 69.64 +/- 6.37 | +0.49 |
| `bce020` | `prediction_bce_weight=0.20` | Citeseer | 73.32 +/- 3.24 | -0.53 |
| `enc000` | `prediction_encoder_weight=0.00` | Cora | 69.53 +/- 7.30 | +0.38 |
| `enc000` | `prediction_encoder_weight=0.00` | Citeseer | 72.79 +/- 1.85 | -1.05 |
| `enc002` | `prediction_encoder_weight=0.02` | Cora | 68.61 +/- 8.28 | -0.53 |
| `enc002` | `prediction_encoder_weight=0.02` | Citeseer | 73.23 +/- 3.13 | -0.62 |
| `enc010` | `prediction_encoder_weight=0.10` | Cora | 71.08 +/- 3.29 | +1.94 |
| `enc010` | `prediction_encoder_weight=0.10` | Citeseer | 73.45 +/- 2.91 | -0.40 |
| `add0005` | `decoded_add_ratio=0.005` | Cora | 68.88 +/- 7.24 | -0.27 |
| `add0005` | `decoded_add_ratio=0.005` | Citeseer | 73.01 +/- 2.97 | -0.84 |
| `add002` | `decoded_add_ratio=0.020` | Cora | 69.37 +/- 5.14 | +0.23 |
| `add002` | `decoded_add_ratio=0.020` | Citeseer | 73.36 +/- 3.36 | -0.48 |

We then confirmed the two promising signals:

- `bce005` to 10 seeds on Cora and Citeseer
- `enc010` to 10 seeds on Cora only

| Dataset | Variant | N | Hit@10 | Delta |
|---|---|---:|---:|---:|
| Cora | current | 10 | 70.49 +/- 5.11 | +0.00 |
| Cora | `bce005` | 10 | 70.17 +/- 4.34 | -0.32 |
| Cora | `enc010` | 10 | 71.14 +/- 2.71 | +0.65 |
| Citeseer | current | 10 | 73.56 +/- 2.73 | +0.00 |
| Citeseer | `bce005` | 10 | 73.69 +/- 3.91 | +0.13 |

Conclusion: do not replace the locked default with these scalar tuning variants. `enc010` is a small Cora-only signal, but it is not a shared replacement.

Source artifacts:

- `results/random_two_decoder_tune5_report.md`
- `results/random_two_decoder_tune_confirm10_report.md`

## 5. Edit Decoder Ablation

We kept the prediction head fixed as `pair_residual_struct` and swapped only the edit decoder.

| Dataset | Edit Decoder | Hit@10 | Delta vs `pair_mlp_struct` |
|---|---|---:|---:|
| Cora | `pair_mlp_struct` | 69.15 +/- 7.07 | +0.00 |
| Cora | `mlp_pair` | 69.22 +/- 7.91 | +0.08 |
| Cora | `bilinear` | 68.80 +/- 9.54 | -0.34 |
| Citeseer | `pair_mlp_struct` | 73.85 +/- 2.74 | +0.00 |
| Citeseer | `mlp_pair` | 72.22 +/- 4.10 | -1.63 |
| Citeseer | `bilinear` | 70.46 +/- 1.89 | -3.38 |

Conclusion: `pair_mlp_struct` remains the best shared default. `mlp_pair` is roughly tied on Cora but worse on Citeseer. `bilinear` is worse overall.

One important diagnostic: `mlp_pair` and `bilinear` can have higher edit-decoder diagnostic Hit@10 while final prediction Hit@10 is worse. Improving the edit decoder's own score is not enough; it must help the final prediction head.

Source artifact:

- `results/random_two_decoder_decoder_ablation_report.md`

## 6. Augmentation And Compactness Ablation

We tested whether decoded additions and compactness are actually needed.

| Dataset | Variant | Add Ratio | Compact W | C0p Targets | ROC-AUC | AP | Hit@10 | Delta vs Current | Added Edges | Radius Delta |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|
| Cora | current | 0.01 | 0.2 | dynamic | 95.93 +/- 0.43 | 96.19 +/- 0.53 | 69.15 +/- 7.07 | +0.00 | 2700 | -0.000051 |
| Cora | `add000` | 0.00 | 0.2 | dynamic | 95.97 +/- 0.52 | 96.20 +/- 0.58 | 69.49 +/- 7.31 | +0.34 | 0 | +0.000000 |
| Cora | `compact000` | 0.01 | 0.0 | dynamic | 96.00 +/- 0.45 | 96.23 +/- 0.50 | 69.30 +/- 6.91 | +0.15 | 2700 | -0.000037 |
| Cora | `add000_compact000` | 0.00 | 0.0 | dynamic | 95.90 +/- 0.52 | 96.20 +/- 0.51 | 70.13 +/- 4.84 | +0.98 | 0 | +0.000000 |
| Cora | `compact020_frozen` | 0.01 | 0.2 | frozen | 95.74 +/- 0.44 | 96.07 +/- 0.51 | 67.59 +/- 7.38 | -1.56 | 2745 | -0.000240 |
| Cora | `compact100_dynamic` | 0.01 | 1.0 | dynamic | 95.98 +/- 0.45 | 96.18 +/- 0.55 | 69.91 +/- 7.69 | +0.76 | 2700 | -0.000030 |
| Cora | `compact100_frozen` | 0.01 | 1.0 | frozen | 95.67 +/- 0.50 | 96.03 +/- 0.52 | 68.27 +/- 7.46 | -0.87 | 2745 | -0.000235 |
| Citeseer | current | 0.01 | 0.2 | dynamic | 96.15 +/- 0.51 | 96.60 +/- 0.40 | 73.85 +/- 2.74 | +0.00 | 2340 | -0.000025 |
| Citeseer | `add000` | 0.00 | 0.2 | dynamic | 96.15 +/- 0.48 | 96.59 +/- 0.40 | 74.20 +/- 2.97 | +0.35 | 0 | +0.000000 |
| Citeseer | `compact000` | 0.01 | 0.0 | dynamic | 96.31 +/- 0.46 | 96.73 +/- 0.32 | 72.92 +/- 3.55 | -0.92 | 2340 | -0.000111 |
| Citeseer | `add000_compact000` | 0.00 | 0.0 | dynamic | 96.33 +/- 0.43 | 96.74 +/- 0.32 | 74.29 +/- 2.99 | +0.44 | 0 | +0.000000 |
| Citeseer | `compact020_frozen` | 0.01 | 0.2 | frozen | 95.72 +/- 0.58 | 96.27 +/- 0.38 | 71.69 +/- 2.18 | -2.16 | 2379 | -0.000061 |
| Citeseer | `compact100_dynamic` | 0.01 | 1.0 | dynamic | 96.22 +/- 0.45 | 96.63 +/- 0.30 | 73.98 +/- 1.66 | +0.13 | 2340 | -0.000078 |
| Citeseer | `compact100_frozen` | 0.01 | 1.0 | frozen | 95.64 +/- 0.52 | 96.18 +/- 0.43 | 70.42 +/- 2.54 | -3.43 | 2379 | -0.000054 |

Conclusion:

- Decoded additions are not necessary on seeds `0-4`; the strongest 5-seed candidate is `add000_compact000`.
- Stronger frozen compactness hurts, especially on Citeseer.
- Radius movement is not aligned with ROC-AUC/AP/Hit@10.

Source artifact:

- `results/random_two_decoder_aug_compact_ablation_report.md`

## 7. Mahalanobis Radius Diagnostic

After adding `--compactness_radius_metric`, we reran the current add+compact setting and the `add000_compact000` candidate with Mahalanobis radius accounting.

Setup:

- seeds `0-4`
- datasets Cora and Citeseer
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

Conclusion: Mahalanobis radius accounting does not rescue the add+compact setting. The no-add/no-compact candidate is still tied or better on ROC-AUC/AP/Hit@10, and the edit-decoder-only diagnostic is much healthier without decoded additions.

This does not replace the recommended 10-seed confirmation, because it only covers seeds `0-4`.

Source artifacts:

- `results/random_two_decoder_mahalanobis_current_20260515b_summary.csv`
- `results/random_two_decoder_mahalanobis_add000_compact000_20260515b_summary.csv`
- `results/mahalanobis_diag_20260515b_driver.log`

## Final Diagnosis For This Commit

1. **The separate structure-aware prediction head is the main win.**

   The matched dot-vs-prediction-head comparison shows large gains, especially on Cora.

2. **The structural edit decoder is useful as the shared default, but not the main source of gain.**

   `pair_mlp_struct` matters for Citeseer and should remain the default, but Cora is not very sensitive between `pair_mlp_struct` and `mlp_pair`.

3. **Removal is harmful.**

   Normal removal is slightly worse, and forced real removal is much worse.

4. **Decoded additions and compactness are questionable.**

   The `add000_compact000` ablation suggests we may not need decoded edge additions or compactness at all.

5. **Compactness is not currently a reliable objective.**

   Stronger/frozen compactness can move radius more, but it does not improve ROC-AUC/AP/Hit@10 and can hurt badly. The Mahalanobis diagnostic also does not make the add+compact setting look better.

6. **The edit-decoder diagnostic is not enough.**

   Higher edit-decoder diagnostic Hit@10 does not guarantee better final prediction Hit@10.

## Current Best Method And Candidate

Locked random-split fair-10 default:

- `two_decoder_pred`
- edit decoder: `pair_mlp_struct`
- prediction head: `pair_residual_struct`
- score source: `pred_decoder`
- no removal
- decoded add ratio `0.01`

New candidate to confirm:

- same as above, but `decoded_add_ratio=0.0` and `compactness_weight=0.0`

This candidate is called `add000_compact000` in the augmentation/compactness ablation.

## Recommended Next Steps

1. Confirm `add000_compact000` to 10 seeds on Cora and Citeseer, checking ROC-AUC/AP/Hit@10 together. The Mahalanobis run was still only seeds `0-4`.
2. If `add000_compact000` holds, simplify the method by disabling decoded graph additions and compactness.
3. Keep removal disabled.
4. Keep `pair_mlp_struct` as the shared edit decoder unless future 10-seed evidence says otherwise.
5. After the 10-seed confirmation, improve augmentation and pulling as separate mechanisms: measure edge proposals, pull targets, and final prediction quality independently.
6. Keep prediction-head training/selection as the other main improvement path, and avoid optimizing stronger compactness or edit-decoder-only diagnostics unless they improve ROC-AUC/AP/Hit@10.
