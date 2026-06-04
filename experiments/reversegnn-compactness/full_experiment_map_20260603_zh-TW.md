# ReverseGNN Full Experiment Map

日期：2026-06-03

這份文件是目前 ReverseGNN / ARON compactness research 的總入口。它把結果依 protocol 分成：

- ARON random no-leak split
- HeaRT samples split
- Official baseline / protocol diagnostics
- Component ablation studies

數字除非特別註明，皆為 Hit@10 百分比 `mean +/- std`。

## 目前主方法

目前採用的完整方法：

| Component | Current choice |
| --- | --- |
| Encoder backbone | VGNAE |
| Architecture | two decoder |
| Edit decoder | `pair_mlp_struct` |
| Prediction head | `pair_residual_struct` |
| Final score source | `pred_decoder` |
| Targeting | dynamic GMM CP/C0p |
| Graph augmentation | temporary decoded graph view |
| Endpoint rule | C0p-to-noncompact CP |
| Degree repair | CP intra-cluster target-1 guarantee |
| Removal | disabled |
| Push / repulsion | disabled in main |
| Main protocol | ARON random no-leak; HeaRT matched run in progress |

## Protocol 1：ARON Random No-Leak Split

這是目前 random-seed 主結果使用的主要 protocol。不要和 full-graph leakage 或 CIMAGE public-code/default protocol 混在一起比較。

### Main Results

All numbers are reported as percentage points. CIMAGE / MaskGAE are official-code baselines evaluated on the same ARON random no-leak split, so they are included in the main comparison table.

| Method / Config | Seeds | Cora AUROC | Cora AP | Cora Hit@10 | Citeseer AUROC | Citeseer AP | Citeseer Hit@10 | Read |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Old ReverseGNN | 10 | 95.62 +/- 0.22 | 95.94 +/- 0.24 | 68.56 +/- 2.56 | 96.06 +/- 0.29 | 96.48 +/- 0.22 | 70.81 +/- 2.23 | old baseline |
| CoEBA | 10 | 95.56 +/- 0.18 | 95.85 +/- 0.22 | 67.78 +/- 3.71 | 95.55 +/- 0.48 | 96.14 +/- 0.28 | 68.92 +/- 2.86 | old baseline |
| Older two-decoder default | 10 | 95.95 +/- 0.40 | 96.32 +/- 0.45 | 70.49 +/- 5.11 | 96.16 +/- 0.55 | 96.59 +/- 0.46 | 73.56 +/- 2.73 | first stable two-decoder result |
| CP target-1 repair `addr020` | 5 | 95.99 +/- 0.51 | 96.40 +/- 0.46 | 71.92 +/- 5.61 | 96.43 +/- 0.64 | 96.96 +/- 0.43 | 76.79 +/- 2.18 | current full-method random baseline |
| Latest pull/push baseline `baseline_cp_pull` | 3 | 95.96 +/- 0.44 | 96.53 +/- 0.24 | 74.45 +/- 2.88 | 96.49 +/- 0.22 | 97.00 +/- 0.16 | 76.70 +/- 2.16 | latest local baseline for push grid |
| Official CIMAGE | 3 | 86.42 +/- 0.27 | 88.52 +/- 0.24 | 47.00 +/- 2.86 | 89.94 +/- 1.41 | 92.10 +/- 1.05 | 61.98 +/- 3.96 | official baseline, weak under no-leak |
| Official MaskGAE Path | 3 | 96.80 +/- 0.15 | 96.95 +/- 0.24 | 72.87 +/- 4.30 | 97.14 +/- 0.12 | 97.51 +/- 0.10 | 76.92 +/- 2.25 | strong official baseline |
| Official MaskGAE Edge | 3 | 96.83 +/- 0.14 | 97.10 +/- 0.18 | 75.71 +/- 4.68 | 96.87 +/- 0.15 | 97.33 +/- 0.09 | 78.24 +/- 0.95 | strongest official no-leak baseline |

Source：

- `results/random_fair10_new_vs_old_baselines.csv`
- `results/random_two_decoder_cp_target1_repair_c0p_noncompact_soft025_cap010_cp_dtarget1_guarantee_addr020_20260520_summary.csv`
- `results/random_two_decoder_pull_push_baseline_cp_pull_20260531_summary.csv`
- `/home/retro/official_baselines/runs/20260528_aron_split/summary.md`

## Protocol 2：HeaRT Samples Split

這是對齊 HeaRT `samples.npy` 的 split。早期方法在這裡明顯低於 BUDDY / NCNC；目前 full method 已有大幅改善。

### Combined Results

All numbers are reported as percentage points. HeaRT comparison should still be read primarily by Hit@10; AUROC/AP are included when the local artifact records them. BUDDY / NCNC local artifacts currently report Hits@K only, so their AUROC/AP cells are left as `--`.

| Method / Config | Type | Seeds | Cora AUROC | Cora AP | Cora Hit@10 | Citeseer AUROC | Citeseer AP | Citeseer Hit@10 | Read |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| BUDDY | external baseline | 10 | -- | -- | 30.34 +/- 1.02 | -- | -- | 48.61 +/- 1.32 | HeaRT baseline |
| NCNC | external baseline | 10 | -- | -- | 36.66 +/- 1.09 | -- | -- | 52.79 +/- 1.03 | strongest reference here |
| Add-only best | ours, early | 10 | -- | -- | 26.57 +/- 1.06 | -- | -- | 38.35 +/- 0.82 | early ReverseGNN weak |
| Ratio-remove best | ours, early | 10 | -- | -- | 27.32 +/- 1.08 | -- | -- | 38.79 +/- 1.02 | small gain only |
| Edit-start sweep best | ours, early | 10 | -- | -- | 27.06 +/- 1.15 | -- | -- | 38.00 +/- 1.08 | not enough |
| Rank-hardening sweep best | ours, early | 10 | -- | -- | 26.85 +/- 1.02 | -- | -- | 37.67 +/- 1.14 | not enough |
| Threshold-remove sweep best | ours, early | 10 | -- | -- | 26.83 +/- 0.96 | -- | -- | 37.65 +/- 0.93 | remove mostly inactive |
| Early two-decoder `two_decoder_pred` | ours | 3 | 90.34 +/- 0.29 | 2.13 +/- 0.18 | 35.29 +/- 1.65 | 91.98 +/- 0.22 | 2.46 +/- 0.31 | 45.35 +/- 2.35 | major step up on Cora |
| Early two-decoder `two_decoder_pred_remove` | ours | 3 | 90.30 +/- 0.28 | 2.11 +/- 0.18 | 35.42 +/- 1.53 | 92.02 +/- 0.16 | 2.87 +/- 0.33 | 46.81 +/- 0.44 | removal still not robust |
| `heart_decoder_scorer_v1` | ours, rejected | 10 | -- | -- | 23.49 / 23.11 | -- | -- | 36.64 / 36.84 | two add-ratio settings; weak |
| Current full method：CP target-1 repair, removal off | ours, current | 10 | 90.27 +/- 0.66 | 2.03 +/- 0.37 | 36.60 +/- 1.60 | 92.75 +/- 0.44 | 3.00 +/- 0.32 | 51.93 +/- 1.34 | Cora essentially tied with NCNC; Citeseer above BUDDY, below NCNC |

Current HeaRT run status：

- Launcher：`run_heart_current_cp_target1_repair_20260602.sh`
- Log：`results/heart_current_cp_target1_repair_20260602_tmux.log`
- Seed logs：`results/sweep_logs/heart_current_cp_target1_repair_20260602_*_two_decoder_pred_s*.txt`
- Summary：`results/heart_current_cp_target1_repair_20260602_summary.csv`
- Completed：Cora seeds `0-9`; Citeseer seeds `0-9`
- Finished：2026-06-04 01:59:14 +08:00

Source：

- `results/heart_two_decoder_editor_20260430_summary.csv`
- `results/heart_current_cp_target1_repair_20260602_summary.csv`
- `results/heart_current_cp_target1_repair_20260602_runs.csv`
- `results/heart_current_cp_target1_repair_20260602_tmux.log`
- `results/sweep_logs/heart_current_cp_target1_repair_20260602_*_two_decoder_pred_s*.txt`
- `heart_aligned_result_note.md`

## Component Ablation Map

Paper-ready table/caption version：`paper_ready_ablation_tables_20260603_zh-TW.md`

### Decision Summary

| Component | What we tested | Result | Decision |
| --- | --- | --- | --- |
| Architecture | old ReverseGNN / CoEBA vs two-decoder | two-decoder wins random fair 10-seed | keep two-decoder |
| Final scorer | dot vs `pair_residual_struct` | `pair_residual_struct` gives +7.21 Cora / +3.15 Citeseer Hit@10 | strongly keep prediction head |
| Edit decoder | `pair_mlp_struct` / `mlp_pair` / `bilinear` | `pair_mlp_struct` best shared default | keep `pair_mlp_struct` |
| Additions / compactness | remove additions, remove compactness, remove both | 5-seed looked good but 10-seed did not hold | keep redesigned augmentation |
| Endpoint rule | broad endpoint vs C0p-to-noncompact CP | C0p-to-noncompact improves Citeseer and balances result | keep |
| Degree target | total-degree 2/3/4, intra-cluster target-1 | CP intra-cluster target-1 best repair story | keep |
| Removal | normal, forced, relaxed | normal not helpful; forced removal hurts | disable |
| Push / repulsion | weak and mid push | active but mixed; Cora/Citeseer tradeoff | do not adopt main |
| Backbone | VGNAE vs integrated MaskGAE / CIMAGE | integrated MaskGAE/CIMAGE weaker in our pipeline | keep VGNAE |
| Scalar tuning | `bce005`, `enc010`, others | no cross-dataset stable gain | keep defaults |
| Protocol | no-leak vs full-graph / public-code | leakage changes story | main claim uses ARON no-leak |
| Runtime | wall-clock audit | current method 90-100 min/seed | optimize next |

### Key Ablation Evidence

| Ablation | Best / Relevant Result | Conclusion |
| --- | --- | --- |
| Two-decoder architecture | Cora 70.49 vs Old 68.56; Citeseer 73.56 vs Old 70.81 | separation helps |
| Prediction scorer | dot 63.50 -> pred 70.71 on Cora; dot 70.84 -> pred 73.99 on Citeseer | largest single gain |
| Edit decoder | `pair_mlp_struct`: Cora 69.15, Citeseer 73.85; `mlp_pair`: Cora 69.22, Citeseer 72.22; `bilinear`: Cora 68.80, Citeseer 70.46 | edit diagnostic alone is misleading |
| Add/compact 10-seed | current 70.49/73.56 vs `add000_compact000` 69.92/72.18 | do not simply remove augmentation |
| Endpoint/cap | `c0p_noncompact_soft025_cap010`: Cora 70.06, Citeseer 75.08 | endpoint choice matters |
| Degree repair | CP target-1 repair `addr020`: Cora 71.92, Citeseer 76.79 | current random main method |
| Removal | forced removal cap 10/20/50 all worse, Citeseer drops hard | disable removal |
| Push | baseline 74.45/76.70; weak 74.64/75.82; mid 73.43/76.92 | mixed, not main |
| Integrated MaskGAE backbone | Cora 67.59, Citeseer 71.52 | weaker than VGNAE current |
| Integrated CIMAGE-full backbone | Cora 60.22, Citeseer 63.22 | reject |
| Official MaskGAE Edge | Cora 75.71, Citeseer 78.24 | strong standalone baseline; integration mismatch matters |
| Runtime | current random ~92-95 min/seed; HeaRT current ~89-94 min/seed completed runs | speed is main engineering limitation |

## What Is Missing / Still Pending

| Item | Status | Priority |
| --- | --- | --- |
| Push + removal interaction | not run | P1 diagnostic only |
| True edited-graph correctness / saved edited adjacency snapshots | missing | P1 |
| Runtime instrumentation / per-component profiling | missing | P1 |
| Candidate-only scoring speed ablation | missing | P1 |
| One-decoder / pred-only simplified method | proposed, not run | P1 |
| VGNAE no-editor / backbone-only baseline | launcher ready, not run | P1 |
| BUDDY / NCNC under ARON random no-leak split | not run locally | P1 if random-split leaderboard needs them |
| CIMAGE / MaskGAE focused objective ablation | optional | P2 |

## Current Narrative

1. On ARON random no-leak, the method evolved from compactness-driven ReverseGNN into a two-decoder method where `pair_residual_struct` final prediction is the main gain.
2. CP intra-cluster target-1 repair is the strongest graph-edit redesign so far.
3. Removal is consistently harmful when it actually removes edges.
4. Push is real but mixed, so it remains diagnostic.
5. VGNAE remains the best integrated backbone; standalone MaskGAE is strong, but integrated MaskGAE/CIMAGE are not aligned with our editor pipeline.
6. On HeaRT, the current full method now reaches Cora `36.60 +/- 1.60`, essentially tied with NCNC `36.66 +/- 1.09`, and Citeseer `51.93 +/- 1.34`, above BUDDY `48.61 +/- 1.32` but still below NCNC `52.79 +/- 1.03`.
7. The main limitation is runtime; the next engineering push should be edge-only validation, candidate-only scoring, caching, and profiling.
