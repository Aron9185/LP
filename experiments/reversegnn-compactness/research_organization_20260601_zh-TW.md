# ReverseGNN 研究整理草稿

日期：2026-06-01

這份是可直接搬到 Notion 的研究整理草稿，使用繁體中文與台灣常見用語。因為目前環境無法讀取 Notion 頁面內容，以下內容主要根據本機的實驗報告、結果 CSV、執行腳本與 git 紀錄整理，範圍包含 `ARON/experiments/reversegnn-compactness` 與 `official_baselines`。

## 一句話主軸

我們從「用 compactness 驅動 ReverseGNN 圖編輯」開始，後來發現只看半徑收縮或 decoder 自身診斷分數並不夠；接著轉向雙 decoder 架構與結構感知 prediction head，再把 augmentation 收斂成 C0p-to-noncompact CP 的 target-1 修補機制，最後把公平 no-leak protocol 與 full-graph / CIMAGE public-code 這類診斷性 protocol 清楚分開。

## 目前主要主張

主要 protocol：ARON random edge split。訓練時移除 validation/test positive edges，使用 cached negatives，避免 full-graph edge leakage。

在公平的 ARON no-leak 設定下：

| 方法 / 設定 | 資料集 | Seeds | AUROC | AP | Hit@10 | 解讀 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| ReverseGNN, VGNAE + CP target-1 repair | Cora | 5 | 95.99 | 96.40 | 71.92 +/- 5.61 | 目前主 baseline |
| ReverseGNN, VGNAE + CP target-1 repair | Citeseer | 5 | 96.43 | 96.96 | 76.79 +/- 2.18 | 目前主 baseline |
| CIMAGE authors code, ARON no-leak wrapper | Cora | 3 | 86.42 | 88.52 | 47.00 +/- 2.86 | no-leak 下明顯較弱 |
| CIMAGE authors code, ARON no-leak wrapper | Citeseer | 3 | 89.94 | 92.10 | 61.98 +/- 3.96 | no-leak 下明顯較弱 |
| MaskGAE Edge authors code, ARON no-leak wrapper | Cora | 3 | 96.83 | 97.10 | 75.71 +/- 4.68 | 這個 3-seed slice 略高於 ReverseGNN |
| MaskGAE Edge authors code, ARON no-leak wrapper | Citeseer | 3 | 96.87 | 97.33 | 78.24 +/- 0.95 | 這個 3-seed slice 略高於 ReverseGNN |

Protocol 注意事項：full-graph training 會讓 ReverseGNN 與 MaskGAE 在 ARON held-out edges 上幾乎飽和。這些結果可以用來診斷資料洩漏與 protocol 敏感性，但不應該混進主要排行榜。

主要來源：

- `post_0531_protocol_comparison_report.md`
- `results/random_two_decoder_cp_target1_repair_c0p_noncompact_soft025_cap010_cp_dtarget1_guarantee_addr020_20260520_summary.csv`
- `/home/retro/official_baselines/runs/20260528_aron_split/summary.md`

## 目前方法快照

建議保留的主線方法：

- Encoder/backbone：VGNAE，目前仍最符合 editor pipeline。
- Cluster/core 邏輯：dynamic GMM labels，CP/C0p target masks。
- Edit decoder：`pair_mlp_struct`。
- Prediction head：`pair_residual_struct`。
- 最終分數來源：`pred_decoder`，不是 dot product，也不是 edit decoder。
- Augmentation：add-only temporary decoded graph view。
- Endpoint policy：same-cluster C0p-to-noncompact CP。
- Degree repair：CP intra-cluster target-1 guarantee。
- Add budget：足以做 repair，最佳目前 baseline 有使用 per-node cap。
- Removal：關閉。

## 時間軸

### 2026-03-26 與 2026-04-02：Compactness / ReverseGNN 基礎

在整體故事中的角色：

- 建立 ReverseGNN compactness 與 graph editing 的核心問題。
- 探索 frozen vs dynamic cluster targets、radius objectives、add ratios、pull strengths，以及基本 bilinear/MLP decoder 變體。
- 早期就看到：只讓 radius 變小，不代表 link prediction Hit@10 會變好。

學到的事：

- Frozen baselines 在 Hit@10 上意外地強。
- Dynamic compactness 可以縮小 radius，但不穩定地改善 ranking。
- Add/remove/pull sweep 必須一起看最終 AUROC/AP/Hit@10，不能只看 radius。

相關 artifacts：

- `results/research_matrix_summary.csv`
- `results/stageA_corrected_summary.csv`
- `results/stageB_results.csv`
- `results/stage2_pull_sweep_results.csv`
- `results/stage3a_addratio_sweep.csv`

### 2026-04-16：Research Matrix 與 Deep Dynamic Comparison

在整體故事中的角色：

- 比較舊的 frozen/dynamic baselines，以及帶有 HeaRT-like ranking signal 的新 dynamic bilinear hybrid 方法。

學到的事：

- 新 dynamic method 在 Cora 與 Citeseer 上優於舊 dynamic baselines。
- 但 Hit@10 還是沒有超過最強 frozen baseline。
- HeaRT-like decoder objective 確實有啟動，但方法仍需要更好的 scoring/training interface。

相關 artifact：

- `results/heart_compare_deep_report.md`

### 2026-04-23 到 2026-04-29：HeaRT-Aligned Measurement

在整體故事中的角色：

- 對齊 HeaRT `samples.npy`、full validation、Hit@10 checkpointing，以及 BUDDY/NCNC 等 baseline。

學到的事：

- ReverseGNN 在 HeaRT-samples 的最佳 Hit@10 仍明顯低於 BUDDY/NCNC。
- Ratio removal 只有小幅訊號。
- Edit-start sweep、較硬的 decoder ranking、threshold removal 都沒有補上差距。
- 下一版架構需要把 edit scoring 與 final link prediction 分開。

相關 artifact：

- `heart_aligned_result_note.md`

### 2026-04-30：Structural Decoder 與雙 Decoder 轉向

在整體故事中的角色：

- 從單一 bilinear/dot scoring path，轉向雙 decoder 架構。

改動內容：

- 加入更強的 pair scorer 與 structural pair features。
- 分離 edit scoring 與 final prediction。
- 用 `pair_mlp_struct` 做 edit decoding。
- 用 `pair_residual_struct` 做 prediction head。
- 將 final score source 改成 `pred_decoder`。

學到的事：

- Separate structure-aware prediction head 是主要增益來源。
- Decoder-as-final-scorer 與純 dot scoring 較弱。

相關 artifacts：

- `post_0430_commit_experiment_report.md`
- `results/random_fair10_report.md`

### 2026-05-14：Random-Split Fair 10-Seed 結果與 Ablations

在整體故事中的角色：

- 鎖定第一個 random-split fair 10-seed 結果，對比舊 ReverseGNN 與 CoEBA。

主要結果：

| 資料集 | 新 two-decoder Hit@10 | 舊 ReverseGNN Hit@10 | CoEBA Hit@10 |
| --- | ---: | ---: | ---: |
| Cora | 70.49 +/- 5.11 | 68.56 +/- 2.56 | 67.78 +/- 3.71 |
| Citeseer | 73.56 +/- 2.73 | 70.81 +/- 2.23 | 68.92 +/- 2.86 |

決策：

- 保留 `pair_residual_struct` prediction head。
- 保留 `pair_mlp_struct` 作為共享 edit decoder。
- 預設關閉 removal。
- 不提升 `bce005` 或 Cora-only `enc010` 這類 scalar tuning variants。
- Decoded additions 與 compactness 仍可疑，但當時還不能直接刪掉。

相關 artifacts：

- `post_0430_commit_experiment_report.md`
- `results/random_fair10_report.md`
- `results/random_two_decoder_capped_remove_diag_report.md`
- `results/random_two_decoder_decoder_ablation_report.md`
- `results/random_two_decoder_tune_confirm10_report.md`
- `results/random_two_decoder_aug_compact_ablation_report.md`

### 2026-05-14 到 2026-05-21：No-Add Control、Endpoint Redesign、Degree Repair

在整體故事中的角色：

- 測試 augmentation/compactness 是否能直接刪掉。
- 當 10-seed no-add/no-compact confirmation 沒通過後，改成重新設計 augmentation，而不是直接放棄。

學到的事：

- `add000_compact000` 在 5 seeds 看起來不錯，但 10 seeds 沒有撐住。
- 舊 decoded additions 太廣、太粗。
- C0p-to-noncompact CP endpoint rule 加上 per-node cap，是第一個比較平衡的改進。
- CP target-1 intra-cluster repair 成為目前最好的 augmentation 機制。

目前關鍵結果：

| Variant | 資料集 | AUROC | AP | Hit@10 | 解讀 |
| --- | --- | ---: | ---: | ---: | --- |
| CP target-1 repair, capped | Cora | 95.99 | 96.40 | 71.92 +/- 5.61 | 目前主 baseline |
| CP target-1 repair, capped | Citeseer | 96.43 | 96.96 | 76.79 +/- 2.18 | 目前主 baseline |

相關 artifacts：

- `post_0514_experiment_report.md`
- `post_0521_experiment_report.md`
- `results/random_two_decoder_add000_compact000_confirm10_20260515_summary.csv`
- `results/random_two_decoder_cp_target1_repair_c0p_noncompact_soft025_cap010_cp_dtarget1_guarantee_addr020_20260520_summary.csv`

### 2026-05-21 到 2026-05-25：GMM 與 Cluster-Orphan 正確性

在整體故事中的角色：

- 驗證 GMM labels、CP/C0p masks、orphan diagnostics 的意義是否正確。

學到的事：

- `gmm_labels` 通過 synthetic correctness checks。
- Graph isolation 不會把 GMM label 改成 noise，因為 GMM 只看 embeddings。
- Original-graph CP intra-orphans 是真實訊號，但對 sparse graph 來說仍合理：
  - Cora CP intra-orphan：8.76%
  - Citeseer CP intra-orphan：10.85%
  - Cora global degree-0：0.00%
  - Citeseer global degree-0：1.44%
- Tail clusters 仍重要：有些 cluster 會到約 28-31% intra-orphans。
- Seed-0 reconstructed target-1 repair comparison 顯示，只加 same-cluster edges 就能消除 CP/C0p intra-orphans。
- Repair logs 顯示 cap-off run 的所有 repair checks 都達到 `need_after=0`、`unrepaired=0`、`cluster_bad_after=0`。

相關 artifacts：

- `post_0521_experiment_report.md`
- `diagnose_gmm_cluster_orphans.py`
- `results/gmm_orphan_diagnostics/`

### 2026-05-27 到 2026-05-28：Integrated Backbone Replacement

在整體故事中的角色：

- 測試把我們 editor pipeline 裡的 VGNAE 換成 MaskGAE 或 CIMAGE-style masked autoencoding 是否能改善。

學到的事：

| Editor 內的 backbone | 資料集 | Seeds | AUROC | AP | Hit@10 | 解讀 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| VGNAE CP target-1 repair | Cora | 5 | 95.99 | 96.40 | 71.92 +/- 5.61 | 最符合 editor |
| VGNAE CP target-1 repair | Citeseer | 5 | 96.43 | 96.96 | 76.79 +/- 2.18 | 最符合 editor |
| MaskGAE backbone | Cora | 5 | 94.46 | 95.41 | 67.59 +/- 2.60 | 低於 VGNAE |
| MaskGAE backbone | Citeseer | 5 | 94.80 | 95.72 | 71.52 +/- 3.62 | 低於 VGNAE |
| CIMAGE-full backbone | Cora | 3 | 91.91 | 92.79 | 60.22 +/- 8.68 | 明顯較弱 |
| CIMAGE-full backbone | Citeseer | 3 | 89.54 | 92.32 | 63.22 +/- 3.74 | 明顯較弱 |

決策：

- 保留 VGNAE 作為主 backbone。
- MaskGAE 與 CIMAGE integration 先視為 ablations；除非要特別研究整合失敗原因，否則不是主線。

相關 artifacts：

- `post_0521_experiment_report.md`
- `results/random_two_decoder_maskgae_backbone_*_summary.csv`
- `results/random_two_decoder_cimage_full_backbone_*_summary.csv`

### 2026-05-28 到 2026-06-01：Official Baselines 與 Protocol 分離

在整體故事中的角色：

- 在多種 protocol 下比較 official CIMAGE / MaskGAE code 與 ReverseGNN。

Protocol 決策：

- 主要 claim 必須使用 ARON no-leak。
- Full-graph ARON 與 CIMAGE public-code/default 只能當診斷或參考，不是公平排行榜。

重點解讀：

- CIMAGE 在 ARON no-leak 下大幅下降。
- Official MaskGAE Edge 在 ARON no-leak 下很強。
- Full-graph visibility 會讓 ReverseGNN 與 MaskGAE 幾乎飽和。
- CIMAGE public-code/default 可以用來解釋 paper/public-code comparability，但對我們的 edge-prediction claim 來說有資料洩漏。

相關 artifacts：

- `post_0531_protocol_comparison_report.md`
- `/home/retro/official_baselines/runs/20260528_aron_split/summary.md`
- `/home/retro/official_baselines/runs/20260529_aron_split_fullgraph_leakage/`
- `/home/retro/official_baselines/runs/20260531_cimage_paper_multiseed_fixed/`

## 決策紀錄

### 保留在主線

| 決策 | 原因 |
| --- | --- |
| 使用 ARON no-leak 作為主要 protocol | 避免 held-out positive edges 被訓練看到 |
| Editor pipeline 保留 VGNAE | MaskGAE/CIMAGE integrated backbones 表現低於 VGNAE，即使 standalone MaskGAE 很強 |
| 保留 separate prediction head | Structure-aware prediction head 是相對 dot scoring 最大的增益 |
| 保留 `pair_mlp_struct` edit decoder | Cora/Citeseer 共享設定下最穩 |
| 預設關閉 removal | Normal removal 幾乎 inactive 且略差；forced removal 明顯傷害指標 |
| 保留 CP target-1 repair 作為目前 augmentation | 可以修補 intra-cluster deficits，且改善目前 baseline |

### 已拒絕或降優先順序

| 想法 | 原因 |
| --- | --- |
| Decoder-as-final-scorer | 弱於 separate prediction head |
| Dot-only final scoring | 輸給 `pred_decoder` |
| Bilinear edit decoder 當預設 | 整體較差，尤其 Citeseer |
| 更強/frozen compactness | Radius 變小沒有轉成 final metrics 改善 |
| 直接 no-add/no-compact 當預設 | 5-seed 訊號在 10 seeds 沒撐住 |
| Forced / removal-heavy editing | 傷害 AUROC/AP/Hit@10 |
| 盲目更換成 MaskGAE/CIMAGE backbone | Integrated versions 都低於 VGNAE |
| Full-graph protocol 當主要排行榜 | 會洩漏 held-out positives，結果接近飽和 |

### 暫定 / 還需要更多證據

| 線索 | 狀態 |
| --- | --- |
| Pull-mask scope：CP vs C0p | 2026-05-31 早期 3-seed grid 顯示 CP pull 比 C0p pull-only 更安全，尤其 Citeseer |
| Noncompact/noise push | 目前可用結果主要是 smoke/forced-rewrite checks，還不是成熟 full run |
| CIMAGE cluster/factor ablations | 只有在想解釋 integration failure 時才值得做，不是 VGNAE 主線必要工作 |
| Edited-graph orphan snapshots | 仍需要保存，才能做 artifact-level after-edit orphan diagnostics |

## 建議 Notion 架構

建議建立一個 master page，裡面放：

1. Executive Status
   - 目前主要 claim
   - 目前方法快照
   - 哪些是公平比較，哪些只是診斷

2. Timeline
   - 2026-03-26 與 2026-04-02：Compactness foundation
   - 2026-04-16：Research matrix
   - 2026-04-23/29：HeaRT alignment
   - Structural Decoder
   - 2026-05-14：Random split and ablations
   - 2026-05-21：Endpoint/degree repair
   - GMM and cluster orphan correctness
   - 2026-05-28：Official baselines
   - 2026-06-01：Backbone/protocol comparison

3. Method State
   - 目前預設設定
   - 試過的主要替代方案
   - 每個方案為什麼保留或淘汰

4. Protocol Ledger
   - ARON no-leak
   - ARON split with full-graph training
   - CIMAGE public-code/default
   - HeaRT samples
   - Random split legacy comparisons

5. Evidence Index
   - 報告路徑
   - CSV 路徑
   - 資料集 / seeds
   - 狀態：accepted、rejected、diagnostic、tentative

6. Next Experiments
   - VGNAE endpoint/repulsion work
   - 儲存 edited graph snapshots
   - Node-level intra-degree logging
   - Optional MaskGAE integration ablation

## 你提供的 Notion 頁面對應

| Notion page title | 在 master organization 裡的角色 |
| --- | --- |
| Experiments 3-26 | 早期 compactness/GMM scaffold 與 dynamic-vs-frozen 問題 |
| Experiments 4-2 | Pull/add/radius sweeps 與早期 compactness diagnostics |
| Experiments 4-16 | Research matrix 與 dynamic method comparison |
| Experiments 4-23 | HeaRT-aligned measurement 與 baseline gap |
| Structural Decoder | 雙 decoder architecture pivot |
| Experiments 5-14 | Random-split fair 10-seed win 與 ablations |
| Experiments 5-21 | Endpoint redesign、target-1 repair、下一步 correctness plan |
| GMM And Cluster Orphan Correctness | GMM/orphan validation 與 repair-mechanism diagnostics |
| Experiments 5-28 | Official CIMAGE/MaskGAE baselines 與 no-leak comparisons |
| Backbone Comparison 6-1 | Protocol/backbone synthesis 與 final claim framing |

## 下一步

最有用的下一步：

1. 將 paper main claim 穩定放在 ARON no-leak results。
2. 視需要擴充或確認目前 VGNAE CP target-1 baseline，讓 seed count 對齊。
3. 如果 repulsion 仍是目標假設，完成真正 full-epoch 的 noncompact/noise push experiment。
4. 儲存 edited adjacency snapshots，讓 after-edit orphan diagnostics 不只依賴 reconstructed views 或 log counters。
5. 在一般 training logs 加入 node-level `cp_intra_mean_degree`、`c0p_intra_mean_degree` 與相關 percentiles。
6. 只有在 paper 需要解釋「為什麼 standalone MaskGAE 強，但 integrated MaskGAE 不強」時，才深入做 MaskGAE/CIMAGE integration studies。

