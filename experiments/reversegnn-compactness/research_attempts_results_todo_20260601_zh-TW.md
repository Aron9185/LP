# ReverseGNN 研究嘗試、結果與結論整理

日期：2026-06-01

這份整理採用「我們做了哪些嘗試 → 結果 → 結論」的格式，最後列出目前採用的完整方法與 TODO。內容根據本機實驗報告與結果檔整理，主要來源在 `ARON/experiments/reversegnn-compactness` 與 `official_baselines`。

## 總結

目前主線不是「單純讓 cluster 更 compact」，而是：

1. 用 VGNAE 產生 embeddings。
2. 用 GMM 找 CP/C0p target masks。
3. 用結構感知 edit decoder 做 temporary graph augmentation。
4. 用 C0p-to-noncompact CP endpoint rule 加上 CP intra-cluster target-1 repair 修補 cluster 內孤兒節點。
5. 用 separate structure-aware prediction head 做最終 link prediction。
6. 評估時以 ARON no-leak protocol 作為主結果，不混入 full-graph leakage 結果。

目前最重要的結論：

- Structure-aware prediction head 是主要增益來源。
- Compactness/radius movement 本身不是可靠目標。
- Removal 不該放進目前預設方法。
- VGNAE 仍比 integrated MaskGAE/CIMAGE backbone 更適合我們的 editor pipeline。
- ARON no-leak 是主 protocol；full-graph 與 CIMAGE public-code/default 只能當診斷或參考。
- 目前方法的訓練速度偏慢，這是正式 limitation；後續需要把 runtime profiling 與 fast-mode ablation 納入主線。

## 嘗試、結果與結論

### 1. Compactness / Radius-Driven ReverseGNN

嘗試：

- 比較 frozen vs dynamic cluster targets。
- 測試 radius objective、hybrid objective、add ratio、pull strength。
- 觀察 CP/C0p radius 是否能對應到 Hit@10 改善。

結果：

- Dynamic compactness 確實可以讓 radius 下降。
- 但 frozen baseline 在 Hit@10 上常常更強。
- Radius 下降與 final link prediction 指標沒有穩定正相關。

結論：

- 不能把「讓 cluster 更 compact」當成單一優化目標。
- 後續所有方法都必須同時看 AUROC、AP、Hit@10，而不是只看 radius 或 compactness diagnostics。

主要 artifacts：

- `results/research_matrix_summary.csv`
- `results/stageA_corrected_summary.csv`
- `results/stageB_results.csv`

### 2. HeaRT-Aligned Evaluation

嘗試：

- 對齊 HeaRT `samples.npy` split。
- 使用 full validation、Hit@10 checkpoint selection、700 epochs、10 seeds。
- 與 BUDDY、NCNC baseline 對照。
- 測試 add-only、ratio-remove、edit-start、rank-hardening、threshold-remove。

結果：

| Method family | Cora Hit@10 | Citeseer Hit@10 | 解讀 |
| --- | ---: | ---: | --- |
| Add-only best | 26.57 +/- 1.06 | 38.35 +/- 0.82 | 穩定但偏弱 |
| Ratio-remove best | 27.32 +/- 1.08 | 38.79 +/- 1.02 | 當時 ReverseGNN 最好設定，但增益小 |
| Edit-start sweep best | 27.06 +/- 1.15 | 38.00 +/- 1.08 | 延後 editing 沒補上差距 |
| Rank-hardening sweep best | 26.85 +/- 1.02 | 37.67 +/- 1.14 | 更硬的 decoder ranking 沒幫助 |
| Threshold-remove sweep best | 26.83 +/- 0.96 | 37.65 +/- 0.93 | 幾乎沒有真的 remove edges |
| BUDDY baseline | 30.34 +/- 1.02 | 48.61 +/- 1.32 | 明顯較強 |
| NCNC baseline | 36.66 +/- 1.09 | 52.79 +/- 1.03 | 明顯較強 |

4/30 之後有 newer HeaRT setting 結果，但要注意它不是目前完整 CP target-1 repair 主方法，而是 two-decoder editor 的早期 HeaRT samples 重跑，且只有 seeds `0-2`：

| Method / Config | Seeds | Cora Hit@10 | Citeseer Hit@10 | 解讀 |
| --- | ---: | ---: | ---: | --- |
| `two_decoder_pred` | 3 | 35.29 +/- 1.65 | 45.35 +/- 2.35 | 比 4/29 舊 ReverseGNN HeaRT 設定明顯好 |
| `two_decoder_pred_remove` | 3 | 35.42 +/- 1.53 | 46.81 +/- 0.44 | 早期 two-decoder HeaRT 裡的最好共享設定 |
| hard-remove best follow-up | 3 | 34.91 +/- 2.28 | 47.03 +/- 1.54 | Citeseer 略高，但 Cora 沒有更好；removal 不穩 |
| `heart_decoder_scorer_v1` | 10 | 23.49 / 23.11 | 36.64 / 36.84 | 兩個 add ratio 都偏弱，不採用 |

目前完整 CP target-1 repair 主方法的 HeaRT run 已有 interim result；Cora 已完成 10 seeds，Citeseer 還在跑：

| Method / Config | Seeds | Cora Hit@10 | Citeseer Hit@10 | 解讀 |
| --- | ---: | ---: | ---: | --- |
| current full method：CP target-1 repair, removal off | Cora 10 / Citeseer 5 done | 36.60 +/- 1.60 | 51.47 +/- 1.27 | Cora 幾乎貼近 NCNC；Citeseer 已高於 BUDDY，待 10 seeds 完成後再正式下結論 |

結論：

- 舊方法與 BUDDY/NCNC 的差距不是單純 measurement mismatch。
- 需要改 scoring/training interface，不能只靠 add/remove heuristic。
- 4/30 two-decoder editor 在 HeaRT samples 上有明顯進步：Cora 約 `35.4`、Citeseer 約 `46.8` Hit@10。
- 但這個 newer HeaRT result 只有 3 seeds，且不是目前採用的 CP target-1 repair 完整方法，所以不能當作最終 HeaRT 主結果。
- 現行完整方法的 HeaRT matched run 已經啟動且部分完成；Cora 10-seed result 已可引用為 interim/full-Cora result，Citeseer 仍需等 seeds `5-9` 完成。

主要 artifact：

- `heart_aligned_result_note.md`
- `results/heart_two_decoder_editor_20260430_summary.csv`
- `results/heart_two_decoder_hard_remove_floor0_add001_rm005_20260501_summary.csv`
- `results/heart_two_decoder_hard_remove_floor0_add0005_rm005_20260502_cora_rescue_summary.csv`
- `results/heart_decoder_scorer_v1_p1p0_a0p01_summary.csv`
- `results/heart_decoder_scorer_v1_p1p0_a0p02_summary.csv`
- `results/heart_current_cp_target1_repair_20260602_tmux.log`

### 3. HeaRT-Like Decoder Objective

嘗試：

- 在 dynamic ReverseGNN 中加入 HeaRT-like ranking objective。
- 比較 old frozen、old dynamic、new dynamic bilinear hybrid。

結果：

- 新 dynamic method 在 Cora/Citeseer 優於舊 dynamic baselines。
- 但仍沒有超過最強 frozen baseline 的 Hit@10。
- `edit_add_rank_mean` 從舊 baseline 的 0 變成正值，代表 decoder ranking objective 確實有作用。

結論：

- HeaRT-like ranking objective 是有用訊號，但還不是完整解法。
- 下一步需要把 edit decoder 與 final prediction scorer 分開。

主要 artifact：

- `results/heart_compare_deep_report.md`

### 4. 雙 Decoder 架構

嘗試：

- 將 edit decoder 與 final prediction head 分離。
- Edit decoder 使用 `pair_mlp_struct`。
- Prediction head 使用 `pair_residual_struct`。
- Final score source 改成 `pred_decoder`。

結果：

Random-split fair 10-seed 結果：

| Dataset | New two-decoder | Old ReverseGNN | CoEBA | 新方法 vs Old ReverseGNN | 新方法 vs CoEBA |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cora Hit@10 | 70.49 +/- 5.11 | 68.56 +/- 2.56 | 67.78 +/- 3.71 | +1.94 | +2.71 |
| Citeseer Hit@10 | 73.56 +/- 2.73 | 70.81 +/- 2.23 | 68.92 +/- 2.86 | +2.75 | +4.64 |

結論：

- 雙 decoder 架構成立。
- Separate prediction head 是當時最重要的提升。
- `pred_decoder` 應作為 final score source。

主要 artifacts：

- `post_0430_commit_experiment_report.md`
- `results/random_fair10_report.md`

### 5. Dot Scorer vs Structure-Aware Prediction Head

嘗試：

- 固定其他設定，比較 dot scorer 與 structure-aware prediction head。

結果：

| Dataset | Dot Scorer Hit@10 | Prediction Head Hit@10 | Gain |
| --- | ---: | ---: | ---: |
| Cora | 63.50 +/- 2.90 | 70.71 +/- 3.35 | +7.21 |
| Citeseer | 70.84 +/- 1.99 | 73.99 +/- 3.84 | +3.15 |

結論：

- Prediction head 是主要增益來源。
- Final evaluation 不應回到 dot product。

### 6. Removal / Edge Deletion

嘗試：

- 測試 normal removal。
- 再用 relaxed constraints 強迫實際 removal 發生，測試 remove cap 10/20/50。

結果：

Normal removal：

| Dataset | No removal Hit@10 | Removal Hit@10 | 差異 |
| --- | ---: | ---: | ---: |
| Cora | 70.49 +/- 5.11 | 70.34 +/- 4.63 | -0.15 |
| Citeseer | 73.56 +/- 2.73 | 72.90 +/- 2.64 | -0.66 |

Forced removal：

| Remove Cap | Dataset | Hit@10 | Baseline Hit@10 | 差異 |
| ---: | --- | ---: | ---: | ---: |
| 10 | Cora | 68.43 +/- 5.38 | 69.15 | -0.72 |
| 10 | Citeseer | 68.26 +/- 2.75 | 73.85 | -5.58 |
| 20 | Cora | 67.86 +/- 6.42 | 69.15 | -1.29 |
| 20 | Citeseer | 67.47 +/- 3.39 | 73.85 | -6.38 |
| 50 | Cora | 67.17 +/- 5.60 | 69.15 | -1.98 |
| 50 | Citeseer | 67.16 +/- 5.31 | 73.85 | -6.69 |

結論：

- Removal 不只是 inactive；一旦強迫它真的刪邊，整體表現更差。
- 目前主線應關閉 removal。

主要 artifact：

- `results/random_two_decoder_capped_remove_diag_report.md`

### 7. Scalar Tuning

嘗試：

- 測試 `prediction_bce_weight=0.05`。
- 測試 `prediction_encoder_weight=0.10`。
- 先 5 seeds screening，再做 10-seed confirmation。

結果：

| Dataset | Variant | Hit@10 | 差異 |
| --- | --- | ---: | ---: |
| Cora | current | 70.49 +/- 5.11 | 0.00 |
| Cora | `bce005` | 70.17 +/- 4.34 | -0.32 |
| Cora | `enc010` | 71.14 +/- 2.71 | +0.65 |
| Citeseer | current | 73.56 +/- 2.73 | 0.00 |
| Citeseer | `bce005` | 73.69 +/- 3.91 | +0.13 |

結論：

- `bce005` 不是跨資料集升級。
- `enc010` 是 Cora-only 小訊號，但 seed-wise 不穩，不適合升級成主設定。

主要 artifact：

- `results/random_two_decoder_tune_confirm10_report.md`

### 8. Edit Decoder Ablation

嘗試：

- 固定 prediction head，只替換 edit decoder：
  - `pair_mlp_struct`
  - `mlp_pair`
  - `bilinear`

結果：

| Dataset | Edit Decoder | Hit@10 | 差異 |
| --- | --- | ---: | ---: |
| Cora | `pair_mlp_struct` | 69.15 +/- 7.07 | 0.00 |
| Cora | `mlp_pair` | 69.22 +/- 7.91 | +0.08 |
| Cora | `bilinear` | 68.80 +/- 9.54 | -0.34 |
| Citeseer | `pair_mlp_struct` | 73.85 +/- 2.74 | 0.00 |
| Citeseer | `mlp_pair` | 72.22 +/- 4.10 | -1.63 |
| Citeseer | `bilinear` | 70.46 +/- 1.89 | -3.38 |

結論：

- `pair_mlp_struct` 是目前最好的共享 edit decoder。
- Edit decoder 自身 diagnostic Hit@10 變高，不代表 final prediction 會變好。
- 不能只最佳化 edit decoder diagnostic。

主要 artifact：

- `results/random_two_decoder_decoder_ablation_report.md`

### 9. Augmentation / Compactness Ablation

嘗試：

- 測試是否能拿掉 decoded additions。
- 測試是否能拿掉 compactness。
- 測試 stronger/frozen compactness。
- 測試 `add000_compact000`。

結果：

5-seed 時 `add000_compact000` 看起來不錯：

| Dataset | Current Hit@10 | `add000_compact000` Hit@10 | 差異 |
| --- | ---: | ---: | ---: |
| Cora | 69.15 +/- 7.07 | 70.13 +/- 4.84 | +0.98 |
| Citeseer | 73.85 +/- 2.74 | 74.29 +/- 2.99 | +0.44 |

但 10-seed confirmation 沒有撐住：

| Dataset | Current add+compact | `add000_compact000` | 差異 |
| --- | ---: | ---: | ---: |
| Cora | 70.49 +/- 5.11 | 69.92 +/- 5.11 | -0.57 |
| Citeseer | 73.56 +/- 2.73 | 72.18 +/- 3.56 | -1.38 |

結論：

- 不能直接刪掉 augmentation 與 compactness。
- 舊 augmentation 可能 noisy，但 final prediction 仍受益。
- 下一步應該 redesign endpoint/pulling，而不是直接 `add000_compact000`。

主要 artifacts：

- `results/random_two_decoder_aug_compact_ablation_report.md`
- `results/random_two_decoder_add000_compact000_confirm10_20260515_summary.csv`

### 10. C0p-to-Noncompact Endpoint Rule

嘗試：

- 新增 `--decoded_require_c0p_noncompact_endpoint`。
- 讓 decoded edges 優先連 C0p anchor 與 non-C0p CP partner。
- 測試 soft pull、endpoint rule、per-node cap。

結果：

| Variant | Cora Hit@10 | Citeseer Hit@10 | 解讀 |
| --- | ---: | ---: | --- |
| current_pull100 | 69.60 +/- 6.38 | 73.41 +/- 2.01 | local baseline |
| softpull025 | 68.39 +/- 7.73 | 71.74 +/- 2.31 | 較差 |
| softpull010 | 71.46 +/- 7.84 | 70.90 +/- 0.94 | Cora-only gain |
| c0p_noncompact_soft025 | 69.15 +/- 7.57 | 75.03 +/- 2.46 | Citeseer 明顯提升 |
| c0p_noncompact_soft025_cap010 | 70.06 +/- 7.19 | 75.08 +/- 2.31 | 最平衡的 5-seed candidate |

結論：

- 問題不是單純 pull strength，而是 endpoint selection。
- C0p-to-noncompact CP endpoint rule 比舊的 broad additions 更合理。
- Per-node cap 對 Cora 較有幫助。

主要 artifact：

- `post_0514_experiment_report.md`

### 11. Degree-Targeted / CP Target-1 Repair

嘗試：

- 加入 `--decoded_add_degree_target`。
- 改成 intra-cluster degree target，而不是只看 total degree。
- 測試 CP target-1 guarantee。
- 允許 repair pass 優先修補 intra-cluster degree 低於 1 的 CP nodes。

結果：

目前主 baseline：

| Variant | Dataset | AUROC | AP | Hit@10 |
| --- | --- | ---: | ---: | ---: |
| CP target-1 repair, capped | Cora | 95.99 | 96.40 | 71.92 +/- 5.61 |
| CP target-1 repair, capped | Citeseer | 96.43 | 96.96 | 76.79 +/- 2.18 |

修補診斷：

- Cap-off `addr020` run 的所有 repair checks 都達到 `need_after=0`、`unrepaired=0`、`cluster_bad_after=0`。
- Cora/Citeseer 各 5 seeds，共 6600 repair checks。

結論：

- Target-1 repair 是目前最有效的 augmentation redesign。
- 它修補的是 GMM cluster 內的 graph-neighborhood mismatch，而不是 global degree isolation。

主要 artifact：

- `results/random_two_decoder_cp_target1_repair_c0p_noncompact_soft025_cap010_cp_dtarget1_guarantee_addr020_20260520_summary.csv`

### 12. GMM 與 Cluster-Orphan Correctness

嘗試：

- 寫 reusable diagnostic：`diagnose_gmm_cluster_orphans.py`。
- 對 synthetic embeddings 測試 GMM labels。
- 對 Cora/Citeseer 檢查 CP/C0p intra-orphans。
- 做 seed-0 t-SNE visualization。
- 做 reconstructed repair comparison。

結果：

Original graph aggregate：

| Dataset | Global Degree-0 | CP Intra-Orphan | C0p Intra-Orphan | CP Intra Avg | C0p Intra Avg |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cora | 0.00% | 8.76% | 5.93% | 2.91 | 3.22 |
| Citeseer | 1.44% | 10.85% | 8.00% | 2.28 | 2.51 |

Repair comparison, seed 0：

| Dataset | Original CP Orphan | Repaired CP Orphan | Original C0p Orphan | Repaired C0p Orphan | Added Edges |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cora | 10.20% | 0.00% | 7.12% | 0.00% | 645 |
| Citeseer | 10.59% | 0.00% | 7.62% | 0.00% | 799 |

結論：

- GMM label 本身的計算是合理的。
- Intra-orphan 大多不是 global isolated nodes，而是「embedding cluster 與原圖鄰居不一致」。
- Target-1 repair 確實能補上 same-cluster connectivity。

主要 artifacts：

- `post_0521_experiment_report.md`
- `results/gmm_orphan_diagnostics/`

### 13. Integrated MaskGAE / CIMAGE Backbone

嘗試：

- 不是只換 editor 的 backbone；我們用 `--ae_backbone` 把整個 ARON pipeline 的 autoencoder encoder 從 VGNAE 換成 MaskGAE / CIMAGE。
- 這個 encoder 產生的 embeddings 會同時供 reconstruction/contrastive learning、GMM/CP/C0p、edit decoder、prediction head 使用。
- 仍保留我們原本 ARON 的 contrastive learning 訓練框架，也就是 `intra_view_CL_loss` / `inter_view_CL_loss` 這些 loss 沒有換成官方 MaskGAE/CIMAGE 的完整訓練流程。
- MaskGAE integrated run 額外加入 masked-feature reconstruction loss。
- CIMAGE integrated run 額外加入 factor reconstruction 與 cluster/modularity-style loss。
- 保持 editor/repair recipe 盡量一致。

因此這組實驗比較精確的名稱應該是：

- `Integrated MaskGAE/CIMAGE AE backbone inside ARON editor pipeline`

而不是：

- `Only editor backbone replacement`

結果：

| Backbone inside editor | Dataset | Seeds | AUROC | AP | Hit@10 |
| --- | --- | ---: | ---: | ---: | ---: |
| VGNAE CP target-1 repair | Cora | 5 | 95.99 | 96.40 | 71.92 +/- 5.61 |
| VGNAE CP target-1 repair | Citeseer | 5 | 96.43 | 96.96 | 76.79 +/- 2.18 |
| MaskGAE backbone | Cora | 5 | 94.46 | 95.41 | 67.59 +/- 2.60 |
| MaskGAE backbone | Citeseer | 5 | 94.80 | 95.72 | 71.52 +/- 3.62 |
| CIMAGE-full backbone | Cora | 3 | 91.91 | 92.79 | 60.22 +/- 8.68 |
| CIMAGE-full backbone | Citeseer | 3 | 89.54 | 92.32 | 63.22 +/- 3.74 |

結論：

- VGNAE 仍是目前 editor pipeline 最適合的 backbone。
- MaskGAE standalone 很強，但 integrated MaskGAE backbone 在我們 pipeline 裡較弱，表示問題是 integration alignment，不是 MaskGAE 本身弱。
- CIMAGE 的 cluster/factor objective 目前與我們的 GMM CP/C0p editor 邏輯不夠對齊。

主要 artifacts：

- `post_0521_experiment_report.md`
- `results/random_two_decoder_maskgae_backbone_*_summary.csv`
- `results/random_two_decoder_cimage_full_backbone_*_summary.csv`

### 14. Official CIMAGE / MaskGAE Baselines 與 Protocol

嘗試：

- 跑 CIMAGE authors code 與 MaskGAE authors code。
- 分成三種 protocol：
  - ARON no-leak。
  - ARON split + full-graph training。
  - CIMAGE public-code/default protocol。

結果：ARON no-leak

| Method | Dataset | AUROC | AP | Hit@10 |
| --- | --- | ---: | ---: | ---: |
| ReverseGNN | Cora | 95.93 +/- 0.59 | 96.51 +/- 0.38 | 74.70 +/- 2.78 |
| CIMAGE authors code | Cora | 86.42 +/- 0.27 | 88.52 +/- 0.24 | 47.00 +/- 2.86 |
| MaskGAE Edge authors code | Cora | 96.83 +/- 0.14 | 97.10 +/- 0.18 | 75.71 +/- 4.68 |
| ReverseGNN | Citeseer | 95.98 +/- 0.12 | 96.65 +/- 0.10 | 75.31 +/- 1.11 |
| CIMAGE authors code | Citeseer | 89.94 +/- 1.41 | 92.10 +/- 1.05 | 61.98 +/- 3.96 |
| MaskGAE Edge authors code | Citeseer | 96.87 +/- 0.15 | 97.33 +/- 0.09 | 78.24 +/- 0.95 |

結果：full-graph leakage diagnostic

- ReverseGNN 與 MaskGAE 幾乎飽和。
- CIMAGE 也大幅提升。
- 這證明 full-graph visibility 會嚴重墊高 link prediction 指標。

結論：

- 主 claim 必須使用 ARON no-leak。
- CIMAGE public-code/default 與 full-graph 結果只能作為 protocol reference。
- 不應該把 leaky 與 no-leak 結果放在同一張 leaderboard。

主要 artifacts：

- `post_0531_protocol_comparison_report.md`
- `/home/retro/official_baselines/runs/20260528_aron_split/summary.md`
- `/home/retro/official_baselines/runs/20260529_aron_split_fullgraph_leakage/`
- `/home/retro/official_baselines/runs/20260531_cimage_paper_multiseed_fixed/`

### 15. Noncompact / Noise Push

嘗試：

- 完成真正 full-epoch run，不再只看 smoke rows。
- 在 CP target-1 repair 主設定上比較：
  - `baseline_cp_pull`：維持 CP pull，不做 push。
  - `c0p_pull_only`：改成 C0p pull，不做 push。
  - `c0p_pull_push_weak`：C0p pull，加 noncompact/noise weak push。
  - `c0p_pull_push_mid`：C0p pull，加 noncompact/noise mid push。

結果：

| Variant | Push strength | Dataset | Seeds | AUROC | AP | Hit@10 |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| `baseline_cp_pull` | none | Cora | 3 | 95.96 +/- 0.44 | 96.53 +/- 0.24 | 74.45 +/- 2.88 |
| `baseline_cp_pull` | none | Citeseer | 3 | 96.49 +/- 0.22 | 97.00 +/- 0.16 | 76.70 +/- 2.16 |
| `c0p_pull_only` | none | Cora | 3 | 95.88 +/- 0.49 | 96.47 +/- 0.27 | 74.38 +/- 1.19 |
| `c0p_pull_only` | none | Citeseer | 3 | 96.04 +/- 0.43 | 96.61 +/- 0.36 | 74.80 +/- 0.83 |
| `c0p_pull_push_weak` | noncompact 0.05, noise 0.02 | Cora | 3 | 96.01 +/- 0.76 | 96.58 +/- 0.58 | 74.64 +/- 3.51 |
| `c0p_pull_push_weak` | noncompact 0.05, noise 0.02 | Citeseer | 3 | 96.25 +/- 0.32 | 96.77 +/- 0.39 | 75.82 +/- 5.57 |
| `c0p_pull_push_mid` | noncompact 0.10, noise 0.05 | Cora | 3 | 96.00 +/- 0.54 | 96.56 +/- 0.34 | 73.43 +/- 2.95 |
| `c0p_pull_push_mid` | noncompact 0.10, noise 0.05 | Citeseer | 3 | 96.47 +/- 0.12 | 96.98 +/- 0.13 | 76.92 +/- 1.72 |

診斷：

- Push 有真的作用，不是 inactive：
  - Cora 約 push `528` 個 noncompact nodes、`36` 個 noise nodes。
  - Citeseer 約 push `646` 個 noncompact nodes、`68` 個 noise nodes。
- 但 final Hit@10 沒有穩定提升：
  - Weak push 對 Cora 只有很小提升，Citeseer 反而低於 `baseline_cp_pull`。
  - Mid push 對 Citeseer 有小幅提升，但 Cora 明顯下降。

結論：

- Full-epoch noncompact/noise push 已完成，但目前不升級成主方法。
- Push signal 會改變 embedding geometry，但 final link prediction gain 不穩定。
- 目前主線仍應保留 `baseline_cp_pull` / CP target-1 repair；push 只能列為 mixed ablation。

主要 artifacts：

- `results/random_two_decoder_pull_push_baseline_cp_pull_20260531_summary.csv`
- `results/random_two_decoder_pull_push_c0p_pull_only_20260531_summary.csv`
- `results/random_two_decoder_pull_push_c0p_pull_push_weak_20260531_summary.csv`
- `results/random_two_decoder_pull_push_c0p_pull_push_mid_20260531_summary.csv`

## 目前採用的完整方法

### Evaluation Protocol

- Main protocol：ARON random split, no-leak。
- Training graph：移除 validation/test positive edges。
- Validation positives：5%。
- Test positives：10%。
- Negatives：使用 ARON cached flat negative edge lists。
- Main metrics：AUROC、AP、Hit@10。
- Checkpoint selection：依 validation metric，避免 test leakage。
- Full-graph training 結果只做診斷，不作為主結果。

### Model / Encoder

- Backbone：VGNAE。
- 保留 VGNAE 是因為目前與 GMM CP/C0p editor、pairwise ranking objective 最對齊。
- 不採用 integrated MaskGAE/CIMAGE backbone 作為主線。

### Cluster / Target Construction

- 使用 dynamic GMM labels。
- 依 embeddings 建立 CP 與 C0p target masks。
- C0p 作為較核心、較可信的 anchor set。
- Noncompact CP nodes 作為需要被拉近或補邊的對象。

### Edit Decoder

- Decoder type：`pair_mlp_struct`。
- 使用 pair embedding features 加上 graph/cluster structural context。
- Edit decoder 負責 decoded graph augmentation proposal。
- Edit decoder 不作為 final scorer。

### Prediction Head

- Prediction decoder：`pair_residual_struct`。
- Final score source：`pred_decoder`。
- 使用 HeaRT-style ranking 與 sampled BCE 訓練。
- 這是目前主要增益來源。

### Graph Augmentation / Repair

- 使用 temporary decoded graph view。
- Add-only；不做 edge removal。
- Same-cluster filtering：開啟。
- Endpoint rule：要求 decoded edge 連接 C0p anchor 與 noncompact CP partner。
- Pull strength：目前主線使用 soft pull 設定。
- Per-node cap：保留，用來避免少數節點吸收過多新增邊。
- Degree target：
  - Target scope：intra-cluster degree。
  - Target nodes：CP。
  - Target value：1。
  - Guarantee repair：開啟。
- Repair priority：先補 degree deficit，再用 decoder confidence 作 tie-breaker。

### Removal Policy

- `decoded_remove_ratio=0.0`。
- 不使用 relaxed removal。
- 不使用 forced cross-cluster removal。
- 原因：normal removal 沒有幫助，forced removal 明顯傷害 AUROC/AP/Hit@10。

### Recommended Current Config Summary

概念上目前採用：

```text
backbone = VGNAE
split_mode = ARON random no-leak
dynamic_c0p_targets = true
edit_decoder = pair_mlp_struct
prediction_head = pair_residual_struct
score_source = pred_decoder
decoded_graph_augment = true
decoded_temporary_view_only = true
decoded_same_cluster_only = true
decoded_require_c0p_noncompact_endpoint = true
decoded_add_degree_target = 1
decoded_add_degree_target_scope = intra_cluster
decoded_add_degree_target_nodes = cp
decoded_guarantee_degree_target = true
decoded_remove_ratio = 0.0
removal = disabled
```

對應主要結果檔：

- `results/random_two_decoder_cp_target1_repair_c0p_noncompact_soft025_cap010_cp_dtarget1_guarantee_addr020_20260520_summary.csv`

## 目前限制：訓練速度

目前主方法的準確率是目前最好的方向之一，但訓練速度明顯偏慢。這不是單純硬體問題，而是方法設計與目前實作共同造成的成本。

最新 runtime audit：

- 詳細整理見 `runtime_speed_audit_20260603_zh-TW.md`。
- 目前 random no-leak CP target-1 repair `addr020`：Cora `94.55 +/- 2.29` 分鐘/seed，Citeseer `91.95 +/- 1.61` 分鐘/seed。
- 目前 HeaRT current full-method run 已完成 Cora seeds `0-2`，平均 `98.15 +/- 8.34` 分鐘/seed；20-run serial job 粗估約 `31-34` 小時。
- 舊版 HeaRT two-decoder 約 `29-30` 分鐘/seed，no-editor `pair_residual_struct` control 約 `27-28` 分鐘/seed；因此 current full method 的慢點不只來自 encoder，而是 prediction head、editor/rewrite/repair 與 full-matrix evaluation 疊加。
- Official CIMAGE wrapper 約 `1.1-1.3` 分鐘/seed；official MaskGAE wrapper 在目前 progress log 是秒級。這不是完全公平 runtime benchmark，但顯示我們 current ARON full method 的工程成本高很多。

主要慢點：

1. Edit decoder `pair_mlp_struct` 與 prediction head `pair_residual_struct` 都是 pairwise structural MLP，會對 node pair 做 chunked all-pairs scoring，時間與記憶體接近 O(N^2)。
2. Structural pair features 包含 degree、common neighbors、RA、AA、cluster/core/CP/C0p/prototype distance 等訊號，特徵比單純 dot product 重很多。
3. Final prediction 使用 `pred_decoder`，所以 validation/test/final scoring 也會走 structure-aware prediction head，而不是便宜的 dot scorer。
4. Dynamic GMM labels、CP/C0p masks、prototype distance、structural context 會在訓練與評估過程重複更新或重設。
5. CP target-1 guarantee repair 需要掃描 same-cluster candidate edges，且 guarantee mode 可能超過原本 add budget/cap 來補齊 degree target。
6. Augmented graph branch 會產生 temporary decoded graph view，等於訓練時多了一段 graph augmentation 與對應的 contrastive/reconstruction 計算。
7. 目前主要 launcher 使用 `epochs=700`、`max_workers=1`、`mlp_pair_max_rows=16`，偏向穩定與省記憶體，但 wall-clock 會很長。

結論：

- 訓練速度是目前方法的主要 engineering limitation。
- 後續不能只報 Hit@10，也應該記錄 wall-clock / per-epoch runtime，否則很難判斷方法是否值得擴到更多資料集或更多 seeds。
- 需要做 fast-mode ablation：保留主要增益來源，同時找出哪些 expensive components 可以降頻、cache、candidate-prune 或延後啟用。

## TODO

重要備註：

- 目前這份 TODO 主要來自本機 artifacts：`post_*.md`、`results/*.csv`、diagnostic reports、launcher scripts，以及目前可追溯的實驗紀錄。
- 因為這個環境讀不到 Notion 頁面內文，所以「只寫在 Notion、但沒有同步到本機報告或腳本」的未完成事項，尚未保證完整列入。
- 如果要把 Notion-only 的待辦也補齊，需要把 Notion 頁面匯出成 Markdown，或直接貼上各頁的未完成 TODO / next steps 區塊，再合併進這份表。

### P0：主結果與論文敘事

1. 將主 claim 固定在 ARON no-leak protocol。
2. 明確標註 full-graph 與 CIMAGE public-code/default 是 diagnostic/reference，不放進主 leaderboard。
3. 決定主表格是否要補齊 matched seed count，例如 ReverseGNN、CIMAGE、MaskGAE 都用同一組 seeds 呈現。
4. 將「ReverseGNN vs CIMAGE」與「ReverseGNN vs MaskGAE」分成兩種敘事：
   - CIMAGE：no-leak 下明顯弱，public-code/default 有 protocol leakage 問題。
   - MaskGAE：standalone 很強，但 integrated backbone 不適合目前 editor pipeline。

### P1：補強目前方法

1. 若需要更強主結果，將 CP target-1 repair 從 5 seeds 擴到更多 seeds。
2. 確認 Cora/Citeseer 是否需要同 seed count 對齊 official baselines。
3. 整理目前採用 config 成一個固定 launcher，避免之後跑錯設定。
4. 將目前主設定寫成 paper-ready method section。

### P1：訓練速度 / Runtime

1. 在 training log 與 summary CSV 補 wall-clock runtime：
   - total training time
   - per-epoch time
   - encoder forward/backward time
   - edit decoder scoring time
   - prediction decoder scoring time
   - GMM / CP / C0p target update time
   - repair pass time
   - validation/test evaluation time
2. 做 runtime profiling，確認真正瓶頸是 edit decoder、prediction decoder、repair pass，還是 evaluation frequency。
3. 做 `mlp_pair_max_rows` sweep，例如 16/32/64；如果 GPU 記憶體允許，較大的 chunk 可能減少 Python loop overhead。
4. 做 candidate-only scoring ablation：
   - Edit decoder 不再每次 full all-pairs score，只 score same-cluster / CP-related / top-k candidate pairs。
   - Prediction decoder 訓練時只 score positive edges 與 sampled negatives；full matrix scoring 只留給必要 evaluation。
5. 將 expensive diagnostics 降頻：
   - 增大 `eval_log_every`
   - repair diagnostics 不必每個 eval 都全量印
   - final full diagnostics 只在 best checkpoint 或最後做
6. 嘗試 cache 可重用的 structural context：
   - degree / CN / RA / AA
   - cluster labels 與 CP/C0p masks
   - prototype distances
   - candidate edge lists
7. 設計 fast-mode launcher：
   - 先用 seeds 0-2 做 sanity check
   - epochs 減半或加 early stopping
   - evaluation 降頻
   - 保留 `pair_residual_struct` 與 CP target-1 repair 的核心設定
8. 在 paper/meeting 表格加入 runtime 欄位，避免只呈現準確率而忽略訓練成本。

### P1：Diagnostics / Correctness

1. 在 training run 中保存 edited adjacency snapshots。
2. 用真實 edited graph 重算 after-edit orphan table，而不是只靠 reconstructed view 或 log counters。
3. 在正常 training logs 加入：
   - `cp_intra_mean_degree`
   - `c0p_intra_mean_degree`
   - CP/C0p intra-degree p10/p50/p90
   - unrepaired node count by reason
4. 補上 guarantee repair 的小型 correctness tests：
   - endpoint constraint 是否會阻擋可修補節點
   - deficit-first greedy 是否會餓死某些節點
   - unrepaired reason 是否能被清楚分類

### P2：下一個方法方向

1. Repulsion / push 已完成 3-seed full-epoch weak/mid run；目前結果 mixed，不升級成主方法。若要繼續，只做更細的 push strength / target scope sweep。
2. 測試 CP pull vs C0p pull 的穩定性，尤其 Citeseer。
3. 若要繼續 MaskGAE integration，做 focused ablation：
   - standalone MaskGAE 強，但 integrated MaskGAE 弱的原因。
   - 是 embedding geometry 不適合 GMM CP/C0p，還是 prediction head/repair 不對齊？
4. 若要繼續 CIMAGE integration，先做 cluster/factor weight ablation，而不是直接當主 backbone。

### P2：整理與交付

1. 將 Notion 頁面整理成：
   - 嘗試總表
   - 結果總表
   - 決策紀錄
   - Protocol ledger
   - TODO board
2. 把重要 artifacts 連到每個 decision。
3. 將 rejected ideas 另外整理，避免之後重複跑已經否定的方向。
4. 為 paper/meeting 準備一頁版 summary：
   - Problem
   - What failed
   - What worked
   - Current method
   - Main result
   - Next step
