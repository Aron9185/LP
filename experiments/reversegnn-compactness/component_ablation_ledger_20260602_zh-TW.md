# ReverseGNN Component Ablation Ledger

日期：2026-06-02

這份文件把目前實驗拆成 component-level ablation ledger。目標是讓每個設計選擇都有清楚的「保留 / 不採用 / mixed / 待跑」判斷，而不是只照日期回顧實驗。

除非特別註明，以下結果都使用 ARON random no-leak split，數字為百分比 `mean +/- std`。

## 總覽

目前主方法的核心 component decision：

| Component | Current choice | Decision | 理由 |
| --- | --- | --- | --- |
| Encoder backbone | VGNAE | 保留 | Integrated MaskGAE / CIMAGE backbone 在我們 editor pipeline 內都輸給 VGNAE |
| Architecture | Two decoder | 保留 | Edit decoder 與 final prediction scorer 分開後，10-seed fair result 優於 old ReverseGNN / CoEBA |
| Final prediction scorer | `pair_residual_struct` | 強烈保留 | 相對 dot scorer 是目前最大單點增益 |
| Edit decoder | `pair_mlp_struct` | 保留 | 跨 Cora/Citeseer 最穩；edit diagnostic 高不代表 final Hit@10 高 |
| Decoded graph augmentation | temporary view, add-only | 保留但需簡化/加速 | 直接拿掉 add+compact 在 10-seed confirmation 沒撐住 |
| Endpoint rule | C0p-to-noncompact CP | 保留 | 比 broad endpoint 更對齊 cluster-neighborhood mismatch |
| Degree repair | CP intra-cluster target-1 guarantee | 保留 | 目前最有效的 augmentation redesign |
| Removal | disabled | 不採用 | normal removal 沒幫助；forced removal 明顯傷害 |
| Push / repulsion | disabled in main | mixed | full-epoch push 有生效，但 final Hit@10 不穩 |
| Scalar weights | default | 保留 default | `bce005` 不穩；`enc010` 是 Cora-only 小訊號 |

## Main Baselines

用來解讀 ablation 的兩個主要 baseline：

| Baseline | Seeds | Cora AUROC | Cora AP | Cora Hit@10 | Citeseer AUROC | Citeseer AP | Citeseer Hit@10 | 用途 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Older two-decoder default | 10 | 95.95 +/- 0.40 | 96.32 +/- 0.45 | 70.49 +/- 5.11 | 96.16 +/- 0.55 | 96.59 +/- 0.46 | 73.56 +/- 2.73 | 對 old ReverseGNN / CoEBA、scalar tuning、add/compact |
| CP target-1 repair `addr020` | 5 | 95.99 +/- 0.51 | 96.40 +/- 0.46 | 71.92 +/- 5.61 | 96.43 +/- 0.64 | 96.96 +/- 0.43 | 76.79 +/- 2.18 | 目前主方法 baseline |
| Latest pull/push grid baseline `baseline_cp_pull` | 3 | 95.96 +/- 0.44 | 96.53 +/- 0.24 | 74.45 +/- 2.88 | 96.49 +/- 0.22 | 97.00 +/- 0.16 | 76.70 +/- 2.16 | 解讀 push / c0p pull ablation |

## 1. Architecture：Two Decoder

嘗試：

- Edit decoder 與 final prediction head 分離。
- Edit decoder 做 graph augmentation proposal。
- Prediction head 做 final link prediction。

結果：

| Dataset | New two-decoder | Old ReverseGNN | CoEBA | vs Old | vs CoEBA |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cora Hit@10 | 70.49 +/- 5.11 | 68.56 +/- 2.56 | 67.78 +/- 3.71 | +1.94 | +2.71 |
| Citeseer Hit@10 | 73.56 +/- 2.73 | 70.81 +/- 2.23 | 68.92 +/- 2.86 | +2.75 | +4.64 |

Decision：保留 two-decoder architecture。

Source：

- `results/random_fair10_report.md`
- `results/random_fair10_new_vs_old_baselines.csv`

## 2. Final Prediction Scorer

嘗試：

- 比較 final score 用 dot product 或 `pair_residual_struct` prediction head。

結果：

| Dataset | Dot scorer Hit@10 | `pair_residual_struct` Hit@10 | Gain |
| --- | ---: | ---: | ---: |
| Cora | 63.50 +/- 2.90 | 70.71 +/- 3.35 | +7.21 |
| Citeseer | 70.84 +/- 1.99 | 73.99 +/- 3.84 | +3.15 |

Decision：強烈保留 `pair_residual_struct`；不要回到 dot scorer 當 final prediction。

Interpretation：

- 這是目前最強、最穩定的單點 component gain。
- Edit decoder 可以調，但 final scorer 不應退回 dot product。

## 3. Edit Decoder

嘗試：

- 固定 prediction head，只換 edit decoder。

結果：

| Dataset | Edit decoder | AUROC | AP | Hit@10 | Decision |
| --- | --- | ---: | ---: | ---: | --- |
| Cora | `pair_mlp_struct` | 95.93 +/- 0.43 | 96.19 +/- 0.53 | 69.15 +/- 7.07 | keep |
| Cora | `mlp_pair` | 96.16 +/- 0.61 | 96.27 +/- 0.65 | 69.22 +/- 7.91 | tied on Cora only |
| Cora | `bilinear` | 95.93 +/- 0.57 | 96.01 +/- 0.66 | 68.80 +/- 9.54 | reject |
| Citeseer | `pair_mlp_struct` | 96.15 +/- 0.51 | 96.60 +/- 0.40 | 73.85 +/- 2.74 | keep |
| Citeseer | `mlp_pair` | 96.26 +/- 0.36 | 96.63 +/- 0.38 | 72.22 +/- 4.10 | worse |
| Citeseer | `bilinear` | 95.98 +/- 0.27 | 96.40 +/- 0.19 | 70.46 +/- 1.89 | reject |

Decision：保留 `pair_mlp_struct`。

Interpretation：

- `mlp_pair` / `bilinear` 的 edit-decoder diagnostic Hit@10 反而更高，但 final prediction 變差，尤其 Citeseer。
- 後續不應只追求 edit decoder 自己的 diagnostic score。

Source：

- `results/random_two_decoder_decoder_ablation_report.md`
- `results/random_two_decoder_decoder_ablate_mlp_pair_20260512_summary.csv`
- `results/random_two_decoder_decoder_ablate_bilinear_20260512_summary.csv`

## 4. Additions / Compactness

嘗試：

- 拿掉 decoded additions。
- 拿掉 compactness objective。
- 同時拿掉 additions 與 compactness。
- 測 stronger / frozen compactness。

5-seed screen：

| Dataset | Current Hit@10 | `add000_compact000` Hit@10 | Delta |
| --- | ---: | ---: | ---: |
| Cora | 69.15 +/- 7.07 | 70.13 +/- 4.84 | +0.98 |
| Citeseer | 73.85 +/- 2.74 | 74.29 +/- 2.99 | +0.44 |

10-seed confirmation：

| Dataset | Current add+compact | `add000_compact000` | Delta |
| --- | ---: | ---: | ---: |
| Cora | 70.49 +/- 5.11 | 69.92 +/- 5.11 | -0.57 |
| Citeseer | 73.56 +/- 2.73 | 72.18 +/- 3.56 | -1.38 |

Decision：不要直接刪掉 additions / compactness。

Interpretation：

- 5-seed 的 no-add/no-compact 訊號沒有在 10-seed 撐住。
- 但這個 ablation 仍然重要：它說明舊 augmentation 可能 noisy，可是 final prediction 仍受益。
- 下一步是 redesign augmentation/endpoint，而不是直接 `add000_compact000`。

Source：

- `results/random_two_decoder_aug_compact_ablation_report.md`
- `results/random_two_decoder_add000_compact000_confirm10_20260515_summary.csv`

## 5. Endpoint Rule / Per-Node Cap

嘗試：

- 從 broad same-cluster C0p endpoint 改成 C0p-to-noncompact CP endpoint。
- 加 per-node cap，避免少數節點吸收太多新增邊。

結果：

| Variant | Cora Hit@10 | Citeseer Hit@10 | Read |
| --- | ---: | ---: | --- |
| `current_pull100` | 69.60 +/- 6.38 | 73.41 +/- 2.01 | local baseline |
| `softpull025` | 68.39 +/- 7.73 | 71.74 +/- 2.31 | worse |
| `c0p_noncompact_soft025` | 69.15 +/- 7.57 | 75.03 +/- 2.46 | Citeseer gain |
| `c0p_noncompact_soft025_cap010` | 70.06 +/- 7.19 | 75.08 +/- 2.31 | best balanced 5-seed candidate |

Decision：保留 C0p-to-noncompact endpoint + cap 作為後續 repair 的基礎。

Interpretation：

- 問題不是單純 pull strength。
- Endpoint selection 比 stronger/softer pull 更重要。

Source：

- `post_0514_experiment_report.md`
- `results/random_two_decoder_pull_endpoint_*_summary.csv`

## 6. Degree-Targeted Repair

嘗試：

- 先測 total-degree target `2/3/4`。
- 後來確認真正要修的是 intra-cluster degree。
- 最終採用 CP nodes 的 intra-cluster target-1 guarantee repair。

Total-degree target screen：

| Variant | Cora Hit@10 | Citeseer Hit@10 | Read |
| --- | ---: | ---: | --- |
| `dtarget2` | 69.75 +/- 9.23 | 73.98 +/- 2.75 | not enough |
| `dtarget3` | 69.98 +/- 12.08 | 73.67 +/- 2.48 | not enough |
| `dtarget4` | 70.97 +/- 6.62 | 75.65 +/- 2.73 | useful hint, not final |

Final CP intra-cluster target-1 repair：

| Variant | Dataset | AUROC | AP | Hit@10 |
| --- | --- | ---: | ---: | ---: |
| CP target-1 repair, capped | Cora | 95.99 +/- 0.51 | 96.40 +/- 0.46 | 71.92 +/- 5.61 |
| CP target-1 repair, capped | Citeseer | 96.43 +/- 0.64 | 96.96 +/- 0.43 | 76.79 +/- 2.18 |

Decision：保留 CP intra-cluster target-1 guarantee repair。

Interpretation：

- 修的是 GMM cluster 內的 graph-neighborhood mismatch，不是 global degree isolation。
- Repair diagnostics 顯示 `need_after=0`、`unrepaired=0`、`cluster_bad_after=0`。

Source：

- `results/random_two_decoder_degree_target_*_summary.csv`
- `results/random_two_decoder_cp_target1_repair_c0p_noncompact_soft025_cap010_cp_dtarget1_guarantee_addr020_20260520_summary.csv`
- `results/gmm_orphan_diagnostics/`

## 7. Removal

嘗試：

- Normal removal。
- Forced / relaxed removal，確認不是因為 removal inactive。

Normal removal：

| Dataset | No removal Hit@10 | Removal Hit@10 | Delta |
| --- | ---: | ---: | ---: |
| Cora | 70.49 +/- 5.11 | 70.34 +/- 4.63 | -0.15 |
| Citeseer | 73.56 +/- 2.73 | 72.90 +/- 2.64 | -0.66 |

Forced removal：

| Remove cap | Cora Hit@10 | Citeseer Hit@10 | Removed edges | Read |
| ---: | ---: | ---: | ---: | --- |
| 1 | 66.11 | 68.00 | 60 | bad |
| 3 | 67.86 | 66.95 | 180 | bad |
| 5 | 67.67 | 67.38 | 300 | bad |
| 10 | 68.43 +/- 5.38 | 68.26 +/- 2.75 | 600 | bad |
| 20 | 67.86 +/- 6.42 | 67.47 +/- 3.39 | 1200 | bad |
| 50 | 67.17 +/- 5.60 | 67.16 +/- 5.31 | 3000 | bad |

Decision：預設關閉 removal。

Interpretation：

- Removal 不是只因為 inactive 才沒幫助；真的刪到邊時，AUROC/AP/Hit@10 都受傷。
- Citeseer 對 removal 特別敏感。

Source：

- `results/random_two_decoder_capped_remove_diag_report.md`
- `results/random_two_decoder_capped_remove_mr*_20260510_summary.csv`
- `results/random_two_decoder_relaxed_remove_mr*_20260514_summary.csv`

## 8. Push / Repulsion

嘗試：

- Full-epoch noncompact/noise push，不再只看 smoke rows。
- 比較 CP pull baseline、C0p pull only、weak push、mid push。

結果：

| Variant | Push strength | Cora Hit@10 | Citeseer Hit@10 | Decision |
| --- | --- | ---: | ---: | --- |
| `baseline_cp_pull` | none | 74.45 +/- 2.88 | 76.70 +/- 2.16 | local baseline |
| `c0p_pull_only` | none | 74.38 +/- 1.19 | 74.80 +/- 0.83 | worse on Citeseer |
| `c0p_pull_push_weak` | noncompact 0.05, noise 0.02 | 74.64 +/- 3.51 | 75.82 +/- 5.57 | mixed |
| `c0p_pull_push_mid` | noncompact 0.10, noise 0.05 | 73.43 +/- 2.95 | 76.92 +/- 1.72 | mixed |

Decision：push 不升級成主方法。

Interpretation：

- Push 有真的作用：Cora 約 push `528` 個 noncompact nodes / `36` 個 noise nodes；Citeseer 約 `646` / `68`。
- 但 final Hit@10 不穩：mid push 救 Citeseer 一點，傷 Cora。

Source：

- `results/random_two_decoder_pull_push_baseline_cp_pull_20260531_summary.csv`
- `results/random_two_decoder_pull_push_c0p_pull_only_20260531_summary.csv`
- `results/random_two_decoder_pull_push_c0p_pull_push_weak_20260531_summary.csv`
- `results/random_two_decoder_pull_push_c0p_pull_push_mid_20260531_summary.csv`

## 9. Push + Removal Interaction

目前狀態：

- 尚未看到已完成的 `push + removal` summary。
- 依既有 evidence，這不應該當成主線：
  - push = mixed；
  - removal = clearly negative；
  - 兩者相加的合理期待是「可能學到交互作用」，不是「高機率提升」。

Recommendation：

- 不建議直接做大 grid。
- 如果要回答 reviewer / internal question，可以跑一個小型 diagnostic。

最小 diagnostic 設計：

| Cell | Pull / Push | Removal | Purpose |
| --- | --- | --- | --- |
| A | `baseline_cp_pull` | conservative removal | removal under latest no-push baseline |
| B | `c0p_pull_push_weak` | same conservative removal | weak push 是否緩和 removal damage |
| C | `c0p_pull_push_mid` | same conservative removal | mid push 是否緩和 removal damage |

建議先用：

- datasets：Cora, Citeseer
- seeds：`0,1,2`
- epochs：700，或先 fast-mode sanity
- `decoded_add_ratio=0.20`
- `decoded_graph_aug_bound=0.10`
- CP intra-cluster target-1 guarantee repair 保持開啟
- removal 先用 very conservative setting，例如：
  - `decoded_remove_ratio=0.002` 或 `0.005`
  - `decoded_max_remove_per_round=1` 或 `10`
  - `decoded_degree_floor=0`

判斷標準：

- 只有在 Cora/Citeseer 都不低於 no-removal baseline，且 removed edges 真的 > 0，才值得擴 seed。
- 若 Citeseer 再次掉，直接關閉這條路。

## 10. Backbone

嘗試：

- 將整個 ARON autoencoder encoder/backbone 從 VGNAE 換成 MaskGAE / CIMAGE-full。
- 不是只換 editor backbone；embeddings 同時供 reconstruction/contrastive、GMM CP/C0p、edit decoder、prediction head 使用。

結果：

| Backbone | Dataset | Seeds | AUROC | AP | Hit@10 | Decision |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| VGNAE CP target-1 repair | Cora | 5 | 95.99 | 96.40 | 71.92 +/- 5.61 | keep |
| VGNAE CP target-1 repair | Citeseer | 5 | 96.43 | 96.96 | 76.79 +/- 2.18 | keep |
| MaskGAE backbone | Cora | 5 | 94.46 | 95.41 | 67.59 +/- 2.60 | reject as integrated backbone |
| MaskGAE backbone | Citeseer | 5 | 94.80 | 95.72 | 71.52 +/- 3.62 | reject as integrated backbone |
| CIMAGE-full backbone | Cora | 3 | 91.91 | 92.79 | 60.22 +/- 8.68 | reject |
| CIMAGE-full backbone | Citeseer | 3 | 89.54 | 92.32 | 63.22 +/- 3.74 | reject |

Decision：主線保留 VGNAE。

Interpretation：

- Official MaskGAE standalone 很強，但 integrated MaskGAE backbone 在我們 pipeline 裡變弱，代表問題是 alignment，不是 MaskGAE 本身弱。
- CIMAGE 的 factor/cluster objective 目前與 GMM CP/C0p editor 不對齊。

Source：

- `post_0521_experiment_report.md`
- `results/random_two_decoder_maskgae_backbone_*_summary.csv`
- `results/random_two_decoder_cimage_full_backbone_*_summary.csv`

## 11. Scalar Tuning

嘗試：

- 調 `prediction_bce_weight`。
- 調 `prediction_encoder_weight`。

10-seed confirmation：

| Dataset | Variant | Setting | Hit@10 | Delta | Decision |
| --- | --- | --- | ---: | ---: | --- |
| Cora | current | default | 70.49 +/- 5.11 | +0.00 | keep |
| Cora | `bce005` | `prediction_bce_weight=0.05` | 70.17 +/- 4.34 | -0.32 | reject |
| Cora | `enc010` | `prediction_encoder_weight=0.10` | 71.14 +/- 2.71 | +0.65 | Cora-only, not shared |
| Citeseer | current | default | 73.56 +/- 2.73 | +0.00 | keep |
| Citeseer | `bce005` | `prediction_bce_weight=0.05` | 73.69 +/- 3.91 | +0.13 | too small / noisy |

Decision：保留 default scalar weights。

Interpretation：

- `bce005` 不是 cross-dataset upgrade。
- `enc010` 是 Cora-only 小訊號，且 paired wins 不夠穩。

Source：

- `results/random_two_decoder_tune_confirm10_report.md`

## 12. Protocol-Level Baselines

這不是 component ablation，但會影響論文敘事。

ARON no-leak official comparison：

| Method | Cora Hit@10 | Citeseer Hit@10 | Read |
| --- | ---: | ---: | --- |
| Ours / ReverseGNN | 74.70 +/- 2.78 | 75.31 +/- 1.11 | competitive |
| Official CIMAGE | 47.00 +/- 2.86 | 61.98 +/- 3.96 | much weaker under no-leak |
| Official MaskGAE Edge | 75.71 +/- 4.68 | 78.24 +/- 0.95 | strongest official no-leak baseline |

Decision：

- 主 claim 使用 ARON no-leak。
- Full-graph / CIMAGE public-code default 只能當 protocol diagnostic。

Source：

- `post_0531_protocol_comparison_report.md`
- `/home/retro/official_baselines/runs/20260528_aron_split/summary.md`

## Open Ablation TODO

1. `push + removal`：只做小型 diagnostic，不進主線。
2. Current full method on HeaRT samples：Cora 10-seed 已完成；Citeseer 5/10 done，剩 seeds `5-9` 完成後補正式 summary。
3. Runtime ablation：初步 wall-clock audit 已整理在 `runtime_speed_audit_20260603_zh-TW.md`；仍需記錄 per-component runtime，尤其 pairwise decoders、repair pass、evaluation。
4. Candidate-only scoring：測 edit/pred decoder 是否能避免 full all-pairs scoring。
5. Repair correctness：保存 edited adjacency snapshots，用真實 edited graph 重算 after-edit orphan table。
