# Runtime / Speed Audit

日期：2026-06-03

這份文件整理目前 ReverseGNN / ARON 主方法的訓練時間，並和內部控制組、舊版方法、官方 baseline wrapper 做速度對照。數字主要來自 log 裡的 `Total training time`；官方 baseline 則使用 `progress.log` 或 tmux log 的 START/END wall-clock。

## 短結論

- 目前主方法不是「稍微慢」，而是落在每個 seed 約 `90-100` 分鐘的量級。
- Random no-leak current full method `CP target-1 repair addr020`：Cora `94.55 +/- 2.29` 分鐘/seed，Citeseer `91.95 +/- 1.61` 分鐘/seed。
- 目前正在跑的 HeaRT current full method：Cora seeds `0-9` 已完成，平均 `94.13 +/- 5.19` 分鐘/seed；Citeseer seeds `0-4` 已完成，平均 `89.44 +/- 1.68` 分鐘/seed。
- 舊版 HeaRT two-decoder 約 `29-30` 分鐘/seed，所以 current full method 大約慢 `3.3x`。
- 在我們 runner 裡，no-editor dot 只要 `1-2` 分鐘/seed；no-editor `pair_residual_struct` prediction head 約 `27-28` 分鐘/seed。這表示 `pair_residual_struct` 本身就是大成本，current editor/repair/eval flow 又再把時間推到 `90+` 分鐘。
- Official CIMAGE wrapper 約 `1.1-1.3` 分鐘/seed；official MaskGAE wrapper 在現有 progress log 是秒級。這些和我們 current method 不是 apples-to-apples，但可以清楚說明目前 ARON full method 的工程成本高很多。

## 2026-06-04 已實作的第一波加速改動

- 新增 `--train_eval_every`：random split 不必每個 epoch 都跑 full validation/test；預設 `1`，所以舊 run 行為不變。
- 新增 `--skip_train_acc`：可跳過每個 epoch 的 full-matrix train accuracy。這會避免非 eval epoch 還為了 `train_acc` 產生完整 `A_pred`。
- 新增 `--decoder_diag_every`：`-1` 維持每個 evaluated epoch 做 diagnostics，`0` 可關掉 training-time decoder diagnostics，只保留 final diagnostics。
- 新增 `--edit_metric_every`：可降低 editor compactness/radius diagnostics 的頻率，避免每個 epoch 都 build decoded graph / re-encode。
- HeaRT runner 現在可調 `--heart-eval-every`、`--heart-val-frac`、`--eval-log-every`，不再硬寫成 full validation every 5 epochs。
- no-editor backbone runner 也能傳 `train_eval_every`、`skip_train_acc`、`decoder_diag_every`、HeaRT eval fraction/interval。
- 已新增兩個 fast-screen launcher，但尚未啟動：
  - `run_vgnae_pred_only_heart_fast_20260604.sh`：VGNAE + `pair_residual_struct` prediction-only 單 decoder，HeaRT fast screen。
  - `run_heart_current_cp_target1_fast_20260604.sh`：目前 full editor 方法的降頻版 fast screen。
- 新增 `--edge_eval` / `--full_matrix_eval`：validation/test/final eval 可直接對 positive/negative edges scoring，不再為少數 eval edges 產生完整 `N x N` matrix。`--edge_eval` 是目前預設；若要回舊路徑可顯式用 `--full_matrix_eval`。
- Final decoder diagnostics 也改為 edge-only：不再為 final diagnostics 同時建立 dot、edit decoder、prediction decoder 三張 full matrix。
- `decoded_temporary_view_only + ver=no` 時，edit decoder 的 graph structural context 不再每 epoch 重算 CN/RA/AA；初始 train graph context 已經相同，訓練中只更新 labels/C0p mask。
- 新增 `--decoded_rewrite_every`：正式預設仍是 `1`（每 epoch rewrite）；fast screen launcher 先用 `5`，測試降低 graph rewrite/candidate scoring 頻率是否仍保留表現。

這一波改動保留正式結果的預設 protocol；只有顯式加上 fast flags 時才會少做 training-time eval/diagnostics。

## 目前主方法訓練時間

| Experiment | Split / Protocol | Seeds | Cora min/seed | Citeseer min/seed | 解讀 |
| --- | --- | ---: | ---: | ---: | --- |
| `CP target-1 repair addr020` | ARON random no-leak | 5 | `94.55 +/- 2.29` | `91.95 +/- 1.61` | 目前 random split 主方法；10 runs 合計約 `15.5h` training time |
| `CP target-1 repair addr010` | ARON random no-leak | 5 | `71.78 +/- 1.48` | `89.53 +/- 1.41` | 較低 add ratio 對 Cora 明顯快；可當 speed/quality tradeoff hint，不是目前主結果 |
| `baseline_cp_pull` | ARON random no-leak | 3 | `96.61 +/- 2.60` | `93.76 +/- 1.45` | Latest pull/push grid 的 no-push baseline |
| `c0p_pull_only` | ARON random no-leak | 3 | `94.53 +/- 2.10` | `93.04 +/- 1.23` | 和 current full method 同量級 |
| `c0p_pull_push_weak` | ARON random no-leak | 3 | `95.34 +/- 1.92` | `91.47 +/- 0.69` | Push 沒有顯著降低 runtime |
| `c0p_pull_push_mid` | ARON random no-leak | 3 | `94.60 +/- 2.32` | `91.35 +/- 0.85` | Push 沒有顯著降低 runtime |
| `heart_current_cp_target1_repair_20260602` | HeaRT samples | Cora 10 / Citeseer 5 done | `94.13 +/- 5.19` | `89.44 +/- 1.68` | 正式 20-run job still running；Citeseer seed `5` running |

## 內部控制組與舊版方法

| Experiment | Seeds | Cora min/seed | Citeseer min/seed | 解讀 |
| --- | ---: | ---: | ---: | --- |
| `cimage_no_editor_dot_20260528` | 3 | `1.33 +/- 0.14` | `1.88 +/- 0.04` | 便宜 dot scorer 的下限 |
| `cimage_no_editor_pred_20260528` | 3 | `27.04 +/- 0.24` | `28.27 +/- 0.34` | 只換成 `pair_residual_struct` prediction head 就變成 27-28 分鐘級 |
| `heart_two_decoder_editor_20260430 two_decoder_pred` | 3 | `29.48 +/- 0.16` | `29.86 +/- 0.16` | 舊版 HeaRT two-decoder；比 current full method 快約 `3.3x` |
| Integrated MaskGAE backbone current | 5 | `99.38 +/- 2.72` | `92.86 +/- 2.11` | 換 MaskGAE backbone 沒有讓 current ARON pipeline 變快 |
| Integrated CIMAGE-full backbone current | Cora 2 / Citeseer 3 | `96.97 +/- 1.43` | `97.55 +/- 2.26` | 換 CIMAGE backbone 也仍是 90+ 分鐘級 |

## Official Baseline Wrapper Wall-Clock

這些是 official baseline repo / wrapper 的 wall-clock，不是同一個 training objective，也不是同一個 implementation complexity。可用來說明速度量級，但不應當成完全公平 runtime benchmark。

| Run | Dataset / Method | Wall-clock | Seeds | Approx min/seed |
| --- | --- | ---: | ---: | ---: |
| `20260528_aron_split` | CIMAGE Cora | `3m55s` | 3 | `1.31` |
| `20260528_aron_split` | CIMAGE Citeseer | `3m23s` | 3 | `1.13` |
| `20260528_aron_split` | MaskGAE Path Cora | `17s` | 3 | `0.09` |
| `20260528_aron_split` | MaskGAE Edge Cora | `17s` | 3 | `0.09` |
| `20260528_aron_split` | MaskGAE Path Citeseer | `18s` | 3 | `0.10` |
| `20260528_aron_split` | MaskGAE Edge Citeseer | `17s` | 3 | `0.09` |
| `20260531_cimage_paper_multiseed_fixed` | CIMAGE Cora paper protocol | `3m58s` | 3 | `1.32` |
| `20260531_cimage_paper_multiseed_fixed` | CIMAGE Citeseer paper protocol | `3m26s` | 3 | `1.14` |

## 為什麼 Current Method 慢

1. Current launcher 使用 `epochs=700`、`max_workers=1`、`mlp_pair_max_rows=16`，偏穩定與省 GPU memory，但 serial wall-clock 很長。
2. `pair_mlp_struct` edit decoder 與 `pair_residual_struct` prediction head 都是 structural pair MLP。`forward()` 會 chunked 產生 NxN score matrix，Cora/Citeseer 這種節點數下每次就是數百萬到上千萬 pair。
3. `score_pairs()` fast path 已存在，training loss 裡 sampled pairs 會優先用它；但 current flow 仍有多處需要 full matrix，例如 `_score_adjacency(Z)`、rewrite candidate scoring、decoder diagnostics、final score diagnostics。
4. `pair_residual_struct` 需要 degree、CN、RA、AA、cluster/core/CP/C0p/prototype-distance 等 structural features，比 dot product 重很多。
5. CP intra-cluster target-1 guarantee repair 會掃 candidate order，且 guarantee mode 可能超過原始 add budget/per-node cap。
6. HeaRT setting 使用 full validation / Hit@10 checkpointing；`heart_eval_every=5` 代表 700 epochs 內會重複做很多次 evaluation。
7. Dynamic GMM target、CP/C0p mask、prototype distances、decoder structural context 在 training/eval/rewrite 過程中會反覆更新。

## 加速 TODO

P0：先加 runtime instrumentation。

- 在 log 與 summary CSV 加上 total / per-epoch / per-block runtime。
- 至少拆：encoder forward/backward、edit decoder objective、prediction decoder objective、GMM/CP/C0p target update、decoded graph rewrite/repair、validation/evaluation、final diagnostics。
- 目標是先知道 90 分鐘裡每塊占比，而不是靠猜。

P0：減少 training-time full NxN scoring。

- 保留 final evaluation metric，但能用 edge-only / candidate-only 的地方就接 `score_pairs()`。
- `_score_adjacency(Z)` 現在在 `score_source=pred_decoder` 時會呼叫完整 `prediction_decoder(z)`；要把 validation / diagnostics 改成只對 val/test positive/negative edge pairs score。（2026-06-04 已完成 edge-only validation/test/final eval；若沒開 `--skip_train_acc`，仍會為 train_acc 產生 full matrix。）
- Rewrite/add candidate selection 只對 same-cluster、CP/C0p-related、top-k candidate pairs 打分，不再每次 full all-pairs。

P1：調 evaluation / diagnostics 頻率。

- HeaRT diagnostic run 先試 `heart_eval_every=10/20`、`eval_log_every=10/20`。（2026-06-04 已加 flags 與 launcher，待 GPU 空檔實測。）
- Full decoder diagnostics 只在 best checkpoint 或 final 做。（2026-06-04 已支援 `decoder_diag_every=0` 關掉 training-time diagnostics。）
- 保留正式結果的完整 final eval；fast mode 只用於 ablation screening。

P1：做 `mlp_pair_max_rows` 與 memory sweep。

- Smoke grid：`16 / 32 / 64`。
- 如果 GPU memory 允許，較大 chunk 可以減少 Python loop overhead。
- 需要記錄 peak memory、sec/epoch、是否 OOM。

P1：cache 重複 structural context。

- CN / RA / AA / degree 如果 graph context 沒變，不要重算。
- Cluster labels、CP/C0p masks、prototype distances、candidate edge lists 能降頻或 cache。
- Repair candidate order 若 constraints 沒變，可重用候選集合，只更新分數。

P1：staged / fast-mode launcher。

- Phase 1 用便宜 scorer 或較低 evaluation frequency warm up。
- Phase 2 才啟用 full `pair_residual_struct` diagnostics / repair。
- Ablation screening 先跑 3 seeds / fewer epochs / lower eval frequency，正式表格再回到完整設定。（2026-06-04 已新增 first-pass fast launchers。）
- Full editor fast screen 先用 `decoded_rewrite_every=5` 測 graph rewrite 降頻；若 Hit@10 沒掉，再試 `10`。

P2：平行跑 seeds。

- 目前 `max_workers=1` 是保守設定。若 profile 顯示單 run GPU memory 足夠，可測 `max_workers=2`。
- 這只縮短 wall-clock，不降低單 run compute；適合跑 grid，但不是根本加速。

## 建議下一步

1. 先做 runtime profiler，不先改演算法。
2. 用 Cora seed 0 跑短程 smoke，測 `mlp_pair_max_rows=16/32/64` 與 `heart_eval_every=5/10/20` 的 sec/epoch。
3. 再實作 edge-only validation / candidate-only rewrite scoring，因為這最有機會把 90 分鐘級拉下來。
4. 最後才做更大的 heuristic 改動，例如 staged scorer 或 approximation。
