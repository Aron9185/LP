# Ablation Missing Check

日期：2026-06-02

這份文件只回答一件事：目前 component ablation 還缺哪些實驗，哪些只是 partial，哪些已經被後續實驗取代。

盤點來源：

- `experiments/reversegnn-compactness/run_*.sh`
- `experiments/reversegnn-compactness/results/*_summary.csv`
- `experiments/reversegnn-compactness/results/*_runs.csv`
- `post_*.md`
- `component_ablation_ledger_20260602_zh-TW.md`

## 結論

目前主要 component ablation 大多已經有結果。真正缺的不是「基本 ablation 沒跑」，而是幾個後續確認與交互作用：

1. Current full method on HeaRT samples：缺 matched 10-seed result。
2. Push + removal interaction：完全沒有 summary。
3. Repair correctness / true edited-graph after-edit orphan table：缺真實 saved edited adjacency snapshots。
4. Runtime / candidate-only scoring：已有初步 wall-clock audit；仍缺 per-component profiling 與 speed ablation。
5. CIMAGE / MaskGAE integration explanation：缺 focused cluster/factor/objective ablation。
6. VGNAE no-editor baseline：目前只有 CIMAGE no-editor controls，VGNAE no-editor launcher 已準備但尚未跑。
7. BUDDY / NCNC random no-leak comparison：目前只有 HeaRT reference baseline，沒有 ARON random split local wrapper/result。

## 已完成的核心 ablations

| Component | Status | Evidence |
| --- | --- | --- |
| Two-decoder architecture | completed | `random_fair10_report.md` |
| Prediction scorer dot vs `pair_residual_struct` | completed | `post_0430_commit_experiment_report.md` |
| Edit decoder：`pair_mlp_struct` / `mlp_pair` / `bilinear` | completed | `random_two_decoder_decoder_ablation_report.md` |
| Additions / compactness removal | completed | 5-seed screen + 10-seed `add000_compact000` confirmation |
| Endpoint rule / per-node cap | completed | `random_two_decoder_pull_endpoint_*_summary.csv` |
| Degree-target total-degree `2/3/4` | completed | `random_two_decoder_degree_target_*_summary.csv` |
| Intra-cluster target-1 large budget | completed | `random_two_decoder_intra_target1_large_budget_*_summary.csv` |
| CP intra-cluster target-1 repair | completed | `random_two_decoder_cp_target1_repair_*_summary.csv` |
| Removal normal / forced / relaxed | completed | `random_two_decoder_capped_remove_*`, `random_two_decoder_relaxed_remove_*` |
| Push / repulsion weak and mid | completed | `random_two_decoder_pull_push_*_summary.csv` |
| Integrated MaskGAE backbone | completed | `random_two_decoder_maskgae_backbone_*_summary.csv` |
| Integrated CIMAGE-full backbone | completed after Cora seed-0 rerun | full summary plus `cora_s0_rerun` |
| CIMAGE no-editor dot/pred controls | completed, not yet emphasized in ledger | `cimage_no_editor_dot/pred_20260528_summary.csv` |
| Scalar tuning | completed | `random_two_decoder_tune_confirm10_report.md` |
| Official CIMAGE / MaskGAE ARON no-leak | completed | `/home/retro/official_baselines/runs/20260528_aron_split/summary.md` |

## Missing / not found

### M1. Current Full Method On HeaRT Samples

Status：Cora completed, Citeseer partial; full 20-run summary pending。

Started run：

- tmux session：`heart_current_cp_target1_20260602`
- launcher：`run_heart_current_cp_target1_repair_20260602.sh`
- prefix：`heart_current_cp_target1_repair_20260602`
- planned rows：Cora/Citeseer x seeds `0-9` = 20 runs
- log：`results/heart_current_cp_target1_repair_20260602_tmux.log`
- expected outputs：
  - `results/heart_current_cp_target1_repair_20260602_runs.csv`
  - `results/heart_current_cp_target1_repair_20260602_summary.csv`

現有結果：

- 有早期 HeaRT two-decoder editor 3-seed result。
- 有 `heart_decoder_scorer_v1` 10-seed result，但它不是目前 CP target-1 repair full method。
- 目前完整 CP target-1 repair / latest baseline 的 HeaRT matched run 已在跑。
- Cora seeds `0-9` 已完成：Hit@10 `36.60 +/- 1.60`。
- Citeseer seeds `0-4` 已完成：Hit@10 `51.47 +/- 1.27`；seed `5` running，seeds `6-9` pending。

Why it matters：

- 如果要對 HeaRT setting 做正式 claim，這是最重要缺口。
- Cora interim/full result 已接近 NCNC baseline；Citeseer partial result 已高於 BUDDY，但 full 10-seed 結果還沒完成。

Suggested minimum：

- Cora/Citeseer, seeds `0-9`
- HeaRT samples split
- Current full method config：VGNAE + `pair_mlp_struct` edit decoder + `pair_residual_struct` prediction head + CP intra-cluster target-1 repair + removal off

Priority：P0 if HeaRT claim is needed; otherwise P1.

### M2. Push + Removal Interaction

Status：missing。

Evidence：

- `random_two_decoder_pull_push_*` 都是 `decoded_remove_ratio=0.0`。
- No result file matching `*push*remove*` or `*remove*push*`.

Why it matters：

- User question / possible reviewer question: push may change geometry enough to make removal less harmful.
- But prior evidence says removal is negative and push is mixed, so this is diagnostic, not main path.

Suggested minimum：

| Cell | Pull / Push | Removal | Purpose |
| --- | --- | --- | --- |
| A | `baseline_cp_pull` | conservative removal | latest no-push removal control |
| B | `c0p_pull_push_weak` | same removal | weak push interaction |
| C | `c0p_pull_push_mid` | same removal | mid push interaction |

Use seeds `0,1,2`, Cora/Citeseer, and require `removed_edges_total > 0`.

Priority：P1 diagnostic; do not run a large grid first.

### M3. True Edited-Graph Correctness / After-Edit Orphan Table

Status：missing.

Evidence：

- Existing orphan diagnostics include original-graph and reconstructed repair comparison.
- Historical CP target-1 repair used `--decoded_temporary_view_only`; exact edited adjacency snapshots were not saved.
- No artifact found matching edited adjacency snapshots / true after-edit orphan table.

Missing pieces：

- Save edited adjacency snapshots during training.
- Recompute CP/C0p intra-orphan table on the actual edited graph.
- Add unrepaired reason counters.
- Unit tests for guarantee repair:
  - endpoint constraint blocks repair?
  - deficit-first greedy starves nodes?
  - repair-only CP-to-CP relaxation helps?

Priority：P0/P1 for correctness story.

### M4. Runtime / Speed Ablations

Status：partially filled; profiling / optimization ablation still missing.

Evidence：

- 已整理初步 wall-clock audit：`runtime_speed_audit_20260603_zh-TW.md`。
- Training logs print total training time, but no per-component runtime summary table was found.
- No result files matching runtime/profile/candidate-only scoring.

Current wall-clock read：

- Current random no-leak CP target-1 repair `addr020`：Cora `94.55 +/- 2.29` min/seed，Citeseer `91.95 +/- 1.61` min/seed。
- Current HeaRT full-method run：Cora seeds `0-2` completed，平均 `98.15 +/- 8.34` min/seed；20-run serial job 粗估 `31-34h`。
- Old HeaRT two-decoder：Cora/Citeseer 約 `29-30` min/seed，所以 current full method 約慢 `3.3x`。
- CIMAGE no-editor controls：dot 約 `1-2` min/seed，pred decoder 約 `27-28` min/seed。

Missing pieces：

- Per-epoch runtime.
- Encoder vs edit decoder vs prediction decoder vs GMM/CP/C0p update vs repair pass vs evaluation timing.
- `mlp_pair_max_rows` sweep.
- Candidate-only scoring ablation.
- Evaluation-frequency / diagnostics-frequency ablation.

Priority：P1 because current method is slow.

### M5. CIMAGE / MaskGAE Integration Explanation

Status：missing / optional.

Completed：

- Integrated MaskGAE backbone full result.
- Integrated CIMAGE-full backbone full result.
- CIMAGE no-editor dot/pred controls.
- Official MaskGAE standalone and CIMAGE standalone comparisons.

Still missing if we want to explain the integration failure：

- CIMAGE `cluster_weight=0` while keeping factor reconstruction.
- CIMAGE factor-only / cluster-only ablation.
- CIMAGE pseudo-label threshold / number of clusters ablation.
- MaskGAE integrated objective ablation:
  - masked feature loss on/off
  - mask rate
  - with/without ARON contrastive objective

Priority：P2 unless the paper needs a detailed explanation of why standalone MaskGAE is strong but integrated MaskGAE is weaker.

## Partial / needs caution

### P1. Latest Pull/Push Is Only 3 Seeds

Status：partial if we want to make a robust claim.

Current：

- `baseline_cp_pull`, `c0p_pull_only`, `weak`, `mid` are all 3 seeds.

Read：

- Good enough for diagnostic.
- Not enough for a final method claim.

Priority：expand only if push/pull becomes part of the final method. Current conclusion is mixed, so expansion is not urgent.

### P2. CP Target-1 Repair Main Result Is 5 Seeds

Status：partial if we need matched seed count.

Current：

- CP target-1 repair main result is 5 seeds.
- Official CIMAGE/MaskGAE no-leak comparison is 3 seeds.
- Older two-decoder fair comparison is 10 seeds.

Read：

- For internal decision, 5 seeds are enough.
- For final paper table, we should decide whether all main methods need matched seed counts.

Priority：P0/P1 depending on paper table requirements.

### P3. CIMAGE-Lite Backbone

Status：partial.

Current：

- Only seed `0` initial result for Cora/Citeseer.
- No full multi-seed CIMAGE-lite backbone result found.

Read：

- Low priority because MaskGAE and CIMAGE-full already show integrated backbone replacement is not the main path.

Priority：P3.

### P4. HeaRT Hard-Remove Initial Runs Had Parse Failures, But Rescue Covered Cora

Status：covered for interpretation, but initial run files contain failed Cora rows.

Current：

- Initial `heart_two_decoder_hard_remove_*_20260501_runs.csv` has parse failures for several Cora rows.
- `20260502_cora_rescue` reruns are OK.

Read：

- Do not use the failed initial rows directly.
- Use combined/curated interpretation from the reports.

Priority：no action unless rebuilding HeaRT tables from raw files.

### P5. CIMAGE-Full Cora Seed-0 Initial Crash

Status：covered.

Current：

- Initial full CIMAGE run has Cora seed-0 parse failure.
- Cora seed-0 rerun completed successfully.

Read：

- Use the rerun-adjusted CIMAGE-full summary from reports.

Priority：no action.

## Superseded / not worth filling unless historical completeness matters

### S1. Intra-Degree Budget Grid Target-2

Status：planned but no results found; superseded.

Planned launcher：

- `run_random_two_decoder_intra_degree_budget_grid_20260518.sh`

Planned variants：

- `intra_dtarget2_addr001`
- `intra_dtarget2_addr003`
- `intra_dtarget2_addr005`

No summary found matching：

- `random_two_decoder_intra_degree_budget_*`
- `*dtarget2_addr*`

Why superseded：

- Later target-1 large-budget and CP target-1 repair directly answered the more relevant question: repair full CP intra-cluster degree deficits.

Priority：do not run unless we need a complete historical trail.

### S2. Integrated Full-Graph Leakage Backbones

Status：scripts exist, no ARON result summaries found for integrated MaskGAE/CIMAGE-full full-graph leakage.

Scripts：

- `run_random_two_decoder_fullgraph_leakage_maskgae_20260529.sh`
- `run_random_two_decoder_fullgraph_leakage_cimage_full_20260529.sh`

Why likely not needed：

- Official full-graph leakage baselines already demonstrate the protocol issue.
- Main claim is ARON no-leak, not integrated-backbone full-graph leakage.

Priority：do not run unless the protocol appendix specifically needs integrated-backbone leaky rows.

## Result Exists But Ledger Could Be Improved

These are not missing experiments; they are missing organization details:

1. Add `cimage_no_editor_dot/pred` controls into the component ledger.
2. Mark `intra_degree_budget_grid` as superseded in the ledger.
3. Mark HeaRT hard-remove rescue and CIMAGE-full seed-0 rerun as raw-file caveats.
4. Separate "diagnostic-only" ablations from "candidate method" ablations in any paper table.

## Recommended Next Action

If we want the smallest useful next batch:

1. Add runtime instrumentation / profiling first, because current method is slow.
2. Run `push + removal` only as a 3-cell diagnostic if we still care about that interaction.
3. Run current full method on HeaRT samples only if we need a formal HeaRT-setting claim.
4. Add true edited-graph correctness snapshots before making a strong structural repair claim.
