# Random-Split Capped-Removal Diagnostic

## Summary

This diagnostic tested whether removal only looked bad because the normal remove setting barely removed edges. We forced real removals by relaxing the endpoint constraints and capping removals per rewrite round, then compared ROC-AUC/AP/Hit@10 together.

Result: actual removal is harmful under this policy. Caps `10/20/50` all lose to the no-removal baseline on seeds `0-4`, and increasing the removal budget does not recover performance.

## Setup

- Datasets: Cora, Citeseer
- Seeds: `0-4`
- Config: `two_decoder_pred_remove`
- Split: random
- Add ratio: `0.01`
- Remove ratio: `0.05`
- Removal caps per rewrite: `10`, `20`, `50`
- Constraints: `decoded_degree_floor=0`, cross-cluster allowed, no C0p endpoint requirement

## Results

Values are percentages over seeds `0-4`. Delta is capped-removal Hit@10 minus the current no-removal baseline on the same seeds.

| Remove Cap | Dataset | ROC-AUC | AP | Hit@10 | Baseline Hit@10 | Delta | Added Edges | Removed Edges | Radius Delta |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | Cora | 95.42 +/- 0.87 | 95.74 +/- 0.75 | 68.43 +/- 5.38 | 69.15 | -0.72 | 2700 | 600 | -0.000049 |
| 10 | Citeseer | 95.22 +/- 0.51 | 95.78 +/- 0.32 | 68.26 +/- 2.75 | 73.85 | -5.58 | 2340 | 600 | -0.000067 |
| 20 | Cora | 95.60 +/- 0.84 | 95.88 +/- 0.80 | 67.86 +/- 6.42 | 69.15 | -1.29 | 2700 | 1200 | -0.000271 |
| 20 | Citeseer | 95.35 +/- 0.73 | 95.77 +/- 0.60 | 67.47 +/- 3.39 | 73.85 | -6.38 | 2340 | 1200 | -0.000243 |
| 50 | Cora | 95.61 +/- 0.90 | 95.77 +/- 1.07 | 67.17 +/- 5.60 | 69.15 | -1.98 | 2700 | 3000 | -0.000889 |
| 50 | Citeseer | 95.12 +/- 0.54 | 95.68 +/- 0.40 | 67.16 +/- 5.31 | 73.85 | -6.69 | 2340 | 3000 | -0.000427 |

## Conclusion

Removal should not be part of the next default method. The normal remove variant was not simply inactive; when removals are forced to happen, ROC-AUC/AP/Hit@10 all trail the no-removal baseline.

The next improvement path should keep old-safe removal defaults: no relaxed removal by default, no C0p max-min-degree guarantee, and no forced cross-cluster/no-C0p removal unless explicitly requested.

Source artifacts:

- `random_two_decoder_capped_remove_mr10_20260510_summary.csv`
- `random_two_decoder_relaxed_remove_mr20_20260514_summary.csv`
- `random_two_decoder_relaxed_remove_mr50_20260514_summary.csv`
