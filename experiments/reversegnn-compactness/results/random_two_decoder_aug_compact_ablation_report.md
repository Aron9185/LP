# Random-Split Augmentation and Compactness Ablation

## Summary

This ablation tests whether the current gain depends on decoded edge additions or the compactness objective, with ROC-AUC/AP/Hit@10 reported together.

Result: the strongest 5-seed candidate is now `add000_compact000`, which disables both decoded additions and compactness. Cora still has a strong 10-seed current baseline, so this should be confirmed before changing the default.

## Setup

- Datasets: Cora, Citeseer
- Seeds: `0-4`
- Split: random
- Config: `two_decoder_pred`
- Edit decoder: `pair_mlp_struct`
- Prediction head: `pair_residual_struct`
- Remove ratio: `0.0`
- Baseline: `random_two_decoder_editor_20260501`, seeds `0-4`

Compared variants:

- `add000`: `decoded_add_ratio=0.0`
- `compact000`: `compactness_weight=0.0`
- `add000_compact000`: `decoded_add_ratio=0.0`, `compactness_weight=0.0`
- `compact020_frozen`: `compactness_weight=0.2` with frozen C0p targets
- `compact100_dynamic`: `compactness_weight=1.0` with dynamic C0p targets
- `compact100_frozen`: `compactness_weight=1.0` with frozen C0p targets

## Results

Values are percentages over seeds `0-4`. Delta is Hit@10 minus the current baseline on the same seeds.

| Dataset | Variant | Add Ratio | Compact W | C0p Targets | Remove Ratio | ROC-AUC | AP | Hit@10 | Delta | Added Edges | Radius Delta | C0p Radius Delta |
|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Cora | current | 0.01 | 0.2 | dynamic | 0.0 | 95.93 +/- 0.43 | 96.19 +/- 0.53 | 69.15 +/- 7.07 | +0.00 | 2700 | -0.000051 | -0.000011 |
| Cora | `add000` | 0.00 | 0.2 | dynamic | 0.0 | 95.97 +/- 0.52 | 96.20 +/- 0.58 | 69.49 +/- 7.31 | +0.34 | 0 | +0.000000 | +0.000000 |
| Cora | `compact000` | 0.01 | 0.0 | dynamic | 0.0 | 96.00 +/- 0.45 | 96.23 +/- 0.50 | 69.30 +/- 6.91 | +0.15 | 2700 | -0.000037 | -0.000028 |
| Cora | `add000_compact000` | 0.00 | 0.0 | dynamic | 0.0 | 95.90 +/- 0.52 | 96.20 +/- 0.51 | 70.13 +/- 4.84 | +0.98 | 0 | +0.000000 | +0.000000 |
| Cora | `compact020_frozen` | 0.01 | 0.2 | frozen | 0.0 | 95.74 +/- 0.44 | 96.07 +/- 0.51 | 67.59 +/- 7.38 | -1.56 | 2745 | -0.000240 | -0.000285 |
| Cora | `compact100_dynamic` | 0.01 | 1.0 | dynamic | 0.0 | 95.98 +/- 0.45 | 96.18 +/- 0.55 | 69.91 +/- 7.69 | +0.76 | 2700 | -0.000030 | -0.000012 |
| Cora | `compact100_frozen` | 0.01 | 1.0 | frozen | 0.0 | 95.67 +/- 0.50 | 96.03 +/- 0.52 | 68.27 +/- 7.46 | -0.87 | 2745 | -0.000235 | -0.000302 |
| Citeseer | current | 0.01 | 0.2 | dynamic | 0.0 | 96.15 +/- 0.51 | 96.60 +/- 0.40 | 73.85 +/- 2.74 | +0.00 | 2340 | -0.000025 | +0.000000 |
| Citeseer | `add000` | 0.00 | 0.2 | dynamic | 0.0 | 96.15 +/- 0.48 | 96.59 +/- 0.40 | 74.20 +/- 2.97 | +0.35 | 0 | +0.000000 | +0.000000 |
| Citeseer | `compact000` | 0.01 | 0.0 | dynamic | 0.0 | 96.31 +/- 0.46 | 96.73 +/- 0.32 | 72.92 +/- 3.55 | -0.92 | 2340 | -0.000111 | -0.000031 |
| Citeseer | `add000_compact000` | 0.00 | 0.0 | dynamic | 0.0 | 96.33 +/- 0.43 | 96.74 +/- 0.32 | 74.29 +/- 2.99 | +0.44 | 0 | +0.000000 | +0.000000 |
| Citeseer | `compact020_frozen` | 0.01 | 0.2 | frozen | 0.0 | 95.72 +/- 0.58 | 96.27 +/- 0.38 | 71.69 +/- 2.18 | -2.16 | 2379 | -0.000061 | -0.000010 |
| Citeseer | `compact100_dynamic` | 0.01 | 1.0 | dynamic | 0.0 | 96.22 +/- 0.45 | 96.63 +/- 0.30 | 73.98 +/- 1.66 | +0.13 | 2340 | -0.000078 | -0.000016 |
| Citeseer | `compact100_frozen` | 0.01 | 1.0 | frozen | 0.0 | 95.64 +/- 0.52 | 96.18 +/- 0.43 | 70.42 +/- 2.54 | -3.43 | 2379 | -0.000054 | -0.000005 |

## Per-Seed Hit@10

| Dataset | Seed | current | `add000` | `compact000` | `add000_compact000` | `compact020_frozen` | `compact100_dynamic` | `compact100_frozen` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Cora | 0 | 68.50 | 66.22 | 67.17 | 68.69 | 63.57 | 70.40 | 63.76 |
| Cora | 1 | 69.07 | 71.73 | 73.24 | 69.07 | 71.73 | 73.81 | 72.30 |
| Cora | 2 | 74.57 | 74.19 | 73.62 | 73.81 | 71.92 | 73.24 | 72.11 |
| Cora | 3 | 57.87 | 58.44 | 58.06 | 63.38 | 56.55 | 56.55 | 57.50 |
| Cora | 4 | 75.71 | 76.85 | 74.38 | 75.71 | 74.19 | 75.52 | 75.71 |
| Citeseer | 0 | 72.97 | 69.45 | 68.57 | 71.21 | 71.65 | 71.43 | 71.65 |
| Citeseer | 1 | 78.24 | 77.36 | 78.46 | 78.02 | 74.07 | 75.60 | 74.29 |
| Citeseer | 2 | 70.77 | 74.29 | 72.09 | 75.60 | 73.63 | 73.41 | 69.23 |
| Citeseer | 3 | 73.19 | 75.82 | 72.53 | 71.21 | 69.01 | 75.16 | 68.13 |
| Citeseer | 4 | 74.07 | 74.07 | 72.97 | 75.38 | 70.11 | 74.29 | 68.79 |

## Interpretation

The prediction head appears to carry the useful improvement more than decoded graph augmentation. On this seed slice, removing decoded additions does not hurt; disabling both additions and compactness is the best Hit@10 candidate and also the best Citeseer ROC-AUC/AP candidate.

Compactness is not clearly helpful. Stronger frozen compactness moves radius more, but that does not translate to better ROC-AUC/AP/Hit@10. Dynamic `compact100` is less harmful than frozen compactness, but still does not clearly beat the no-compact/no-add candidate.

## Recommendation

Promote `add000_compact000` to a 10-seed confirmation candidate before changing the default. If it holds, the method can be simplified: keep `pair_mlp_struct` and the structure-aware prediction head, but disable decoded edge additions and compactness.

Do not pursue stronger compactness as the next improvement path unless the objective is redesigned to align with final prediction quality and checked with ROC-AUC/AP, not only Hit@10.
