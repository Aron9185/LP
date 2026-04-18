# ARON Sweep Analysis Results

## Overview
Analyzed 432 total runs across 4 datasets.

## Best Configurations Found
   dataset  pull_strength  add_ratio  remove_ratio  test_hit10
      cora            0.6      0.020         0.002    0.665402
  citeseer            0.1      0.005         0.000    0.609524
   Cora_ML            0.0      0.050         0.000    0.517178
LastFMAsia            0.2      0.010         0.000    0.339209

## Hypothesis Validation: Compactness Correlation
Checking if smaller radii (more compactness) correlate with higher Hit@10.
> Negative correlation means smaller radius = higher performance (Hypothesis is VALID if negative)

| Dataset | Radius vs Hit@10 Correlation | Verdict |
| :--- | :--- | :--- |
| **cora** | -0.3514 | ✅ Strong Match |
| **citeseer** | 0.1259 | ❌ Inverse Correlation |
| **Cora_ML** | -0.4163 | ✅ Strong Match |
| **LastFMAsia** | 0.4508 | ❌ Inverse Correlation |
