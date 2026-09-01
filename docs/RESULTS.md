# TwinLine Backtest Results

Simulated data — see Home page for assumptions.

## Ablation table

| method | PR-AUC | precision@budget | recall@budget | mean lead time (s) | rework avoided | investigation cost | net benefit |
|---|---|---|---|---|---|---|---|
| rules-only | 0.048 | 0.000 | 0.000 | nan | 0 | 200 | -200 |
| ML-only | 0.058 | 0.200 | 0.043 | 561 | 250 | 160 | 90 |
| hybrid | 0.058 | 0.200 | 0.043 | 561 | 250 | 160 | 90 |
| hybrid+soft-sensors | 0.058 | 0.200 | 0.043 | 561 | 250 | 160 | 90 |

## Scorecard by instrumentation level

| instrumentation_level | n | precision | recall | false_alarms | mean_lead_time_s | calibration_error | abstention_rate |
|---|---|---|---|---|---|---|---|
| overall | 46080 | 0.065 | 0.089 | 2325 | 1629.199 | 0.148 | 0.000 |
| manual | 9600 | 0.880 | 0.116 | 6 | 1501.169 | 0.015 | 0.000 |
| partial | 9600 | 0.038 | 0.237 | 2291 | 1679.600 | 0.304 | 0.000 |
| rich | 26880 | 0.500 | 0.026 | 28 | 1668.388 | 0.150 | 0.000 |
