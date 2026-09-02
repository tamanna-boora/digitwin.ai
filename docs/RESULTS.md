# TwinLine Backtest Results

Simulated data — see Home page for assumptions.

## Ablation table

| method | PR-AUC | precision@budget | recall@budget | mean lead time (s) | rework avoided | investigation cost | net benefit |
|---|---|---|---|---|---|---|---|
| rules-only | 0.074 | 0.000 | 0.000 | nan | 0 | 200 | -200 |
| ML-only | 0.058 | 0.200 | 0.043 | 561 | 250 | 160 | 90 |
| hybrid | 0.071 | 0.000 | 0.000 | nan | 0 | 200 | -200 |
| hybrid+soft-sensors | 0.071 | 0.000 | 0.000 | nan | 0 | 200 | -200 |

## Finding: the ablation is a negative result

Only **ML-only** is net-positive at the current alarm budget (+90 currency in the backtest window). **rules-only, hybrid, and hybrid+soft-sensors are all net-negative (-200)** — investigation cost on false alarms outweighs rework avoided for every method that includes the rules layer.

**Global ranking and top-K selection are decoupled here.** rules-only has the *best* PR-AUC of the four methods (0.074, vs. ML-only's 0.058) and simultaneously the *worst* top-K precision (0.000). A score can be well-ordered overall and still pick the wrong 5-per-shift.

We found the rules score saturating at 1.0 on every unit — a worst-of-16-stations aggregation with a ±1200s match tolerance was matching 99.8% of the entire run timeline, so every unit hit a "critical" signal somewhere regardless of its own path. We fixed that: matching now requires the unit's visit to fall inside the signal's own real aggregation window (not a wide symmetric tolerance), and aggregation across stations is a mean instead of a max, so the score is no longer constant. The rules score is now genuinely discriminative — 191 distinct values across 1920 test units, spread 0.02–0.53, no saturation at either end.

With that fixed, **hybrid still lost.** Averaging the now-real rules score 50/50 with ML-only's calibrated probability produces a worse top-5-per-shift pick than ML-only alone, not a better one. We did not tune the blend weight to try to recover a positive result — the brief was to make the rules score discriminative, not to make hybrid win.

The rules layer stays in the product, but for **explainability, not ranking** — it's the human-readable "why" attached to an alert (which SPC/anomaly rule fired, at which station, how severe), not an input that should be blended into the score driving alarm selection at this budget.

## Scorecard by instrumentation level

| instrumentation_level | n | precision | recall | false_alarms | mean_lead_time_s | calibration_error | abstention_rate |
|---|---|---|---|---|---|---|---|
| overall | 46080 | 0.065 | 0.089 | 2325 | 1629.199 | 0.148 | 0.000 |
| manual | 9600 | 0.880 | 0.116 | 6 | 1501.169 | 0.015 | 0.000 |
| partial | 9600 | 0.038 | 0.237 | 2291 | 1679.600 | 0.304 | 0.000 |
| rich | 26880 | 0.500 | 0.026 | 28 | 1668.388 | 0.150 | 0.000 |
