# Decisions

Twelve calls we made on this build, and why.

## 1. Time-based train/test split

We chose a chronological split: earliest data trains, latest data tests.
We rejected a random split.
Tool wear drifts slowly. A random split lets the model see the future.

## 2. Quantile regression for soft sensors

We chose quantile regression at 0.1, 0.5, 0.9 for blind-station estimates.
We rejected a single point estimate.
An unqualified number at a blind station is worse than no number.

## 3. A confidence floor that abstains

We chose a floor below which the twin returns nothing.
We rejected always answering.
The twin saying "I don't know here" is a designed behavior, not a gap.

## 4. Alarm budget ranked by expected cost avoided

We chose to rank and budget alerts by expected cost avoided.
We rejected a fixed probability threshold.
Supervisor attention is the scarce resource, not probability mass.

## 5. Shadow mode, advisory-only output

We chose advisory output addressed to a named owner, with no write path to the line.
We rejected closed-loop control.
Plants retrofit in a few scheduled windows a year, and nobody accepts liability for an automated line stop.

## 6. 24 stations

We chose 24 stations.
We rejected 42.
The deadline was real.

## 7. Ground truth in a separate, guarded file

We chose a separate ground-truth file with a runtime check that no feature derives from it.
We rejected keeping causes in the main table.
It would have made every accuracy number meaningless.

## 8. Fixed the anomaly scorer twice

We chose percentile rank, then max absolute z-score, then fixed both when each broke.
We rejected shipping either broken version.
Percentile rank guaranteed half the points sat above the midpoint. Max-|z| across roughly 100 correlated columns on roughly 96 samples inflated false positives through multiple comparisons. False alarms went from 311 to 303, matching the configured 8% contamination.

## 9. Fixed rules-score saturation

We chose real window containment and mean-not-max aggregation across stations.
We rejected the original worst-of-16-stations match within plus or minus 1200 seconds.
That match returned a constant 1.0 for every unit. The fix gives 191 distinct values with spread 0.02 to 0.53.

## 10. Did not tune the blend weight to force a win

We chose to report that hybrid still lost to ML-only after the rules-score fix.
We rejected tuning the blend weight until it didn't.
Tuning it against the test set to make a claim true would have been dishonest.

## 11. Constraint recompute keyed to real buffer improvement

We chose to key the constraint-station recompute to actual buffer-utilisation improvement.
We rejected keying it to buffer_slots.
Speed-only changes could clear a bottleneck without the display ever updating.

## 12. Leadership business case sources the net-positive row

We chose to source the business case from the ML-only row and generate the callout from the data.
We rejected hardcoding the callout text or sourcing from hybrid+soft-sensors.
ML-only is the only net-positive configuration, and a hardcoded callout can go stale.

## 13. units_at_risk allows zero

We chose `Field(ge=0)` on AlertCandidate.units_at_risk.
We rejected keeping `Field(gt=0)`.
A validation rule that forces the caller to fabricate a value with max(len(x), 1) is worse than no validation. Zero units at risk is a real and meaningful state, not an error.

## 14. Containment panel renders the same union the KPI sums

We chose to render the full at-risk union in the containment table.
We rejected showing only the top-ranked alert's list.
Two numbers on one screen that can legitimately disagree always read as a bug.

## 15. Risk-first tie-break when deduplicating alerts

We chose to prefer the group member with real containment risk, confidence second.
We rejected picking by highest confidence alone.
Confidence-only picked an already-resolved alert as the representative in a real test case and hid a sibling's genuine at-risk units. The surfaced alert should be the one with real containment value, not whichever one the trace happened to be more confident about.

## Verified, not fixed: zero containment at clock=12 and clock=16

We checked the units-at-risk display at clock=12.00h and clock=16.00h and found zero at both. We traced it station by station and confirmed this is correct, not a bug: every defect detected by those clock positions had already had its containment window close before the clock caught up to it. The first genuinely open containment window in this run is at clock=14.25h. We did not change the underlying window logic to manufacture a non-zero result at 12 or 16.
