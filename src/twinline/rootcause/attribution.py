"""attribution.py: which driver most plausibly explains a defect pattern.

driver_importance_report() runs permutation importance on the trained
defect-risk model and rolls per-feature importance up to driver categories.
Most of our journey features (sensor deviation, process-state proxies) can't
distinguish tool_wear from supplier_batch from ambient on their own — only
manual-check failures and shift timing map cleanly to a driver at the global
model level — so this is honestly dominated by "unknown"; per-incident
attribution is what matched_cohort_comparison() is for.

matched_cohort_comparison() compares defect rate at a station between units
exposed to the suspected anomalous window and a control cohort matched on
variant and shift (so it isn't just measuring "SUV night shift is riskier"),
with a normal-approximation CI on the rate difference. Below the minimum
cohort size, it returns "insufficient evidence" rather than a spurious CI.
"""

import math

import pandas as pd
from sklearn.inspection import permutation_importance

from twinline.predict.defect_risk import DefectRiskModel, build_feature_matrix
from twinline.schemas import CohortComparison, DefectType, DriverImportance, OriginCandidate, PlantLineConfig, RootCauseConfig

_DRIVERS = ("tool_wear", "supplier_batch", "operator_variation", "ambient", "unknown")
_IMPORTANCE_SEED = 42


def _feature_driver(feature_name: str) -> str:
    # Only these features map unambiguously to a driver at the GLOBAL model level —
    # a manual check failure is specifically how operator_variation and PT-05's
    # manifest in this line, and operator_variation is explicitly shift-dependent.
    # Everything else (sensor deviation, process-state) is consistent with more than
    # one driver without per-incident timing context, so it's honestly "unknown" here.
    if feature_name == "own_manual_check_fail_count" or feature_name.startswith("shift_id_"):
        return "operator_variation"
    return "unknown"


def driver_importance_report(risk_model: DefectRiskModel, rows: pd.DataFrame, n_repeats: int = 10) -> list[DriverImportance]:
    x = build_feature_matrix(risk_model, rows)
    y = rows["y"].to_numpy(dtype=int)
    result = permutation_importance(
        risk_model.model, x, y, n_repeats=n_repeats, random_state=_IMPORTANCE_SEED, scoring="average_precision"
    )

    driver_totals = {driver: 0.0 for driver in _DRIVERS}
    for i, feature in enumerate(risk_model.feature_columns):
        driver_totals[_feature_driver(feature)] += max(float(result.importances_mean[i]), 0.0)

    total = sum(driver_totals.values())
    shares = {d: (v / total if total > 0 else 0.0) for d, v in driver_totals.items()}
    return sorted(
        (DriverImportance(driver=d, importance_share=s) for d, s in shares.items()),
        key=lambda d: d.importance_share, reverse=True,
    )


def infer_suspected_driver(candidate: OriginCandidate) -> str:
    signals = {e.signal for e in candidate.evidence}
    rules = {e.rule for e in candidate.evidence}
    if "manual_check" in signals:
        return "operator_variation"
    if rules & {"rule4_run_same_side", "cusum"}:
        return "tool_wear"
    if "rule1_beyond_3sigma" in rules:
        return "supplier_batch"
    if "anomaly" in signals:
        return "ambient"
    return "unknown"


def matched_cohort_comparison(
    station_id: str,
    suspected_driver: str,
    window_start_s: float,
    window_end_s: float,
    defect_type: DefectType,
    units: pd.DataFrame,
    defects: pd.DataFrame,
    plant: PlantLineConfig,
    cfg: RootCauseConfig,
) -> CohortComparison:
    station = plant.station_by_id(station_id)
    visit_time_s = units["start_time_s"] + station.sequence * plant.takt_seconds
    exposed_mask = (visit_time_s >= window_start_s) & (visit_time_s <= window_end_s)

    # Match the control cohort to the same variant/shift combinations seen in the
    # exposed cohort, so the comparison isn't confounded by "this variant/shift is
    # just riskier in general" — it isolates exposure to the suspected window itself.
    variant_shift_key = units["variant_id"] + "|" + units["shift_id"].astype(str)
    exposed_keys = set(variant_shift_key[exposed_mask])
    control_mask = (~exposed_mask) & variant_shift_key.isin(exposed_keys)

    exposed_ids = units.loc[exposed_mask, "unit_id"]
    control_ids = units.loc[control_mask, "unit_id"]
    exposed_n, control_n = len(exposed_ids), len(control_ids)

    if exposed_n < cfg.min_cohort_size or control_n < cfg.min_cohort_size:
        return CohortComparison(
            station_id=station_id, suspected_driver=suspected_driver, exposed_n=exposed_n, control_n=control_n,
            sufficient_evidence=False,
            reason=f"cohort too small (exposed={exposed_n}, control={control_n}, need >= {cfg.min_cohort_size} each)",
        )

    defective_ids = set(defects.loc[defects["detected"] & (defects["defect_type"] == defect_type.value), "unit_id"])
    exposed_rate = float(exposed_ids.isin(defective_ids).mean())
    control_rate = float(control_ids.isin(defective_ids).mean())
    diff = exposed_rate - control_rate

    se = math.sqrt(
        exposed_rate * (1.0 - exposed_rate) / exposed_n + control_rate * (1.0 - control_rate) / control_n
    )
    margin = cfg.cohort_ci_z_score * se

    return CohortComparison(
        station_id=station_id, suspected_driver=suspected_driver, exposed_n=exposed_n, control_n=control_n,
        exposed_defect_rate=exposed_rate, control_defect_rate=control_rate, rate_difference=diff,
        ci_lo=diff - margin, ci_hi=diff + margin, sufficient_evidence=True,
    )
