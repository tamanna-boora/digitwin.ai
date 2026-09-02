"""Shared alert-building orchestration: composes trace_defect -> units_at_risk
-> infer_suspected_driver -> matched_cohort_comparison -> build_recommendation
into ranked, alarm-budgeted alerts. Used by both the Floor Supervisor page
(live feed) and the Trust & Validation page (acknowledgement history).

Multiple detected defects tracing back to the same (station, driver) within a
shift are the same underlying issue seen more than once, not distinct alerts
worth separate alarm-budget slots -- they're collapsed to the
highest-confidence occurrence before budgeting, with occurrence_count
recording how many incidents fed into it.
"""

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from twinline.actions.recommender import build_recommendation
from twinline.config import AppConfig
from twinline.predict.calibration import select_alerts
from twinline.rootcause.attribution import infer_suspected_driver, matched_cohort_comparison
from twinline.rootcause.evidence import confidence_from_score
from twinline.rootcause.trace import trace_defect, units_at_risk
from twinline.schemas import (
    AlertCandidate,
    AnomalySignal,
    DefectType,
    Recommendation,
    RootCauseTrace,
    SPCSignal,
    UnitsAtRisk,
)


@dataclass(frozen=True)
class AlertDetail:
    candidate: AlertCandidate
    trace: RootCauseTrace
    driver: str
    risk: UnitsAtRisk
    recommendation: Recommendation
    occurrence_count: int = 1


def build_open_alerts(
    cfg: AppConfig,
    units: pd.DataFrame,
    defects: pd.DataFrame,
    manual_checks: pd.DataFrame,
    spc_signals: list[SPCSignal],
    anomaly_signals: list[AnomalySignal],
    as_of_time_s: float,
) -> tuple[list[AlertDetail], list[AlertDetail]]:
    """Returns (selected, digest) — both alarm-budgeted, ranked by expected cost avoided."""
    units_by_id = units.set_index("unit_id")
    detected = defects[defects["detected"] & (defects["detection_time_s"] <= as_of_time_s)]

    details_by_id: dict[str, AlertDetail] = {}
    for row in detected.itertuples(index=False):
        defect_type = DefectType(row.defect_type)
        trace = trace_defect(
            row.unit_id, defect_type, row.detection_station_id, row.detection_time_s, units, defects, manual_checks,
            cfg.plant, spc_signals, anomaly_signals, cfg.model.rootcause,
        )
        if not trace.candidates:
            continue
        top = trace.candidates[0]
        driver = infer_suspected_driver(top)
        if driver == "unknown":
            continue

        risk = units_at_risk(units, cfg.plant, top.station_id, top.window_start_s, top.window_end_s, defect_type, as_of_time_s)
        comparison = matched_cohort_comparison(
            top.station_id, driver, top.window_start_s, top.window_end_s, defect_type, units, defects, cfg.plant,
            cfg.model.rootcause,
        )
        confidence = confidence_from_score(top.score, cfg.model.rootcause.confidence_reference_score)
        if comparison.sufficient_evidence and comparison.exposed_defect_rate is not None:
            probability = comparison.exposed_defect_rate
        else:
            probability = float(np.clip(confidence, 0.05, 0.95))

        unit_shift = str(units_by_id.loc[row.unit_id, "shift_id"])
        rework_cost = cfg.model.predict.alarm_budget.rework_cost_currency
        recommendation = build_recommendation(
            driver, top.station_id, risk, probability, confidence, rework_cost, cfg.model.actions, shift_id=unit_shift
        )

        alert_id = f"{row.unit_id}::{top.station_id}"
        candidate = AlertCandidate(
            id=alert_id, station_id=top.station_id, shift_id=unit_shift, probability=probability,
            units_at_risk=len(risk.unit_ids), rework_cost=rework_cost,
            reason=f"{driver} suspected at {top.station_id}",
        )
        details_by_id[alert_id] = AlertDetail(
            candidate=candidate, trace=trace, driver=driver, risk=risk, recommendation=recommendation
        )

    deduped = _dedupe_by_station_and_driver(list(details_by_id.values()))
    deduped_by_id = {d.candidate.id: d for d in deduped}

    selected_candidates, digest_candidates = select_alerts(
        [d.candidate for d in deduped], cfg.model.predict.alarm_budget
    )
    selected = [deduped_by_id[c.id] for c in selected_candidates]
    digest = [deduped_by_id[c.id] for c in digest_candidates]
    return selected, digest


def _dedupe_by_station_and_driver(details: list[AlertDetail]) -> list[AlertDetail]:
    """Collapse repeat incidents at the same (station, driver, shift) to one
    representative, recording how many incidents fed into it. Scoped per
    shift because the alarm budget itself is per shift -- collapsing across
    shifts would let one shift's occurrence count eat another's slot.

    The representative is picked by (has units still at risk, confidence),
    in that order -- an occurrence whose containment window is still open is
    strictly more actionable than one that's already resolved, regardless of
    which one the root-cause trace happens to be more confident about."""
    groups: dict[tuple[str, str, str], list[AlertDetail]] = {}
    for detail in details:
        key = (detail.candidate.station_id, detail.driver, detail.candidate.shift_id)
        groups.setdefault(key, []).append(detail)

    result = []
    for group in groups.values():
        best = max(group, key=lambda d: (len(d.risk.unit_ids) > 0, d.recommendation.confidence))
        result.append(replace(best, occurrence_count=len(group)))
    return result


def current_station_id(cfg: AppConfig, unit_id: str, units_by_id: pd.DataFrame, now_s: float) -> str:
    start_time_s = float(units_by_id.loc[unit_id, "start_time_s"])
    elapsed_stations = int((now_s - start_time_s) // cfg.plant.takt_seconds)
    sequence = max(1, min(elapsed_stations, len(cfg.plant.stations)))
    return next(s.id for s in cfg.plant.stations if s.sequence == sequence)
