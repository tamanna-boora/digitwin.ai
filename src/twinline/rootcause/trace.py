"""trace.py: given a detected defect, infer which upstream station most
likely caused it by scoring candidates on three things — was the station
anomalous when this unit was there, does its process area naturally produce
this defect_mode, and how many other recently-defective units of the same
type also passed it during the same anomalous window. Never reads
ground_truth.csv — that's the answer key, and consulting it here would
defeat the point of tracing rather than looking it up.

units_at_risk() is the containment list: every unit that passed the
suspected origin station during its anomalous window and hasn't reached its
inspection gate yet as of "now". The line is a deterministic takt-paced
pipeline (sim/line.py), so any unit's visit time at any station is exactly
its start_time_s plus that station's sequence times the takt — no need to
re-derive it from the feature store.
"""

import pandas as pd

from twinline.rootcause.evidence import (
    anomaly_evidence_near_visit,
    combined_window,
    own_manual_check_evidence,
    signal_severity_score,
    spc_evidence_near_visit,
)
from twinline.schemas import (
    AnomalySignal,
    DefectType,
    OriginCandidate,
    PlantLineConfig,
    RootCauseConfig,
    RootCauseTrace,
    SPCSignal,
    StationConfig,
    UnitsAtRisk,
)

# Which process area naturally produces which defect mode — plant-level domain
# knowledge (mirrors sim/line.py's fallback typing), not this specific defect's cause.
AREA_OWNS_DEFECT_TYPE = {
    "BODY": DefectType.WELD_DEFECT,
    "PAINT": DefectType.PAINT_DEFECT,
    "FINAL_ASSEMBLY": DefectType.ASSEMBLY_DEFECT,
}


def _visit_time_s(unit_start_time_s: float, station: StationConfig, takt_seconds: float) -> float:
    return unit_start_time_s + station.sequence * takt_seconds


def _within_tolerance(t: float, start: float, end: float, tolerance_seconds: float) -> bool:
    return (start - tolerance_seconds) <= t <= (end + tolerance_seconds)


def trace_defect(
    unit_id: str,
    defect_type: DefectType,
    detection_station_id: str,
    detection_time_s: float,
    units: pd.DataFrame,
    defects: pd.DataFrame,
    manual_checks: pd.DataFrame,
    plant: PlantLineConfig,
    spc_signals: list[SPCSignal],
    anomaly_signals: list[AnomalySignal],
    cfg: RootCauseConfig,
) -> RootCauseTrace:
    units_by_id = units.set_index("unit_id")
    unit_start_time_s = float(units_by_id.loc[unit_id, "start_time_s"])
    gate_sequence = plant.station_by_id(detection_station_id).sequence

    same_type_unit_ids = defects.loc[
        (defects["defect_type"] == defect_type.value) & defects["detected"] & (defects["unit_id"] != unit_id),
        "unit_id",
    ]
    same_type_start_times = units_by_id.loc[units_by_id.index.isin(same_type_unit_ids), "start_time_s"]

    candidates = []
    for station in [s for s in plant.stations if s.can_cause_defect and s.sequence < gate_sequence]:
        visit_time_s = _visit_time_s(unit_start_time_s, station, plant.takt_seconds)
        spc_matches = [
            s for s in spc_signals
            if s.station_id == station.id
            and _within_tolerance(visit_time_s, s.window_start_s, s.window_end_s, cfg.visit_window_tolerance_seconds)
        ]
        anomaly_matches = [
            a for a in anomaly_signals
            if a.station_id == station.id
            and _within_tolerance(visit_time_s, a.bucket_end_s, a.bucket_end_s, cfg.visit_window_tolerance_seconds)
        ]
        manual_evidence = own_manual_check_evidence(station.id, unit_id, visit_time_s, manual_checks)
        if not spc_matches and not anomaly_matches and not manual_evidence:
            continue

        manual_only = manual_evidence and not spc_matches and not anomaly_matches
        window_tolerance = cfg.manual_only_window_seconds if manual_only else cfg.visit_window_tolerance_seconds
        window_start_s, window_end_s = combined_window(
            spc_matches, anomaly_matches, visit_time_s, window_tolerance, cfg.max_window_span_seconds
        )

        co_occurrence = sum(
            1
            for other_start in same_type_start_times
            if window_start_s <= _visit_time_s(float(other_start), station, plant.takt_seconds) <= window_end_s
        )

        area_bonus = cfg.area_affinity_bonus if AREA_OWNS_DEFECT_TYPE.get(station.area.value) == defect_type else 0.0
        score = (
            signal_severity_score(spc_matches, anomaly_matches, bool(manual_evidence), cfg.manual_check_fail_bonus)
            + area_bonus
            + cfg.co_occurrence_weight * co_occurrence
        )

        evidence = manual_evidence + spc_evidence_near_visit(station.id, visit_time_s, spc_signals, cfg)
        evidence += anomaly_evidence_near_visit(station.id, visit_time_s, anomaly_signals, cfg)
        evidence = sorted(evidence, key=lambda e: e.timestamp_s)[: cfg.max_evidence_per_candidate]

        candidates.append(
            OriginCandidate(
                station_id=station.id, score=score, window_start_s=window_start_s, window_end_s=window_end_s,
                evidence=evidence,
            )
        )

    candidates.sort(key=lambda c: c.score, reverse=True)
    return RootCauseTrace(
        unit_id=unit_id, defect_type=defect_type.value, detection_station_id=detection_station_id,
        detection_time_s=detection_time_s, candidates=candidates[: cfg.max_origin_candidates],
    )


def units_at_risk(
    units: pd.DataFrame,
    plant: PlantLineConfig,
    origin_station_id: str,
    window_start_s: float,
    window_end_s: float,
    defect_type: DefectType,
    as_of_time_s: float,
) -> UnitsAtRisk:
    origin_station = plant.station_by_id(origin_station_id)
    gate = plant.gate_for_defect_type(defect_type)
    gate_sequence = plant.station_by_id(gate.station_id).sequence
    takt = plant.takt_seconds

    origin_visit_time_s = units["start_time_s"] + origin_station.sequence * takt
    gate_visit_time_s = units["start_time_s"] + gate_sequence * takt

    in_window = (origin_visit_time_s >= window_start_s) & (origin_visit_time_s <= window_end_s)
    not_yet_inspected = gate_visit_time_s > as_of_time_s

    at_risk_ids = units.loc[in_window & not_yet_inspected, "unit_id"].tolist()
    return UnitsAtRisk(
        origin_station_id=origin_station_id, window_start_s=window_start_s, window_end_s=window_end_s,
        as_of_time_s=as_of_time_s, unit_ids=at_risk_ids,
    )
