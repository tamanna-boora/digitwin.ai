"""Pure helpers matching detect/ signals — and a unit's own manual check
outcome — to a specific station visit. These are the building blocks
trace.py scores candidate origin stations with. Manual-check evidence
matters a lot here: several stations (FA-03..06) carry no sensors and no
soft-sensor coverage at all, so a failed check on THIS unit's own visit is
the only evidence that can ever implicate them.
"""

import math

import pandas as pd

from twinline.schemas import AnomalySignal, Evidence, RootCauseConfig, Severity, SPCSignal

_SEVERITY_SCORE = {Severity.WATCH: 0.3, Severity.WARN: 0.6, Severity.CRITICAL: 1.0}


def spc_evidence_near_visit(
    station_id: str, visit_time_s: float, spc_signals: list[SPCSignal], cfg: RootCauseConfig
) -> list[Evidence]:
    matches = []
    for signal in spc_signals:
        if signal.station_id != station_id:
            continue
        if _within_tolerance(visit_time_s, signal.window_start_s, signal.window_end_s, cfg.visit_window_tolerance_seconds):
            matches.append(
                Evidence(
                    signal="spc", rule=signal.rule_name, station_id=station_id, timestamp_s=signal.window_end_s,
                    provenance=signal.provenance,
                    detail=f"{signal.rule_name} on {signal.sensor}: value={signal.value:.3f} vs center={signal.center_line:.3f}",
                )
            )
    return matches


def anomaly_evidence_near_visit(
    station_id: str, visit_time_s: float, anomaly_signals: list[AnomalySignal], cfg: RootCauseConfig
) -> list[Evidence]:
    matches = []
    for signal in anomaly_signals:
        if signal.station_id != station_id:
            continue
        if _within_tolerance(visit_time_s, signal.bucket_end_s, signal.bucket_end_s, cfg.visit_window_tolerance_seconds):
            provenance = "soft" if signal.method == "modified_z_soft" else "real"
            matches.append(
                Evidence(
                    signal="anomaly", rule=signal.method, station_id=station_id, timestamp_s=signal.bucket_end_s,
                    provenance=provenance,
                    detail=f"{signal.method} score={signal.score:.2f} (confidence={signal.confidence_weight:.2f}) "
                    f"on {', '.join(signal.contributing_features)}",
                )
            )
    return matches


def own_manual_check_evidence(
    station_id: str, unit_id: str, visit_time_s: float, manual_checks: pd.DataFrame
) -> list[Evidence]:
    own = manual_checks[(manual_checks["station_id"] == station_id) & (manual_checks["unit_id"] == unit_id)]
    failed = own[~own["check_pass"]]
    if failed.empty:
        return []
    row = failed.iloc[0]
    return [
        Evidence(
            signal="manual_check", rule="check_fail", station_id=station_id, timestamp_s=float(row["timestamp_s"]),
            provenance="manual", detail=f"operator {row['operator_id']} failed the checklist for this unit here",
        )
    ]


def signal_severity_score(
    spc_matches: list[SPCSignal], anomaly_matches: list[AnomalySignal], manual_check_failed: bool, manual_check_fail_bonus: float
) -> float:
    spc_score = sum(_SEVERITY_SCORE[s.severity] for s in spc_matches)
    anomaly_score = sum(_SEVERITY_SCORE[s.severity] * s.confidence_weight for s in anomaly_matches)
    manual_score = manual_check_fail_bonus if manual_check_failed else 0.0
    return spc_score + anomaly_score + manual_score


def combined_window(
    spc_matches: list[SPCSignal],
    anomaly_matches: list[AnomalySignal],
    fallback_time_s: float,
    tolerance_seconds: float,
    max_span_seconds: float,
) -> tuple[float, float]:
    starts = [s.window_start_s for s in spc_matches] + [s.bucket_end_s - tolerance_seconds for s in anomaly_matches]
    ends = [s.window_end_s for s in spc_matches] + [s.bucket_end_s for s in anomaly_matches]
    if not starts:
        start, end = fallback_time_s - tolerance_seconds, fallback_time_s + tolerance_seconds
    else:
        start, end = min(starts), max(ends)

    if end - start > max_span_seconds:
        # Several rule violations with different characteristic durations clustered
        # near the visit — re-center a bounded window on the visit itself rather than
        # keep the full envelope, which can otherwise span hours.
        half = max_span_seconds / 2.0
        start, end = fallback_time_s - half, fallback_time_s + half
    return start, end


def _within_tolerance(visit_time_s: float, window_start_s: float, window_end_s: float, tolerance_seconds: float) -> bool:
    return (window_start_s - tolerance_seconds) <= visit_time_s <= (window_end_s + tolerance_seconds)


def confidence_from_score(score: float, reference_score: float) -> float:
    """Smooth saturating map of an unbounded evidence score onto [0, 1) — a score at
    reference_score reads as ~63% confident, well past it approaches but never reaches 1.
    """
    return 1.0 - math.exp(-score / reference_score)
