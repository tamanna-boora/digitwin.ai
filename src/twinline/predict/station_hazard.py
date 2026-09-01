"""station_hazard_now(): a live, per-station risk score independent of any
single unit's prediction — the worst recent SPC/anomaly signal at that
station, decayed by staleness and (for anomaly signals) down-weighted by
the confidence of the soft-sensor input that produced it.
"""

import numpy as np

from twinline.schemas import AnomalySignal, SPCSignal, Severity, StationHazardConfig

_SEVERITY_WEIGHT_FIELD = {
    Severity.WATCH: "severity_weight_watch",
    Severity.WARN: "severity_weight_warn",
    Severity.CRITICAL: "severity_weight_critical",
}


def station_hazard_now(
    station_id: str,
    as_of_time_s: float,
    spc_signals: list[SPCSignal],
    anomaly_signals: list[AnomalySignal],
    cfg: StationHazardConfig,
) -> float:
    contributions = []

    for signal in spc_signals:
        if signal.station_id != station_id:
            continue
        contribution = _decayed_contribution(signal.window_end_s, signal.severity, as_of_time_s, 1.0, cfg)
        if contribution is not None:
            contributions.append(contribution)

    for signal in anomaly_signals:
        if signal.station_id != station_id:
            continue
        contribution = _decayed_contribution(
            signal.bucket_end_s, signal.severity, as_of_time_s, signal.confidence_weight, cfg
        )
        if contribution is not None:
            contributions.append(contribution)

    return float(np.clip(max(contributions), 0.0, 1.0)) if contributions else 0.0


def _decayed_contribution(
    signal_time_s: float, severity: Severity, as_of_time_s: float, confidence_weight: float, cfg: StationHazardConfig
) -> float | None:
    age = as_of_time_s - signal_time_s
    if age < 0.0 or age > cfg.lookback_seconds:
        return None
    severity_weight = getattr(cfg, _SEVERITY_WEIGHT_FIELD[severity])
    decay = 0.5 ** (age / cfg.half_life_seconds)
    return severity_weight * decay * confidence_weight
