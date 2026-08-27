"""run_anomaly_detection(): IsolationForest + robust modified z-score on each
rich/partial station's own feature vector, combined into one [0,1] score.
Blind stations with soft-sensor coverage get a lighter univariate check on
their estimated series, scaled by that estimate's own confidence — a shaky
soft reading should move the needle less than a real one, not the same.
"""

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import IsolationForest

from twinline.features.soft_sensors import SoftSensorStore, estimate
from twinline.features.station_features import StationFeatureFrame
from twinline.schemas import (
    AnomalyDetectConfig,
    AnomalySignal,
    ArchetypeConfig,
    InstrumentationTier,
    PlantLineConfig,
    Severity,
)

_MIN_SAMPLES = 8
_ISO_SEED = 42


@dataclass(frozen=True)
class _ScoredBucket:
    bucket_end_s: float
    combined_score: float
    top_features: list[str]
    confidence_weight: float


def run_anomaly_detection(
    station_features: StationFeatureFrame,
    plant: PlantLineConfig,
    detect_cfg: AnomalyDetectConfig,
    soft_store: SoftSensorStore | None = None,
) -> list[AnomalySignal]:
    signals: list[AnomalySignal] = []

    for station in sorted(plant.stations, key=lambda s: s.sequence):
        if station.instrumentation == InstrumentationTier.MANUAL:
            continue
        signals.extend(_station_anomaly_signals(station.id, station_features, detect_cfg))

    if soft_store is not None:
        for station_id, archetype in soft_store.archetypes_by_station.items():
            signals.extend(_soft_station_anomaly_signals(station_id, archetype, soft_store, detect_cfg))

    return signals


_PROCESS_STATE_COLUMNS = {
    "cycle_time_variance", "blocked_ratio", "starved_ratio", "buffer_utilisation", "micro_stoppage_count",
    "check_pass_rate", "n_distinct_operators", "dominant_operator_share",
}


def _select_anomaly_columns(columns: list[str]) -> list[str]:
    # Keep one column per underlying signal (its "_mean") plus process-state summaries.
    # The full feature store also carries _std/_p95/_ewma/_slope/mix-fraction columns —
    # highly correlated with _mean and with each other, and with ~96 station-buckets
    # of data, keeping all ~100 columns both inflates multiple-comparison false
    # positives on the z-score check and pushes IsolationForest into the curse of
    # dimensionality (n_features approaching n_samples).
    return [c for c in columns if c.endswith("_mean") or c in _PROCESS_STATE_COLUMNS]


def _station_anomaly_signals(
    station_id: str, station_features: StationFeatureFrame, cfg: AnomalyDetectConfig
) -> list[AnomalySignal]:
    frame = station_features.wide.loc[station_id]
    frame = frame.dropna(axis=1, how="all")
    if frame.empty or len(frame) < _MIN_SAMPLES:
        return []
    keep = _select_anomaly_columns(list(frame.columns))
    frame = frame[keep] if keep else frame
    frame = frame.fillna(frame.median(numeric_only=True))
    frame = frame.dropna(axis=1, how="any")
    if frame.shape[1] == 0:
        return []

    x = frame.to_numpy(dtype=float)
    iso_scores = _isolation_forest_scores(x, cfg.isolation_forest_contamination)
    z_scores, top_feature_idx = _modified_z_scores(x, cfg.modified_z_threshold)

    combined = cfg.weight_isolation_forest * iso_scores + cfg.weight_modified_z * z_scores
    columns = list(frame.columns)

    signals = []
    for i, bucket_end_s in enumerate(frame.index.to_numpy(dtype=float)):
        signal = _make_signal(
            station_id, bucket_end_s, combined[i], [columns[top_feature_idx[i]]], "isolation_forest+modified_z",
            confidence_weight=1.0, cfg=cfg,
        )
        if signal is not None:
            signals.append(signal)
    return signals


def _soft_station_anomaly_signals(
    station_id: str, archetype: ArchetypeConfig, soft_store: SoftSensorStore, cfg: AnomalyDetectConfig
) -> list[AnomalySignal]:
    app_rows = soft_store.datasets[archetype.id].application[station_id]
    values, confidences, bucket_ends = [], [], []
    for bucket_end_s in app_rows["bucket_end_s"].to_numpy():
        est = estimate(soft_store, station_id, float(bucket_end_s))
        if est is None:
            continue
        values.append(est.value)
        confidences.append(est.confidence)
        bucket_ends.append(bucket_end_s)

    if len(values) < _MIN_SAMPLES:
        return []

    x = np.array(values).reshape(-1, 1)
    z_scores, _ = _modified_z_scores(x, cfg.modified_z_threshold)

    signals = []
    for i, bucket_end_s in enumerate(bucket_ends):
        signal = _make_signal(
            station_id, float(bucket_end_s), z_scores[i], [archetype.target_sensor], "modified_z_soft",
            confidence_weight=confidences[i], cfg=cfg,
        )
        if signal is not None:
            signals.append(signal)
    return signals


def _isolation_forest_scores(x: np.ndarray, contamination: float) -> np.ndarray:
    model = IsolationForest(contamination=contamination, random_state=_ISO_SEED, n_estimators=100)
    model.fit(x)
    # decision_function is >= 0 for points sklearn considers normal (given `contamination`)
    # and negative for outliers — clip normal points to exactly 0 rather than rank them,
    # so the "most points are fine" assumption survives into the combined score instead
    # of a percentile rank spreading everything uniformly across (0, 1].
    outlier_depth = np.clip(-model.decision_function(x), 0.0, None)
    scale = outlier_depth.max()
    return outlier_depth / scale if scale > 0 else outlier_depth


def _modified_z_scores(x: np.ndarray, threshold: float) -> tuple[np.ndarray, np.ndarray]:
    median = np.median(x, axis=0)
    mad = np.median(np.abs(x - median), axis=0)
    mad_safe = np.where(mad == 0, 1e-9, mad)
    z = 0.6745 * (x - median) / mad_safe
    abs_z = np.abs(z)
    top_feature_idx = np.argmax(abs_z, axis=1)
    max_abs_z = np.max(abs_z, axis=1)
    normalized = np.clip(max_abs_z / threshold, 0.0, 1.0)
    return normalized, top_feature_idx


def _make_signal(
    station_id: str,
    bucket_end_s: float,
    score: float,
    top_features: list[str],
    method: str,
    confidence_weight: float,
    cfg: AnomalyDetectConfig,
) -> AnomalySignal | None:
    weighted_score = float(np.clip(score * confidence_weight, 0.0, 1.0))
    if weighted_score >= cfg.severity_critical_threshold:
        severity = Severity.CRITICAL
    elif weighted_score >= cfg.severity_warn_threshold:
        severity = Severity.WARN
    elif weighted_score >= cfg.severity_watch_threshold:
        severity = Severity.WATCH
    else:
        return None
    return AnomalySignal(
        station_id=station_id, bucket_end_s=bucket_end_s, method=method, score=weighted_score,
        severity=severity, contributing_features=top_features, confidence_weight=confidence_weight,
    )
