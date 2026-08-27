"""run_spc(): control-chart signals (EWMA, CUSUM, Western Electric rules 1-4)
over every real sensor series, plus soft-estimated series at blind stations
that clear the soft-sensor confidence floor.

Deliberately does NOT reuse build_station_features' bucket means: those use
heavily overlapping trailing windows (tuned for soft-sensor training volume),
and Western Electric / CUSUM rules assume roughly independent samples — fed
autocorrelated overlapping means, rule4 and CUSUM fire on smoothing artifacts
regardless of whether a real fault is present. SPC recomputes disjoint,
non-overlapping bucket means straight from readings instead.
"""

import numpy as np
import pandas as pd

from twinline.detect.spc_rules import (
    cusum_mask,
    ewma_mask,
    rule1_beyond_3sigma,
    rule2_two_of_three_beyond_2sigma,
    rule3_four_of_five_beyond_1sigma,
    rule4_run_same_side,
)
from twinline.features.soft_sensors import SoftSensorStore, estimate
from twinline.features.windowing import bucket_end_times, trailing_window_stats
from twinline.schemas import DetectConfig, FeaturesConfig, InstrumentationTier, PlantLineConfig, SPCConfig, SPCSignal, Severity

_MIN_BASELINE_POINTS = 5
EWMA_ALPHA = 0.3

_RULE_SEVERITY = {
    "rule1_beyond_3sigma": Severity.CRITICAL,
    "cusum": Severity.CRITICAL,
    "ewma": Severity.WARN,
    "rule2_two_of_three_beyond_2sigma": Severity.WARN,
    "rule3_four_of_five_beyond_1sigma": Severity.WATCH,
    "rule4_run_same_side": Severity.WATCH,
}


def run_spc(
    readings: pd.DataFrame,
    plant: PlantLineConfig,
    features_cfg: FeaturesConfig,
    detect_cfg: DetectConfig,
    soft_store: SoftSensorStore | None = None,
) -> list[SPCSignal]:
    bucket_seconds = features_cfg.station_window.bucket_minutes * 60.0
    signals: list[SPCSignal] = []

    for station in sorted(plant.stations, key=lambda s: s.sequence):
        if station.instrumentation == InstrumentationTier.MANUAL:
            continue
        st_readings = readings[readings["station_id"] == station.id]
        for sensor in station.sensors:
            sensor_readings = st_readings[st_readings["sensor_name"] == sensor]
            if sensor_readings.empty:
                continue
            bucket_ends = bucket_end_times(
                float(sensor_readings["timestamp_s"].min()), float(sensor_readings["timestamp_s"].max()), bucket_seconds
            )
            stats = trailing_window_stats(
                sensor_readings["timestamp_s"].to_numpy(), sensor_readings["value"].to_numpy(), bucket_ends, bucket_seconds
            )
            series = pd.Series(stats["mean"].to_numpy(), index=stats["bucket_end_s"].to_numpy())
            signals.extend(_evaluate_series(station.id, sensor, series, "real", detect_cfg))

    if soft_store is not None:
        stride = max(round(features_cfg.station_window.window_minutes / features_cfg.station_window.bucket_minutes), 1)
        for station_id, archetype in soft_store.archetypes_by_station.items():
            app_rows = soft_store.datasets[archetype.id].application[station_id].iloc[::stride]
            bucket_ends = app_rows["bucket_end_s"].to_numpy()
            values = []
            for bucket_end_s in bucket_ends:
                est = estimate(soft_store, station_id, float(bucket_end_s))
                values.append(est.value if est is not None else np.nan)
            series = pd.Series(values, index=bucket_ends)
            signals.extend(_evaluate_series(station_id, archetype.target_sensor, series, "soft", detect_cfg))

    return signals


def _evaluate_series(
    station_id: str, sensor: str, series: pd.Series, provenance: str, detect_cfg: DetectConfig
) -> list[SPCSignal]:
    valid = series.dropna()
    if len(valid) < _MIN_BASELINE_POINTS:
        return []

    bucket_ends = valid.index.to_numpy(dtype=float)
    values = valid.to_numpy(dtype=float)
    center = float(np.mean(values))
    sigma = float(np.std(values, ddof=0))
    if sigma == 0.0:
        return []

    cfg = detect_cfg.spc
    masks = {
        "rule1_beyond_3sigma": rule1_beyond_3sigma(values, center, sigma),
        "rule2_two_of_three_beyond_2sigma": rule2_two_of_three_beyond_2sigma(values, center, sigma, cfg),
        "rule3_four_of_five_beyond_1sigma": rule3_four_of_five_beyond_1sigma(values, center, sigma, cfg),
        "rule4_run_same_side": rule4_run_same_side(values, center, cfg.rule4_run_length),
        "ewma": ewma_mask(values, EWMA_ALPHA, center, sigma, cfg.ewma_control_limit_l),
        "cusum": cusum_mask(values, center, sigma, cfg.cusum_k_sigma, cfg.cusum_h_sigma),
    }

    signals: list[SPCSignal] = []
    for rule_name, mask in masks.items():
        lookback = _rule_lookback(rule_name, cfg)
        for i in np.flatnonzero(mask):
            window_start = bucket_ends[max(0, i - lookback + 1)]
            signals.append(
                SPCSignal(
                    rule_name=rule_name,
                    station_id=station_id,
                    sensor=sensor,
                    severity=_RULE_SEVERITY[rule_name],
                    window_start_s=float(window_start),
                    window_end_s=float(bucket_ends[i]),
                    value=float(values[i]),
                    center_line=center,
                    provenance=provenance,
                )
            )
    return signals


def _rule_lookback(rule_name: str, cfg: SPCConfig) -> int:
    return {
        "rule1_beyond_3sigma": 1,
        "rule2_two_of_three_beyond_2sigma": cfg.rule2_window,
        "rule3_four_of_five_beyond_1sigma": cfg.rule3_window,
        "rule4_run_same_side": cfg.rule4_run_length,
        "ewma": 1,
        "cusum": 1,
    }[rule_name]
