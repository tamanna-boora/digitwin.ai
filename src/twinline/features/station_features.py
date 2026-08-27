"""build_station_features(): per station, per trailing time-bucket, process
condition — sensor stats, EWMA/slope, process-state proxies, operator/shift/
variant mix. Every bucket_end_s only aggregates data at or before it.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from twinline.features.categorical import fixed_category_fractions_trailing, operator_diversity_trailing
from twinline.features.provenance import Provenance
from twinline.features.windowing import bucket_end_times, ewma_series, rolling_slope, trailing_window_stats
from twinline.schemas import FeaturesConfig, InstrumentationTier, PlantLineConfig, ProcessStateConfig, StationConfig


@dataclass(frozen=True)
class StationFeatureFrame:
    wide: pd.DataFrame
    provenance: pd.DataFrame


def build_station_features(
    readings: pd.DataFrame,
    manual_checks: pd.DataFrame,
    units: pd.DataFrame,
    plant: PlantLineConfig,
    features_cfg: FeaturesConfig,
    window_minutes: float | None = None,
) -> StationFeatureFrame:
    window_cfg = features_cfg.station_window
    bucket_seconds = window_cfg.bucket_minutes * 60.0
    window_seconds = (window_minutes if window_minutes is not None else window_cfg.window_minutes) * 60.0

    all_times = pd.concat([readings["timestamp_s"], manual_checks["timestamp_s"]])
    if all_times.empty:
        raise ValueError("no readings or manual checks to build station features from")
    bucket_ends = bucket_end_times(float(all_times.min()), float(all_times.max()), bucket_seconds)

    units_by_id = units.set_index("unit_id")
    variant_ids = [v.id for v in plant.variants]
    shift_ids = [s.id.value for s in plant.shifts]

    frames, prov_frames = [], []
    for station in sorted(plant.stations, key=lambda s: s.sequence):
        if station.instrumentation == InstrumentationTier.MANUAL:
            frame, prov = _manual_station_block(
                station, manual_checks, units_by_id, bucket_ends, window_seconds, variant_ids, shift_ids
            )
        else:
            frame, prov = _sensored_station_block(
                station,
                readings,
                units_by_id,
                bucket_ends,
                window_seconds,
                window_cfg.ewma_alpha,
                window_cfg.rolling_slope_lookback_buckets,
                features_cfg.process_state,
                plant.takt_seconds,
                variant_ids,
                shift_ids,
            )
        frames.append(frame)
        prov_frames.append(prov)

    wide = pd.concat(frames, ignore_index=True).set_index(["station_id", "bucket_end_s"])
    provenance = pd.concat(prov_frames, ignore_index=True).set_index(["station_id", "bucket_end_s"])
    return StationFeatureFrame(wide=wide, provenance=provenance)


def _mix_block(
    visit_timestamps: np.ndarray,
    visit_unit_ids: np.ndarray,
    units_by_id: pd.DataFrame,
    bucket_ends: np.ndarray,
    window_seconds: float,
    variant_ids: list[str],
    shift_ids: list[str],
) -> dict[str, np.ndarray]:
    variant_of_visit = units_by_id.loc[visit_unit_ids, "variant_id"].to_numpy()
    shift_of_visit = units_by_id.loc[visit_unit_ids, "shift_id"].to_numpy()

    variant_fracs = fixed_category_fractions_trailing(
        visit_timestamps, variant_of_visit, bucket_ends, window_seconds, variant_ids
    )
    shift_fracs = fixed_category_fractions_trailing(
        visit_timestamps, shift_of_visit, bucket_ends, window_seconds, shift_ids
    )
    out: dict[str, np.ndarray] = {}
    for vid, arr in variant_fracs.items():
        out[f"variant_frac_{vid}"] = arr
    for sid, arr in shift_fracs.items():
        out[f"shift_frac_{sid}"] = arr
    return out


def _sensored_station_block(
    station: StationConfig,
    readings: pd.DataFrame,
    units_by_id: pd.DataFrame,
    bucket_ends: np.ndarray,
    window_seconds: float,
    ewma_alpha: float,
    slope_lookback: int,
    process_cfg: ProcessStateConfig,
    takt_seconds: float,
    variant_ids: list[str],
    shift_ids: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    st_readings = readings[readings["station_id"] == station.id]
    columns: dict[str, np.ndarray] = {}
    provenance: dict[str, str] = {}

    for sensor in station.sensors:
        sensor_readings = st_readings[st_readings["sensor_name"] == sensor]
        stats = trailing_window_stats(
            sensor_readings["timestamp_s"].to_numpy(),
            sensor_readings["value"].to_numpy(),
            bucket_ends,
            window_seconds,
        )
        ewma = ewma_series(stats["mean"], ewma_alpha)
        slope = rolling_slope(ewma, slope_lookback)
        columns[f"{sensor}_mean"] = stats["mean"].to_numpy()
        columns[f"{sensor}_std"] = stats["std"].to_numpy()
        columns[f"{sensor}_p95"] = stats["p95"].to_numpy()
        columns[f"{sensor}_ewma"] = ewma.to_numpy()
        columns[f"{sensor}_slope"] = slope.to_numpy()
        for suffix in ("mean", "std", "p95", "ewma", "slope"):
            provenance[f"{sensor}_{suffix}"] = Provenance.REAL.value

    if "cycle_time_s" in station.sensors:
        columns["cycle_time_variance"] = columns["cycle_time_s_std"] ** 2
        blocked, starved, buffer_util, micro_stops = _process_state_trailing(
            st_readings[st_readings["sensor_name"] == "cycle_time_s"]["timestamp_s"].to_numpy(),
            st_readings[st_readings["sensor_name"] == "cycle_time_s"]["value"].to_numpy(),
            bucket_ends,
            window_seconds,
            takt_seconds,
            process_cfg,
        )
        columns["blocked_ratio"] = blocked
        columns["starved_ratio"] = starved
        columns["buffer_utilisation"] = buffer_util
        columns["micro_stoppage_count"] = micro_stops
        for name in ("cycle_time_variance", "blocked_ratio", "starved_ratio", "buffer_utilisation", "micro_stoppage_count"):
            provenance[name] = Provenance.REAL.value

    visits = st_readings.drop_duplicates(subset=["unit_id", "timestamp_s"])
    mix = _mix_block(
        visits["timestamp_s"].to_numpy(), visits["unit_id"].to_numpy(), units_by_id, bucket_ends, window_seconds,
        variant_ids, shift_ids,
    )
    columns.update(mix)
    for name in mix:
        provenance[name] = Provenance.REAL.value

    return _finalize_block(station.id, bucket_ends, columns), _finalize_block(station.id, bucket_ends, provenance)


def _manual_station_block(
    station: StationConfig,
    manual_checks: pd.DataFrame,
    units_by_id: pd.DataFrame,
    bucket_ends: np.ndarray,
    window_seconds: float,
    variant_ids: list[str],
    shift_ids: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    st_checks = manual_checks[manual_checks["station_id"] == station.id]
    timestamps = st_checks["timestamp_s"].to_numpy()

    pass_fracs = fixed_category_fractions_trailing(
        timestamps, st_checks["check_pass"].to_numpy(), bucket_ends, window_seconds, [True]
    )
    n_distinct_ops, dominant_share = operator_diversity_trailing(
        timestamps, st_checks["operator_id"].to_numpy(), bucket_ends, window_seconds
    )

    columns: dict[str, np.ndarray] = {
        "check_pass_rate": pass_fracs[True],
        "n_distinct_operators": n_distinct_ops,
        "dominant_operator_share": dominant_share,
    }
    mix = _mix_block(timestamps, st_checks["unit_id"].to_numpy(), units_by_id, bucket_ends, window_seconds, variant_ids, shift_ids)
    columns.update(mix)

    provenance = {name: Provenance.MANUAL.value for name in columns}
    return _finalize_block(station.id, bucket_ends, columns), _finalize_block(station.id, bucket_ends, provenance)


def _process_state_trailing(
    timestamps: np.ndarray,
    cycle_times: np.ndarray,
    bucket_ends: np.ndarray,
    window_seconds: float,
    takt_seconds: float,
    cfg: ProcessStateConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(timestamps)
    sorted_ts = timestamps[order]
    sorted_ct = cycle_times[order]
    blocked_thr = takt_seconds * cfg.blocked_cycle_time_frac
    starved_thr = takt_seconds * cfg.starved_cycle_time_frac
    micro_thr = takt_seconds * cfg.micro_stoppage_frac

    n = len(bucket_ends)
    blocked_ratio = np.full(n, np.nan)
    starved_ratio = np.full(n, np.nan)
    buffer_utilisation = np.full(n, np.nan)
    micro_stoppage_count = np.zeros(n)

    for i, end in enumerate(bucket_ends):
        window_ct = sorted_ct[(sorted_ts > end - window_seconds) & (sorted_ts <= end)]
        if window_ct.size == 0:
            continue
        blocked_ratio[i] = float(np.mean(window_ct > blocked_thr))
        starved_ratio[i] = float(np.mean(window_ct < starved_thr))
        buffer_utilisation[i] = float(np.mean(window_ct) / takt_seconds)
        micro_stoppage_count[i] = float(np.sum(window_ct > micro_thr))

    return blocked_ratio, starved_ratio, buffer_utilisation, micro_stoppage_count


def _finalize_block(station_id: str, bucket_ends: np.ndarray, columns: dict[str, object]) -> pd.DataFrame:
    data = {"station_id": [station_id] * len(bucket_ends), "bucket_end_s": bucket_ends}
    data.update(columns)
    return pd.DataFrame(data)
