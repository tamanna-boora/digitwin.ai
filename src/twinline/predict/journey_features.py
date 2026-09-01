"""build_unit_journey_features(): one row per unit, aggregating everything
knowable about its journey so far into a fixed-width, uniform feature vector
— uniform because every station reports the same process-state column names
regardless of which sensors it happens to carry, so aggregation doesn't need
per-sensor special-casing. Built entirely from features/store.py's already
point-in-time-safe UnitFeatureFrame; pass as_of_sequence to score a unit
mid-journey using only the stations it has reached so far.
"""

import numpy as np
import pandas as pd

from twinline.features.soft_sensors import SoftSensorStore, estimate
from twinline.features.store import UnitFeatureFrame
from twinline.schemas import InstrumentationTier, PlantLineConfig, SensorSpecConfig

_PROCESS_COLUMNS = ["blocked_ratio", "starved_ratio", "buffer_utilisation", "micro_stoppage_count", "cycle_time_variance"]
_MANUAL_COLUMNS = ["check_pass_rate", "n_distinct_operators", "dominant_operator_share"]


def build_unit_journey_features(
    unit_features: UnitFeatureFrame,
    units: pd.DataFrame,
    readings: pd.DataFrame,
    manual_checks: pd.DataFrame,
    sensor_specs: dict[str, SensorSpecConfig],
    plant: PlantLineConfig,
    soft_store: SoftSensorStore | None = None,
    as_of_sequence: int | None = None,
) -> pd.DataFrame:
    wide = unit_features.wide.reset_index()
    if as_of_sequence is not None:
        wide = wide[wide["sequence"] <= as_of_sequence]

    # Restrict readings/checks to exactly the (unit, station) visits present above —
    # filtering by unit_id alone would leak a mid-journey unit's not-yet-reached stations.
    visited_pairs = wide[["unit_id", "station_id"]].drop_duplicates()
    readings = readings.merge(visited_pairs, on=["unit_id", "station_id"], how="inner")
    manual_checks = manual_checks.merge(visited_pairs, on=["unit_id", "station_id"], how="inner")

    own_deviation = _own_reading_deviation(readings, sensor_specs)
    own_check_fail = manual_checks.groupby("unit_id")["check_pass"].agg(lambda s: float((~s).sum()))
    visit_quality = _classify_visits(wide, plant, soft_store)

    rows = []
    for unit_id, group in wide.groupby("unit_id"):
        unit_deviations = own_deviation.get(unit_id, np.array([]))
        n_fails = float(own_check_fail.get(unit_id, 0.0))
        quality_counts = visit_quality.get(unit_id, {"real": 0, "soft": 0, "blind": 0})
        rows.append(_aggregate_unit(unit_id, group, unit_deviations, n_fails, quality_counts))

    features = pd.DataFrame(rows).set_index("unit_id")
    return features.join(units.set_index("unit_id")[["variant_id", "shift_id", "sequence_number"]], how="left")


def _classify_visits(
    wide: pd.DataFrame, plant: PlantLineConfig, soft_store: SoftSensorStore | None
) -> dict[str, dict[str, int]]:
    # Per-unit provenance actually needs to vary with time/context (a soft estimate's
    # confidence fluctuates), not just reflect the static rich/partial/manual layout —
    # so this classifies each visit using the live soft-sensor store rather than the
    # feature store's real/manual-only provenance table. Estimates are precomputed once
    # per (station, bucket) since many units share the same bucket grid, not per visit.
    estimate_cache: dict[tuple[str, float], bool] = {}

    def is_soft_available(station_id: str, bucket_end_s: float) -> bool:
        if soft_store is None or pd.isna(bucket_end_s):
            return False
        key = (station_id, float(bucket_end_s))
        if key not in estimate_cache:
            estimate_cache[key] = estimate(soft_store, station_id, float(bucket_end_s)) is not None
        return estimate_cache[key]

    manual_station_ids = {s.id for s in plant.stations if s.instrumentation == InstrumentationTier.MANUAL}
    soft_eligible_ids = set(soft_store.archetypes_by_station.keys()) if soft_store is not None else set()

    counts: dict[str, dict[str, int]] = {}
    for row in wide.itertuples(index=False):
        bucket = counts.setdefault(row.unit_id, {"real": 0, "soft": 0, "blind": 0})
        if row.station_id in soft_eligible_ids:
            bucket["soft" if is_soft_available(row.station_id, row.bucket_end_s) else "blind"] += 1
        elif row.station_id in manual_station_ids:
            bucket["blind"] += 1
        else:
            bucket["real"] += 1
    return counts


def _own_reading_deviation(readings: pd.DataFrame, sensor_specs: dict[str, SensorSpecConfig]) -> dict[str, np.ndarray]:
    # A unit's OWN reading at a station — not the station's windowed bucket average
    # across many units — is what the simulator actually shifts when it injects a
    # defect there. This is legitimate, real-time-available telemetry (a real plant
    # knows this specific unit's own weld current, not just a rolling average), and
    # it's far less diluted than the station-bucket features above.
    relevant = readings[readings["sensor_name"].isin(sensor_specs.keys())].copy()
    nominal = relevant["sensor_name"].map(lambda s: sensor_specs[s].nominal)
    relevant["abs_relative_deviation"] = np.abs((relevant["value"] - nominal) / nominal.abs().clip(lower=1e-9))
    return {unit_id: g["abs_relative_deviation"].to_numpy() for unit_id, g in relevant.groupby("unit_id")}


def _aggregate_unit(
    unit_id: str,
    group: pd.DataFrame,
    own_deviations: np.ndarray,
    n_own_check_fails: float,
    quality_counts: dict[str, int],
) -> dict[str, float | str]:
    row: dict[str, float | str] = {"unit_id": unit_id, "n_stations_visited": float(len(group))}
    row["max_own_sensor_deviation"] = _safe_nanmax(own_deviations)
    row["mean_own_sensor_deviation"] = _safe_nanmean(own_deviations)
    row["own_manual_check_fail_count"] = n_own_check_fails

    for col in _PROCESS_COLUMNS:
        values = group[col].to_numpy(dtype=float) if col in group.columns else np.array([])
        row[f"max_{col}"] = _safe_nanmax(values)
        row[f"mean_{col}"] = _safe_nanmean(values)

    for col in _MANUAL_COLUMNS:
        values = group[col].to_numpy(dtype=float) if col in group.columns else np.array([])
        if col == "check_pass_rate":
            row["min_check_pass_rate"] = _safe_nanmin(values)
        else:
            row[f"mean_{col}"] = _safe_nanmean(values)

    total = max(sum(quality_counts.values()), 1)
    row["real_fraction"] = quality_counts["real"] / total
    row["soft_fraction"] = quality_counts["soft"] / total
    row["manual_fraction"] = quality_counts["blind"] / total
    return row


def _safe_nanmax(values: np.ndarray) -> float:
    valid = values[~np.isnan(values)] if values.size else values
    return float(np.max(valid)) if valid.size else np.nan


def _safe_nanmin(values: np.ndarray) -> float:
    valid = values[~np.isnan(values)] if values.size else values
    return float(np.min(valid)) if valid.size else np.nan


def _safe_nanmean(values: np.ndarray) -> float:
    valid = values[~np.isnan(values)] if values.size else values
    return float(np.mean(valid)) if valid.size else np.nan
