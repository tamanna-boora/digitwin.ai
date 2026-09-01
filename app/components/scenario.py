"""Analytical what-if transforms for the Scenario Lab. Deliberately NOT a
full re-simulation — re-running the simulator and the whole feature/detect
pipeline per slider tweak would take seconds to tens of seconds, too slow
for an interactive control. Instead these recompute derived metrics directly
from already-observed data under the changed assumption. The sensor-retrofit
transform is the one exception that's formula-exact, not approximate: it
reuses the real confidence_components() and just asks "what would this
factor be with a co-located rich neighbour" rather than approximating.
"""

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from twinline.features.soft_sensor_model import predict_quantiles
from twinline.features.soft_sensors import SoftSensorStore, confidence_components


def adjusted_buffer_utilisation(
    wide: pd.DataFrame, station_id: str, speed_multiplier: float, buffer_slots: int, damping_per_slot: float
) -> pd.Series:
    """speed_multiplier scales mean cycle time (buffer_utilisation is directly
    proportional to it); each added buffer slot damps how much of the excess-
    over-takt actually shows up as a bottleneck, with diminishing returns."""
    if station_id not in wide.index.get_level_values("station_id"):
        return pd.Series(dtype=float)
    series = wide.loc[station_id]["buffer_utilisation"].dropna()
    scaled = series * speed_multiplier
    excess = np.clip(scaled - 1.0, 0.0, None)
    damping = max(1.0 - buffer_slots * damping_per_slot, 0.0)
    return 1.0 + excess * damping


def reweighted_defect_rate(units: pd.DataFrame, defects: pd.DataFrame, variant_mix_pct: dict[str, float]) -> float:
    defective_ids = set(defects.loc[defects["detected"], "unit_id"])
    flagged = units.assign(is_defective=units["unit_id"].isin(defective_ids))
    by_variant_rate = flagged.groupby("variant_id")["is_defective"].mean()
    total_weight = sum(variant_mix_pct.values()) or 1.0
    return sum(by_variant_rate.get(v, 0.0) * (w / total_weight) for v, w in variant_mix_pct.items())


def bad_batch_projected_rate(baseline_rate: float, added_rate: float, batch_size_units: int, total_units: int) -> float:
    affected_fraction = min(batch_size_units / max(total_units, 1), 1.0)
    return baseline_rate * (1.0 - affected_fraction) + (baseline_rate + added_rate) * affected_fraction


@dataclass(frozen=True)
class RetrofitResult:
    before_confidence: float | None
    after_confidence: float
    interval_score: float
    distance_score_before: float
    support_score: float
    variant_score: float


def retrofit_confidence(store: SoftSensorStore, station_id: str, bucket_end_s: float) -> RetrofitResult | None:
    archetype = store.archetypes_by_station.get(station_id)
    if archetype is None:
        return None
    app_rows = store.datasets[archetype.id].application[station_id]
    row_df = app_rows[app_rows["bucket_end_s"] == bucket_end_s]
    if row_df.empty:
        return None

    model = store.models[archetype.id]
    pred = predict_quantiles(model, row_df)
    lo, hi = float(pred["lo"].iloc[0]), float(pred["hi"].iloc[0])
    row = row_df.iloc[0]
    contributing = [d for d in archetype.rich_members if not math.isnan(row[f"donor_{d}"])]

    before = confidence_components(store, archetype, station_id, row, lo, hi, contributing, None)
    after_confidence = before["interval_score"] * 1.0 * before["variant_score"] * before["support_score"]
    before_confidence = before["confidence"] if before["confidence"] >= store.model_cfg.soft_sensor_confidence_floor else None

    return RetrofitResult(
        before_confidence=before_confidence, after_confidence=after_confidence,
        interval_score=before["interval_score"], distance_score_before=before["distance_score"],
        support_score=before["support_score"], variant_score=before["variant_score"],
    )


def total_abstention_count(predictions: pd.DataFrame) -> int:
    """Every unit visits every station in this line, so there's no clean per-station
    subset of predictions to attribute a retrofit's effect to without re-deriving
    each unit's per-visit real/soft/blind classification. This reports the honest,
    directly-available number instead: how many predictions abstain today, plant-wide."""
    return int(predictions["abstained"].sum()) if not predictions.empty else 0
