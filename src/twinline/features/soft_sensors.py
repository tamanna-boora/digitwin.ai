"""Public soft-sensor API: fit_soft_sensor_store() once, then estimate() per
(station, bucket). Confidence degrades multiplicatively — wide interval, a
far rich neighbour, an under-represented variant, or thin donor support each
independently drag it down, so any single weak factor is enough to honestly
kill an estimate. Below model.yaml's soft_sensor_confidence_floor, estimate()
returns None rather than a guess.
"""

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from twinline.features.soft_sensor_data import ArchetypeDataset, build_archetype_dataset
from twinline.features.soft_sensor_model import ArchetypeModel, fit_all_archetypes, predict_quantiles
from twinline.features.station_features import StationFeatureFrame
from twinline.schemas import ArchetypeConfig, ModelConfig, PlantLineConfig, SoftSensorEstimate, SoftSensorsConfig


@dataclass(frozen=True)
class SoftSensorStore:
    archetypes_by_station: dict[str, ArchetypeConfig]
    datasets: dict[str, ArchetypeDataset]
    models: dict[str, ArchetypeModel]
    plant: PlantLineConfig
    model_cfg: ModelConfig
    ss_cfg: SoftSensorsConfig


def fit_soft_sensor_store(
    station_features: StationFeatureFrame, plant: PlantLineConfig, model_cfg: ModelConfig, ss_cfg: SoftSensorsConfig
) -> SoftSensorStore:
    datasets = {a.id: build_archetype_dataset(station_features, plant, a) for a in ss_cfg.archetypes}
    models = fit_all_archetypes(datasets)
    archetypes_by_station = {
        blind_id: a for a in ss_cfg.archetypes for blind_id in a.blind_members
    }
    return SoftSensorStore(
        archetypes_by_station=archetypes_by_station, datasets=datasets, models=models, plant=plant,
        model_cfg=model_cfg, ss_cfg=ss_cfg,
    )


def estimate(
    store: SoftSensorStore, station_id: str, bucket_end_s: float, variant_id: str | None = None
) -> SoftSensorEstimate | None:
    archetype = store.archetypes_by_station.get(station_id)
    if archetype is None:
        raise KeyError(f"{station_id} is not a configured blind station for any soft-sensor archetype")

    app_rows = store.datasets[archetype.id].application[station_id]
    row = app_rows[app_rows["bucket_end_s"] == bucket_end_s]
    if row.empty:
        return None

    model = store.models[archetype.id]
    pred = predict_quantiles(model, row)
    lo, value, hi = float(pred["lo"].iloc[0]), float(pred["value"].iloc[0]), float(pred["hi"].iloc[0])

    contributing = [d for d in archetype.rich_members if not math.isnan(row[f"donor_{d}"].iloc[0])]
    confidence = _score_confidence(store, archetype, station_id, row.iloc[0], lo, hi, contributing, variant_id)

    if confidence < store.model_cfg.soft_sensor_confidence_floor:
        return None

    return SoftSensorEstimate(
        value=value, lo=lo, hi=hi, confidence=confidence,
        method=f"quantile_gbr::{archetype.id}", contributing_stations=contributing,
    )


def confidence_components(
    store: SoftSensorStore,
    archetype: ArchetypeConfig,
    station_id: str,
    row: pd.Series,
    lo: float,
    hi: float,
    contributing: list[str],
    variant_id: str | None,
) -> dict[str, float]:
    """The four named sub-scores behind estimate()'s confidence, exposed so a
    caller can reason about (or interactively perturb, e.g. a 'what if this
    station had a rich neighbour' retrofit scenario) each factor individually."""
    cfg = store.ss_cfg.confidence

    nominal = store.model_cfg.sensor_specs[archetype.target_sensor].nominal
    width_frac = (hi - lo) / max(abs(nominal), 1e-9)
    interval_score = _clip01(1.0 - width_frac / cfg.interval_width_frac_reference)

    if contributing:
        own_sequence = store.plant.station_by_id(station_id).sequence
        distance = min(abs(own_sequence - store.plant.station_by_id(d).sequence) for d in contributing)
        distance_score = _clip01(1.0 - distance / cfg.donor_distance_reference)
    else:
        distance_score = 0.0

    if variant_id is not None:
        variant_frac = float(row.get(f"self_variant_frac_{variant_id}", 0.0))
        variant_score = _clip01(variant_frac / cfg.variant_frac_reference)
    else:
        variant_score = 1.0

    support_score = _clip01(float(row["donor_support_count"]) / cfg.donor_support_reference)

    return {
        "interval_score": interval_score,
        "distance_score": distance_score,
        "variant_score": variant_score,
        "support_score": support_score,
        "confidence": interval_score * distance_score * variant_score * support_score,
    }


def _score_confidence(
    store: SoftSensorStore,
    archetype: ArchetypeConfig,
    station_id: str,
    row: pd.Series,
    lo: float,
    hi: float,
    contributing: list[str],
    variant_id: str | None,
) -> float:
    return confidence_components(store, archetype, station_id, row, lo, hi, contributing, variant_id)["confidence"]


def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def coverage_report(store: SoftSensorStore) -> pd.DataFrame:
    """Per station: real % (directly measured), soft % (estimate cleared the
    confidence floor), blind % (no reading and no usable estimate)."""
    rich_station_ids = {d for a in store.ss_cfg.archetypes for d in a.rich_members}
    rows = []

    for station in sorted(store.plant.stations, key=lambda s: s.sequence):
        if station.id in store.archetypes_by_station:
            archetype = store.archetypes_by_station[station.id]
            app_rows = store.datasets[archetype.id].application[station.id]
            n_buckets = len(app_rows)
            n_soft = sum(
                1
                for bucket_end_s in app_rows["bucket_end_s"]
                if estimate(store, station.id, float(bucket_end_s)) is not None
            )
            real_pct, soft_pct = 0.0, 100.0 * n_soft / n_buckets if n_buckets else 0.0
            blind_pct = 100.0 - soft_pct
        elif station.id in rich_station_ids or station.sensors:
            real_pct, soft_pct, blind_pct = 100.0, 0.0, 0.0
        else:
            real_pct, soft_pct, blind_pct = 0.0, 0.0, 100.0

        rows.append(
            {"station_id": station.id, "instrumentation": station.instrumentation.value,
             "real_pct": real_pct, "soft_pct": soft_pct, "blind_pct": blind_pct}
        )

    return pd.DataFrame(rows)
