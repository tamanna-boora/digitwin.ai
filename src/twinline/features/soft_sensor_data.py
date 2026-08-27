"""Builds soft-sensor training/application rows for one archetype.

Training uses leave-one-out among rich members: each rich station's own
target-sensor columns are excluded from its context (X), and its true
target-sensor mean is the label (y) — donors are its OTHER rich siblings.
This is also what makes held-out validation trivial later: a rich station
never sees its own signal, so predicting it IS the "treat it as blind" test.
Application rows for blind stations reuse the exact same builder with every
rich member as a donor and no label.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from twinline.features.station_features import StationFeatureFrame
from twinline.schemas import ArchetypeConfig, PlantLineConfig

SECONDS_PER_HOUR = 3600.0


@dataclass(frozen=True)
class ArchetypeDataset:
    training: pd.DataFrame  # columns: feature_columns + ["y", "station_id", "bucket_end_s"]
    application: dict[str, pd.DataFrame]  # blind station_id -> feature_columns + ["station_id", "bucket_end_s"]
    feature_columns: list[str]
    donor_columns: list[str]


def build_archetype_dataset(
    station_features: StationFeatureFrame, plant: PlantLineConfig, archetype: ArchetypeConfig
) -> ArchetypeDataset:
    wide = station_features.wide
    donor_columns = [f"donor_{donor_id}" for donor_id in archetype.rich_members]

    train_frames = []
    for focal_id in archetype.rich_members:
        donors = [d for d in archetype.rich_members if d != focal_id]
        rows = _build_rows(wide, focal_id, donors, archetype.rich_members, archetype.target_sensor)
        rows["y"] = wide.loc[focal_id, f"{archetype.target_sensor}_mean"].to_numpy()
        rows["station_id"] = focal_id
        train_frames.append(rows)
    training = pd.concat(train_frames, ignore_index=True).dropna(subset=["y"])

    application = {}
    for blind_id in archetype.blind_members:
        rows = _build_rows(wide, blind_id, archetype.rich_members, archetype.rich_members, archetype.target_sensor)
        rows["station_id"] = blind_id
        application[blind_id] = rows

    feature_columns = [c for c in training.columns if c not in ("y", "station_id", "bucket_end_s")]
    return ArchetypeDataset(
        training=training, application=application, feature_columns=feature_columns, donor_columns=donor_columns
    )


def _build_rows(
    wide: pd.DataFrame,
    focal_station_id: str,
    donor_ids: list[str],
    all_rich_ids: list[str],
    target_sensor: str,
) -> pd.DataFrame:
    focal_own = wide.loc[focal_station_id]
    index = focal_own.index.copy()
    data: dict[str, np.ndarray] = {"hour_of_day": (index.to_numpy() / SECONDS_PER_HOUR) % 24.0}

    own_context_cols = [c for c in focal_own.columns if not c.startswith(f"{target_sensor}_")]
    for col in own_context_cols:
        data[f"self_{col}"] = focal_own[col].to_numpy()

    # Donor columns are stable across the whole archetype (one per rich member) so every
    # row — training or application — has the same feature schema even when a particular
    # donor is this row's own excluded station (left as NaN, never leaked).
    for donor_id in all_rich_ids:
        if donor_id in donor_ids:
            donor_ewma = wide.loc[donor_id][f"{target_sensor}_ewma"]
            data[f"donor_{donor_id}"] = donor_ewma.reindex(index).to_numpy()
        else:
            data[f"donor_{donor_id}"] = np.full(len(index), np.nan)

    donor_cols = [f"donor_{d}" for d in all_rich_ids]
    donor_matrix = np.column_stack([data[c] for c in donor_cols])
    data["donor_support_count"] = np.sum(~np.isnan(donor_matrix), axis=1)

    rows = pd.DataFrame(data, index=index)
    rows.index.name = "bucket_end_s"
    return rows.reset_index()
