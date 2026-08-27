"""build_unit_features(): for each unit, and each station it passed through,
the condition of that station AT THE TIME the unit was there. Backed by
merge_asof(direction="backward") against build_station_features's trailing
snapshots, so a unit can only ever see a bucket that had already closed by
the time it arrived — getting this join direction wrong would leak the future.
"""

from dataclasses import dataclass

import pandas as pd

from twinline.features.station_features import StationFeatureFrame
from twinline.schemas import PlantLineConfig

_FORBIDDEN_COLUMNS = {
    "detected", "detection_station_id", "detection_time_s", "defect_type",  # InspectionResult
    "origin_station_id", "causes", "created_time_s", "gap_units",  # GroundTruthCause
}


@dataclass(frozen=True)
class UnitFeatureFrame:
    wide: pd.DataFrame
    provenance: pd.DataFrame


def build_unit_features(
    readings: pd.DataFrame,
    manual_checks: pd.DataFrame,
    units: pd.DataFrame,
    plant: PlantLineConfig,
    station_features: StationFeatureFrame,
) -> UnitFeatureFrame:
    _assert_no_forbidden_columns(readings, manual_checks, units, station_features.wide)

    visits = _unit_station_visits(readings, manual_checks, plant)

    wide_blocks: list[pd.DataFrame] = []
    prov_blocks: list[pd.DataFrame] = []
    for station_id, group in visits.groupby("station_id"):
        station_feat = station_features.wide.loc[station_id].reset_index().sort_values("bucket_end_s")
        station_prov = station_features.provenance.loc[station_id].reset_index().sort_values("bucket_end_s")
        group_sorted = group.sort_values("visit_time_s")

        wide_blocks.append(
            pd.merge_asof(
                group_sorted, station_feat, left_on="visit_time_s", right_on="bucket_end_s", direction="backward"
            )
        )
        prov_blocks.append(
            pd.merge_asof(
                group_sorted[["unit_id", "station_id", "sequence", "visit_time_s"]],
                station_prov,
                left_on="visit_time_s",
                right_on="bucket_end_s",
                direction="backward",
            )
        )

    wide = pd.concat(wide_blocks, ignore_index=True).set_index(["unit_id", "station_id", "sequence"])
    provenance = pd.concat(prov_blocks, ignore_index=True).set_index(["unit_id", "station_id", "sequence"])

    _assert_no_forbidden_columns(wide)
    return UnitFeatureFrame(wide=wide, provenance=provenance)


def _unit_station_visits(readings: pd.DataFrame, manual_checks: pd.DataFrame, plant: PlantLineConfig) -> pd.DataFrame:
    reading_visits = readings.drop_duplicates(subset=["unit_id", "station_id", "timestamp_s"])[
        ["unit_id", "station_id", "timestamp_s"]
    ]
    manual_visits = manual_checks[["unit_id", "station_id", "timestamp_s"]]
    visits = pd.concat([reading_visits, manual_visits], ignore_index=True).rename(
        columns={"timestamp_s": "visit_time_s"}
    )
    sequence_by_station = {s.id: s.sequence for s in plant.stations}
    visits["sequence"] = visits["station_id"].map(sequence_by_station)
    return visits


def _assert_no_forbidden_columns(*frames: pd.DataFrame) -> None:
    for frame in frames:
        leaked = _FORBIDDEN_COLUMNS.intersection(frame.columns)
        if leaked:
            raise AssertionError(
                f"unit features would derive from InspectionResult/GroundTruthCause columns {leaked} — "
                "this is a future-leak, fix the upstream join"
            )
