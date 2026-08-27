"""Typed model of simulation & fault-injection parameters (configs/model.yaml)."""

import re

from pydantic import BaseModel, Field, model_validator

from twinline.schemas.enums import DefectType, FaultKind, ShiftId
from twinline.schemas.plant import STATION_ID_PATTERN


class SensorNoiseConfig(BaseModel):
    rich_std_frac: float = Field(gt=0.0)
    partial_std_frac: float = Field(gt=0.0)


class AmbientHumidityConfig(BaseModel):
    mean_pct: float = Field(ge=0.0, le=100.0)
    amplitude_pct: float = Field(ge=0.0)
    noise_std_pct: float = Field(ge=0.0)
    period_hours: float = Field(gt=0.0)


class SensorSpecConfig(BaseModel):
    """Baseline behavior of one named sensor channel, shared across stations that use it."""

    nominal: float
    unit: str
    defect_shift_frac: float


class FaultSourceConfig(BaseModel):
    id: str
    kind: FaultKind
    station_ids: list[str]
    defect_type: DefectType

    # tool_wear
    onset_hour: float | None = Field(default=None, ge=0.0)
    ramp_per_hour: float | None = Field(default=None, gt=0.0)
    max_added_rate: float | None = Field(default=None, gt=0.0)

    # supplier_batch
    batch_size_units: int | None = Field(default=None, gt=0)
    bad_batch_probability: float | None = Field(default=None, gt=0.0, le=1.0)
    bad_batch_added_rate: float | None = Field(default=None, gt=0.0)

    # operator_variation
    shift_multipliers: dict[ShiftId, float] | None = None
    added_rate: float | None = Field(default=None, gt=0.0)

    # ambient
    humidity_threshold_pct: float | None = Field(default=None, ge=0.0, le=100.0)
    added_rate_above_threshold: float | None = Field(default=None, gt=0.0)

    @model_validator(mode="after")
    def _station_ids_match_pattern(self) -> "FaultSourceConfig":
        for station_id in self.station_ids:
            if not re.match(STATION_ID_PATTERN, station_id):
                raise ValueError(f"fault source {self.id}: invalid station id {station_id}")
        return self

    @model_validator(mode="after")
    def _required_fields_for_kind(self) -> "FaultSourceConfig":
        required_by_kind: dict[FaultKind, tuple[str, ...]] = {
            FaultKind.TOOL_WEAR: ("onset_hour", "ramp_per_hour", "max_added_rate"),
            FaultKind.SUPPLIER_BATCH: (
                "batch_size_units",
                "bad_batch_probability",
                "bad_batch_added_rate",
            ),
            FaultKind.OPERATOR_VARIATION: ("shift_multipliers", "added_rate"),
            FaultKind.AMBIENT: ("humidity_threshold_pct", "added_rate_above_threshold"),
        }
        for field_name in required_by_kind[self.kind]:
            if getattr(self, field_name) is None:
                raise ValueError(f"fault source {self.id} ({self.kind}): missing required field {field_name}")
        return self


class ModelConfig(BaseModel):
    seed: int
    simulation_days: int = Field(gt=0)
    base_defect_rate: float = Field(ge=0.0, le=1.0)
    soft_sensor_confidence_floor: float = Field(ge=0.0, le=1.0)
    sensor_noise: SensorNoiseConfig
    ambient_humidity: AmbientHumidityConfig
    sensor_specs: dict[str, SensorSpecConfig]
    fault_sources: list[FaultSourceConfig]
