"""Typed model of the physical line topology (configs/plant_line_a.yaml)."""

from pydantic import BaseModel, Field, model_validator

from twinline.schemas.enums import AreaCode, DefectType, InstrumentationTier, ShiftId

STATION_ID_PATTERN = r"^(BC|PT|FA)-\d{2}$"


class ShiftConfig(BaseModel):
    id: ShiftId
    start_hour: float = Field(ge=0.0, lt=24.0)
    end_hour: float = Field(gt=0.0, le=24.0)


class VariantConfig(BaseModel):
    id: str
    mix_ratio: float = Field(gt=0.0, le=1.0)
    cycle_time_multiplier: float = Field(gt=0.0)


class StationConfig(BaseModel):
    id: str = Field(pattern=STATION_ID_PATTERN)
    name: str
    area: AreaCode
    sequence: int = Field(ge=1)
    instrumentation: InstrumentationTier
    sensors: list[str]
    can_cause_defect: bool
    is_inspection_gate: bool
    base_cycle_time_s: float = Field(gt=0.0)

    @model_validator(mode="after")
    def _manual_stations_have_no_sensors(self) -> "StationConfig":
        if self.instrumentation == InstrumentationTier.MANUAL and self.sensors:
            raise ValueError(f"manual station {self.id} must not declare sensors")
        if self.instrumentation != InstrumentationTier.MANUAL and not self.sensors:
            raise ValueError(f"non-manual station {self.id} must declare sensors")
        return self


class InspectionGateConfig(BaseModel):
    station_id: str = Field(pattern=STATION_ID_PATTERN)
    name: str
    detects: list[DefectType]
    detection_probability: float = Field(gt=0.0, le=1.0)
    lag_mean_units: float = Field(gt=0.0)
    lag_std_units: float = Field(ge=0.0)
    lag_min_units: float = Field(ge=0.0)
    lag_max_units: float = Field(gt=0.0)

    @model_validator(mode="after")
    def _lag_bounds_are_ordered(self) -> "InspectionGateConfig":
        if not (self.lag_min_units <= self.lag_mean_units <= self.lag_max_units):
            raise ValueError(f"gate {self.station_id}: lag bounds must satisfy min <= mean <= max")
        return self


class PlantLineConfig(BaseModel):
    line_id: str
    takt_seconds: float = Field(gt=0.0)
    shifts: list[ShiftConfig]
    variants: list[VariantConfig]
    stations: list[StationConfig]
    inspection_gates: list[InspectionGateConfig]

    @model_validator(mode="after")
    def _cross_check(self) -> "PlantLineConfig":
        variant_mix_total = sum(v.mix_ratio for v in self.variants)
        if abs(variant_mix_total - 1.0) > 1e-6:
            raise ValueError(f"variant mix_ratio values must sum to 1.0, got {variant_mix_total}")

        sequences = sorted(s.sequence for s in self.stations)
        if sequences != list(range(1, len(self.stations) + 1)):
            raise ValueError("station sequence numbers must be a contiguous 1..N range")

        station_ids = {s.id for s in self.stations}
        if len(station_ids) != len(self.stations):
            raise ValueError("station ids must be unique")

        gate_stations = {g.station_id for g in self.inspection_gates}
        for gate in self.inspection_gates:
            if gate.station_id not in station_ids:
                raise ValueError(f"inspection gate references unknown station {gate.station_id}")
        for station in self.stations:
            if station.id in gate_stations and not station.is_inspection_gate:
                raise ValueError(f"station {station.id} is a gate target but is_inspection_gate=False")
            if station.is_inspection_gate and station.id not in gate_stations:
                raise ValueError(f"station {station.id} marked is_inspection_gate but has no gate config")
        return self

    def station_by_id(self, station_id: str) -> StationConfig:
        for station in self.stations:
            if station.id == station_id:
                return station
        raise KeyError(f"unknown station id {station_id}")

    def gate_for_defect_type(self, defect_type: DefectType) -> InspectionGateConfig:
        for gate in self.inspection_gates:
            if defect_type in gate.detects:
                return gate
        raise KeyError(f"no inspection gate detects {defect_type}")
