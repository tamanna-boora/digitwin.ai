"""Typed simulation outputs — the data that crosses from sim/ into scripts/."""

from pydantic import BaseModel, Field

from twinline.schemas.enums import DefectType, ShiftId


class UnitRecord(BaseModel):
    unit_id: str
    variant_id: str
    shift_id: ShiftId
    sequence_number: int = Field(ge=0)
    start_time_s: float = Field(ge=0.0)


class Reading(BaseModel):
    unit_id: str
    station_id: str
    timestamp_s: float = Field(ge=0.0)
    sensor_name: str
    value: float


class ManualCheck(BaseModel):
    unit_id: str
    station_id: str
    timestamp_s: float = Field(ge=0.0)
    operator_id: str
    check_pass: bool


class DefectRecord(BaseModel):
    unit_id: str
    origin_station_id: str
    defect_type: DefectType
    causes: list[str]
    created_time_s: float = Field(ge=0.0)
    detected: bool
    detection_station_id: str | None = None
    detection_time_s: float | None = Field(default=None, ge=0.0)
    gap_units: float | None = Field(default=None, ge=0.0)


class SimulationOutput(BaseModel):
    units: list[UnitRecord]
    readings: list[Reading]
    manual_checks: list[ManualCheck]
    defects: list[DefectRecord]
