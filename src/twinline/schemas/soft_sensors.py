"""Typed config + output record for soft sensors (configs/soft_sensors.yaml)."""

import re

from pydantic import BaseModel, Field, model_validator

from twinline.schemas.plant import STATION_ID_PATTERN


class SoftSensorConfidenceConfig(BaseModel):
    interval_width_frac_reference: float = Field(gt=0.0)
    donor_distance_reference: float = Field(gt=0.0)
    variant_frac_reference: float = Field(gt=0.0, le=1.0)
    donor_support_reference: float = Field(gt=0.0)


class ArchetypeConfig(BaseModel):
    id: str
    target_sensor: str
    rich_members: list[str] = Field(min_length=1)
    blind_members: list[str] = Field(min_length=1)

    @model_validator(mode="after")
    def _no_overlap(self) -> "ArchetypeConfig":
        overlap = set(self.rich_members) & set(self.blind_members)
        if overlap:
            raise ValueError(f"archetype {self.id}: stations {overlap} listed as both rich and blind")
        for station_id in [*self.rich_members, *self.blind_members]:
            if not re.match(STATION_ID_PATTERN, station_id):
                raise ValueError(f"archetype {self.id}: invalid station id {station_id}")
        return self


class SoftSensorsConfig(BaseModel):
    archetypes: list[ArchetypeConfig]
    confidence: SoftSensorConfidenceConfig


class SoftSensorEstimate(BaseModel):
    value: float
    lo: float
    hi: float
    confidence: float = Field(ge=0.0, le=1.0)
    method: str
    contributing_stations: list[str]
