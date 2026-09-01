"""Typed config + output record for actions/ (nested under ModelConfig)."""

from pydantic import BaseModel, Field


class ActionsConfig(BaseModel):
    tool_wear_change_window_hours: float = Field(gt=0.0)
    monitoring_units_window: int = Field(gt=0)


class Recommendation(BaseModel):
    driver: str
    controllable_lever: str
    action: str
    expected_impact: float = Field(ge=0.0)
    impact_units: str
    owner_role: str
    confidence: float = Field(ge=0.0, le=1.0)
    monitoring_plan: str
    requires_maintenance_window: bool
