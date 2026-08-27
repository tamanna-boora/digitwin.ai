"""Typed config for feature engineering (configs/features.yaml)."""

from pydantic import BaseModel, Field


class StationWindowConfig(BaseModel):
    bucket_minutes: float = Field(gt=0.0)
    window_minutes: float = Field(gt=0.0)
    ewma_alpha: float = Field(gt=0.0, le=1.0)
    rolling_slope_lookback_buckets: int = Field(ge=2)


class ProcessStateConfig(BaseModel):
    blocked_cycle_time_frac: float = Field(gt=1.0)
    starved_cycle_time_frac: float = Field(gt=0.0, lt=1.0)
    micro_stoppage_frac: float = Field(gt=1.0)


class FeaturesConfig(BaseModel):
    station_window: StationWindowConfig
    process_state: ProcessStateConfig
