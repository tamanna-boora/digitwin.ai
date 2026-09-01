"""Typed config for predict/ (nested under ModelConfig, configs/model.yaml)."""

from pydantic import BaseModel, Field, model_validator


class HistGBCConfig(BaseModel):
    max_iter: int = Field(gt=0)
    max_depth: int = Field(gt=0)
    learning_rate: float = Field(gt=0.0)
    min_samples_leaf: int = Field(gt=0)
    l2_regularization: float = Field(ge=0.0)


class DefectRiskConfig(BaseModel):
    train_fraction: float = Field(gt=0.0, lt=1.0)
    calibration_fraction: float = Field(gt=0.0, lt=1.0)
    hist_gbc: HistGBCConfig

    @model_validator(mode="after")
    def _fractions_leave_room_for_test(self) -> "DefectRiskConfig":
        if self.train_fraction + self.calibration_fraction >= 1.0:
            raise ValueError("train_fraction + calibration_fraction must leave room for a test split")
        return self


class CalibrationConfig(BaseModel):
    uncertainty_band_lo: float = Field(ge=0.0, le=1.0)
    uncertainty_band_hi: float = Field(ge=0.0, le=1.0)
    soft_fraction_abstain_threshold: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _band_is_ordered(self) -> "CalibrationConfig":
        if self.uncertainty_band_lo >= self.uncertainty_band_hi:
            raise ValueError("uncertainty_band_lo must be < uncertainty_band_hi")
        return self


class AlarmBudgetConfig(BaseModel):
    max_alerts_per_shift: int = Field(gt=0)
    rework_cost_currency: float = Field(gt=0.0)


class StationHazardConfig(BaseModel):
    lookback_seconds: float = Field(gt=0.0)
    half_life_seconds: float = Field(gt=0.0)
    severity_weight_watch: float = Field(ge=0.0, le=1.0)
    severity_weight_warn: float = Field(ge=0.0, le=1.0)
    severity_weight_critical: float = Field(ge=0.0, le=1.0)


class PredictConfig(BaseModel):
    defect_risk: DefectRiskConfig
    calibration: CalibrationConfig
    alarm_budget: AlarmBudgetConfig
    station_hazard: StationHazardConfig


class CalibratedPrediction(BaseModel):
    unit_id: str
    raw_probability: float = Field(ge=0.0, le=1.0)
    calibrated_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    abstained: bool
    reason: str | None = None


class AlertCandidate(BaseModel):
    id: str
    station_id: str | None = None
    shift_id: str
    probability: float = Field(ge=0.0, le=1.0)
    units_at_risk: int = Field(gt=0)
    rework_cost: float = Field(gt=0.0)
    reason: str
