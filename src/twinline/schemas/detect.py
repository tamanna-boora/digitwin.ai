"""Typed config + output records for detect/ (configs/detect.yaml)."""

from enum import Enum

from pydantic import BaseModel, Field


class Severity(str, Enum):
    WATCH = "watch"
    WARN = "warn"
    CRITICAL = "critical"


class SPCConfig(BaseModel):
    ewma_control_limit_l: float = Field(gt=0.0)
    cusum_k_sigma: float = Field(gt=0.0)
    cusum_h_sigma: float = Field(gt=0.0)
    rule2_window: int = Field(ge=2)
    rule2_hits: int = Field(ge=1)
    rule2_sigma: float = Field(gt=0.0)
    rule3_window: int = Field(ge=2)
    rule3_hits: int = Field(ge=1)
    rule3_sigma: float = Field(gt=0.0)
    rule4_run_length: int = Field(ge=2)


class AnomalyDetectConfig(BaseModel):
    isolation_forest_contamination: float = Field(gt=0.0, lt=0.5)
    modified_z_threshold: float = Field(gt=0.0)
    weight_isolation_forest: float = Field(ge=0.0, le=1.0)
    weight_modified_z: float = Field(ge=0.0, le=1.0)
    severity_watch_threshold: float = Field(ge=0.0, le=1.0)
    severity_warn_threshold: float = Field(ge=0.0, le=1.0)
    severity_critical_threshold: float = Field(ge=0.0, le=1.0)


class DetectConfig(BaseModel):
    spc: SPCConfig
    anomaly: AnomalyDetectConfig


class SPCSignal(BaseModel):
    rule_name: str
    station_id: str
    sensor: str
    severity: Severity
    window_start_s: float
    window_end_s: float
    value: float
    center_line: float
    provenance: str


class AnomalySignal(BaseModel):
    station_id: str
    bucket_end_s: float
    method: str
    score: float = Field(ge=0.0, le=1.0)
    severity: Severity
    contributing_features: list[str]
    confidence_weight: float = Field(ge=0.0, le=1.0)
