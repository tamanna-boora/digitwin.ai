"""Typed config + output records for rootcause/ (nested under ModelConfig)."""

from pydantic import BaseModel, Field


class RootCauseConfig(BaseModel):
    # A signal's window counts as "station was anomalous when the unit was there" if
    # the unit's own visit falls within this many seconds of the signal's window.
    visit_window_tolerance_seconds: float = Field(gt=0.0)
    # A manual check failure with no corroborating SPC/anomaly signal only pins down
    # one instant — this widens its containment window to "recently", instead of the
    # tight visit tolerance above, without ballooning to shift-scale (which would flag
    # hundreds of units per incident and stop being operationally actionable).
    manual_only_window_seconds: float = Field(gt=0.0)
    # Several rule violations with different characteristic durations (a single-point
    # rule1 hit, an 8-bucket rule4 run) can cluster near the same visit; taking the
    # outer envelope of all of them can occasionally stretch across hours. Clamping the
    # total span keeps the containment list operationally actionable regardless.
    max_window_span_seconds: float = Field(gt=0.0)
    area_affinity_bonus: float = Field(ge=0.0)
    co_occurrence_weight: float = Field(ge=0.0)
    # A failed manual check on this exact unit's own visit is near-definitive evidence
    # (the simulator sets check_pass = not defect_created_here) and, for stations with
    # no sensors or soft coverage at all, the ONLY evidence that can ever implicate them.
    manual_check_fail_bonus: float = Field(ge=0.0)
    # Evidence score at which confidence_from_score() reads as ~63% confident —
    # calibrated to one strong direct piece of evidence (e.g. manual_check_fail_bonus).
    confidence_reference_score: float = Field(gt=0.0)
    max_origin_candidates: int = Field(gt=0)
    max_evidence_per_candidate: int = Field(gt=0)
    min_cohort_size: int = Field(gt=0)
    cohort_ci_z_score: float = Field(gt=0.0)


class Evidence(BaseModel):
    signal: str
    rule: str | None = None
    station_id: str
    timestamp_s: float
    provenance: str
    detail: str


class OriginCandidate(BaseModel):
    station_id: str
    score: float = Field(ge=0.0)
    window_start_s: float
    window_end_s: float
    evidence: list[Evidence]


class RootCauseTrace(BaseModel):
    unit_id: str
    defect_type: str
    detection_station_id: str
    detection_time_s: float
    candidates: list[OriginCandidate]


class UnitsAtRisk(BaseModel):
    origin_station_id: str
    window_start_s: float
    window_end_s: float
    as_of_time_s: float
    unit_ids: list[str]


class DriverImportance(BaseModel):
    driver: str
    importance_share: float = Field(ge=0.0, le=1.0)


class CohortComparison(BaseModel):
    station_id: str
    suspected_driver: str
    exposed_n: int = Field(ge=0)
    control_n: int = Field(ge=0)
    exposed_defect_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    control_defect_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    rate_difference: float | None = None
    ci_lo: float | None = None
    ci_hi: float | None = None
    sufficient_evidence: bool
    reason: str | None = None
