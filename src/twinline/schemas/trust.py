"""Typed config for trust/ (nested under ModelConfig)."""

from pydantic import BaseModel, Field


class TrustConfig(BaseModel):
    investigation_cost_currency: float = Field(gt=0.0)
    # Backtest predicts using only features visited up to this station sequence,
    # not the unit's full journey — this is what makes "predicting with only what
    # was knowable at each tick" honest rather than using post-hoc full-journey data.
    backtest_checkpoint_sequence: int = Field(gt=0)
    # Relative sensor deviation at which the per-visit scorecard probability
    # (used only for the by-instrumentation-level ledger demo) reads as ~63% risky.
    visit_deviation_reference: float = Field(gt=0.0)
    visit_manual_fail_probability: float = Field(gt=0.0, le=1.0)
    visit_manual_pass_probability: float = Field(gt=0.0, le=1.0)
    visit_alert_probability_threshold: float = Field(gt=0.0, le=1.0)
