"""calibration.py: isotonic-calibrate the raw defect-risk probabilities on a
held-out calibration split (never train or test — see defect_risk.py's
three-way chronological split), then abstain honestly rather than force a
number when the calibrated probability is too close to a coin flip or the
evidence behind it is mostly soft/manual. select_alerts() then budgets
whatever survives down to what a shift can actually act on.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from twinline.predict.defect_risk import DefectRiskModel, predict_defect_risk
from twinline.schemas import AlarmBudgetConfig, AlertCandidate, CalibratedPrediction, CalibrationConfig


@dataclass(frozen=True)
class Calibrator:
    isotonic: IsotonicRegression


def fit_calibrator(risk_model: DefectRiskModel, calibration_rows: pd.DataFrame) -> Calibrator:
    raw = predict_defect_risk(risk_model, calibration_rows)
    y = calibration_rows["y"].to_numpy(dtype=int)
    isotonic = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    isotonic.fit(raw, y)
    return Calibrator(isotonic=isotonic)


def calibration_error(calibrator: Calibrator, rows: pd.DataFrame, risk_model: DefectRiskModel, n_bins: int = 10) -> float:
    """Mean absolute gap between predicted and observed rate, bucketed by calibrated probability."""
    raw = predict_defect_risk(risk_model, rows)
    calibrated = calibrator.isotonic.predict(raw)
    y = rows["y"].to_numpy(dtype=int)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(calibrated, bins) - 1, 0, n_bins - 1)

    gaps, weights = [], []
    for b in range(n_bins):
        mask = bin_idx == b
        if not mask.any():
            continue
        gaps.append(abs(calibrated[mask].mean() - y[mask].mean()))
        weights.append(mask.sum())
    return float(np.average(gaps, weights=weights)) if gaps else float("nan")


def predict_with_abstention(
    risk_model: DefectRiskModel, calibrator: Calibrator, rows: pd.DataFrame, cfg: CalibrationConfig
) -> list[CalibratedPrediction]:
    raw = predict_defect_risk(risk_model, rows)
    calibrated = calibrator.isotonic.predict(raw)

    predictions = []
    for i, unit_id in enumerate(rows.index):
        # manual_fraction, not soft_fraction + manual_fraction: every unit visits the
        # same fixed set of blind stations, so that sum is a constant of the line
        # topology. manual_fraction (visits where even the soft estimate abstained)
        # is what actually varies unit-to-unit and signals genuinely thin evidence.
        blind_fraction = float(rows["manual_fraction"].iloc[i])
        in_band = cfg.uncertainty_band_lo <= calibrated[i] <= cfg.uncertainty_band_hi
        thin_evidence = blind_fraction > cfg.soft_fraction_abstain_threshold

        if in_band or thin_evidence:
            predictions.append(
                CalibratedPrediction(
                    unit_id=unit_id, raw_probability=float(raw[i]), calibrated_probability=None,
                    abstained=True, reason=_abstain_reason(in_band, thin_evidence),
                )
            )
        else:
            predictions.append(
                CalibratedPrediction(
                    unit_id=unit_id, raw_probability=float(raw[i]), calibrated_probability=float(calibrated[i]),
                    abstained=False, reason=None,
                )
            )
    return predictions


def _abstain_reason(in_band: bool, thin_evidence: bool) -> str:
    if in_band and thin_evidence:
        return "calibrated probability is in the uncertainty band and most evidence is soft/manual"
    if in_band:
        return "calibrated probability is in the uncertainty band"
    return "most evidence behind this prediction is soft or manual, below confidence needed to act"


def expected_cost_avoided(candidate: AlertCandidate) -> float:
    return candidate.probability * candidate.units_at_risk * candidate.rework_cost


def select_alerts(candidates: list[AlertCandidate], cfg: AlarmBudgetConfig) -> tuple[list[AlertCandidate], list[AlertCandidate]]:
    by_shift: dict[str, list[AlertCandidate]] = {}
    for candidate in candidates:
        by_shift.setdefault(candidate.shift_id, []).append(candidate)

    selected: list[AlertCandidate] = []
    digest: list[AlertCandidate] = []
    for shift_candidates in by_shift.values():
        ranked = sorted(shift_candidates, key=expected_cost_avoided, reverse=True)
        selected.extend(ranked[: cfg.max_alerts_per_shift])
        digest.extend(ranked[cfg.max_alerts_per_shift :])
    return selected, digest
