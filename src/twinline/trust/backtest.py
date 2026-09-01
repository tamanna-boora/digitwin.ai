"""run_backtest(): replay the run using only features knowable at a fixed
mid-journey checkpoint (trust.backtest_checkpoint_sequence stations in, not
the unit's full journey), and score four methods against the same
chronological test split defect_risk.py already uses:

  rules-only        — worst SPC/anomaly severity seen at any visited station
  ML-only           — HistGBC trained on journey features WITHOUT soft sensors
  hybrid            — average of rules-only and ML-only
  hybrid+soft       — average of rules-only and an HistGBC trained WITH soft sensors

Each is scored at the same alarm budget so the table compares apples to
apples, plus a cost model of rework avoided (true positives) against the
cost of investigating false alarms.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from twinline.features.soft_sensors import SoftSensorStore
from twinline.features.station_features import StationFeatureFrame
from twinline.features.store import UnitFeatureFrame
from twinline.predict.calibration import fit_calibrator, predict_with_abstention
from twinline.predict.defect_risk import build_labeled_dataset, time_based_split, train_defect_risk_model
from twinline.predict.journey_features import build_unit_journey_features
from twinline.schemas import AnomalySignal, ModelConfig, PlantLineConfig, SPCSignal

_SEVERITY_SCORE = {"watch": 0.3, "warn": 0.6, "critical": 1.0}
_SIGNAL_TOLERANCE_S = 1200.0


@dataclass(frozen=True)
class AblationRow:
    method: str
    pr_auc: float
    precision_at_budget: float
    recall_at_budget: float
    mean_lead_time_s: float
    rework_avoided: float
    investigation_cost: float
    net_benefit: float


def _rules_based_scores(
    units: pd.DataFrame, plant: PlantLineConfig, checkpoint_sequence: int,
    spc_signals: list[SPCSignal], anomaly_signals: list[AnomalySignal],
) -> pd.Series:
    signals_by_station: dict[str, list[tuple[float, float]]] = {}
    for s in spc_signals:
        signals_by_station.setdefault(s.station_id, []).append((s.window_end_s, _SEVERITY_SCORE[s.severity.value]))
    for a in anomaly_signals:
        signals_by_station.setdefault(a.station_id, []).append(
            (a.bucket_end_s, _SEVERITY_SCORE[a.severity.value] * a.confidence_weight)
        )

    stations = [s for s in plant.stations if s.sequence <= checkpoint_sequence]
    takt = plant.takt_seconds

    scores = np.zeros(len(units))
    for station in stations:
        station_signals = signals_by_station.get(station.id, [])
        if not station_signals:
            continue
        visit_times = units["start_time_s"].to_numpy() + station.sequence * takt
        for i, visit_time in enumerate(visit_times):
            best = max(
                (sev for t, sev in station_signals if abs(t - visit_time) <= _SIGNAL_TOLERANCE_S), default=0.0
            )
            scores[i] = max(scores[i], best)
    return pd.Series(scores, index=units["unit_id"].to_numpy())


def run_backtest(
    unit_features: UnitFeatureFrame,
    station_features: StationFeatureFrame,
    units: pd.DataFrame,
    readings: pd.DataFrame,
    manual_checks: pd.DataFrame,
    defects: pd.DataFrame,
    plant: PlantLineConfig,
    model_cfg: ModelConfig,
    soft_store: SoftSensorStore,
    spc_signals: list[SPCSignal],
    anomaly_signals: list[AnomalySignal],
) -> list[AblationRow]:
    checkpoint = model_cfg.trust.backtest_checkpoint_sequence

    journey_no_soft = build_unit_journey_features(
        unit_features, units, readings, manual_checks, model_cfg.sensor_specs, plant, None, checkpoint
    )
    journey_with_soft = build_unit_journey_features(
        unit_features, units, readings, manual_checks, model_cfg.sensor_specs, plant, soft_store, checkpoint
    )

    labeled_no_soft = build_labeled_dataset(journey_no_soft)
    labeled_with_soft = build_labeled_dataset(journey_with_soft)

    split_no_soft = time_based_split(labeled_no_soft, model_cfg.predict.defect_risk)
    split_with_soft = time_based_split(labeled_with_soft, model_cfg.predict.defect_risk)

    model_no_soft = train_defect_risk_model(split_no_soft, model_cfg.predict.defect_risk.hist_gbc)
    model_with_soft = train_defect_risk_model(split_with_soft, model_cfg.predict.defect_risk.hist_gbc)

    calibrator_no_soft = fit_calibrator(model_no_soft, split_no_soft.calibration)
    calibrator_with_soft = fit_calibrator(model_with_soft, split_with_soft.calibration)

    ml_only_preds = predict_with_abstention(model_no_soft, calibrator_no_soft, split_no_soft.test, model_cfg.predict.calibration)
    hybrid_soft_preds = predict_with_abstention(
        model_with_soft, calibrator_with_soft, split_with_soft.test, model_cfg.predict.calibration
    )

    ml_only_scores = pd.Series(
        {p.unit_id: (p.calibrated_probability or 0.0) for p in ml_only_preds}
    ).reindex(split_no_soft.test.index).fillna(0.0)
    hybrid_soft_ml_scores = pd.Series(
        {p.unit_id: (p.calibrated_probability or 0.0) for p in hybrid_soft_preds}
    ).reindex(split_with_soft.test.index).fillna(0.0)

    rules_scores_all = _rules_based_scores(units, plant, checkpoint, spc_signals, anomaly_signals)
    rules_scores = rules_scores_all.reindex(split_no_soft.test.index).fillna(0.0)

    y_test = split_no_soft.test["y"]
    detection_time_by_unit = (
        defects.loc[defects["detected"]].set_index("unit_id")["detection_time_s"]
        if defects["detected"].any() else pd.Series(dtype=float)
    )
    checkpoint_visit_time = units.set_index("unit_id")["start_time_s"] + checkpoint * plant.takt_seconds

    methods = {
        "rules-only": rules_scores,
        "ML-only": ml_only_scores,
        "hybrid": (rules_scores + ml_only_scores) / 2.0,
        "hybrid+soft-sensors": (rules_scores + hybrid_soft_ml_scores) / 2.0,
    }

    rework_cost = model_cfg.predict.alarm_budget.rework_cost_currency
    investigation_cost = model_cfg.trust.investigation_cost_currency
    budget_per_shift = model_cfg.predict.alarm_budget.max_alerts_per_shift

    rows = []
    for method_name, scores in methods.items():
        rows.append(
            _score_method(
                method_name, scores, y_test, detection_time_by_unit, checkpoint_visit_time, units,
                budget_per_shift, rework_cost, investigation_cost,
            )
        )
    return rows


def _score_method(
    method_name: str,
    scores: pd.Series,
    y_test: pd.Series,
    detection_time_by_unit: pd.Series,
    checkpoint_visit_time: pd.Series,
    units: pd.DataFrame,
    budget_per_shift: int,
    rework_cost: float,
    investigation_cost: float,
) -> AblationRow:
    pr_auc = float(average_precision_score(y_test, scores)) if y_test.sum() > 0 else float("nan")

    shift_by_unit = units.set_index("unit_id")["shift_id"]
    ranked = pd.DataFrame({"score": scores, "shift_id": shift_by_unit.reindex(scores.index)})
    selected_ids = (
        ranked.groupby("shift_id", group_keys=False)
        .apply(lambda g: g.sort_values("score", ascending=False).head(budget_per_shift), include_groups=False)
        .index
    )

    tp = int(y_test.loc[selected_ids].sum())
    fp = len(selected_ids) - tp
    fn = int(y_test.sum()) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")

    tp_ids = [uid for uid in selected_ids if y_test.loc[uid] == 1 and uid in detection_time_by_unit.index]
    lead_times = [detection_time_by_unit.loc[uid] - checkpoint_visit_time.loc[uid] for uid in tp_ids]
    mean_lead_time = float(np.mean(lead_times)) if lead_times else float("nan")

    rework_avoided = tp * rework_cost
    cost_of_investigating = fp * investigation_cost

    return AblationRow(
        method=method_name, pr_auc=pr_auc, precision_at_budget=precision, recall_at_budget=recall,
        mean_lead_time_s=mean_lead_time, rework_avoided=rework_avoided, investigation_cost=cost_of_investigating,
        net_benefit=rework_avoided - cost_of_investigating,
    )
