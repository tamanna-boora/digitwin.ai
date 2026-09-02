"""Run the chronological backtest ablation, populate the trust ledger with a
per-visit scorecard demonstration broken down by instrumentation level, print
both, and write docs/RESULTS.md.
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd

from twinline.config import load_app_config, load_detect_config, load_features_config, load_soft_sensors_config
from twinline.data_access import load_defects, load_manual_checks, load_readings, load_units
from twinline.detect.anomaly import run_anomaly_detection
from twinline.detect.spc import run_spc
from twinline.features.soft_sensors import fit_soft_sensor_store
from twinline.features.store import build_station_features, build_unit_features
from twinline.schemas import InstrumentationTier, ModelConfig, PlantLineConfig
from twinline.trust.backtest import run_backtest
from twinline.trust.ledger import PredictionLogEntry, auto_resolve_pass, log_predictions_batch, open_ledger, scorecard

LEDGER_PATH = Path("data/models/trust_ledger.sqlite")
RESULTS_PATH = Path("docs/RESULTS.md")


def _log_visit_scorecard_predictions(
    conn, units: pd.DataFrame, readings: pd.DataFrame, manual_checks: pd.DataFrame, plant: PlantLineConfig,
    model_cfg: ModelConfig,
) -> None:
    """A simple, deterministic per-(unit, station)-visit risk score — NOT the
    trained defect-risk model — used only to demonstrate the ledger/scorecard
    machinery broken down by real instrumentation tier (rich/partial/manual).
    Outcomes are filled in afterward by auto_resolve_pass() against defects.csv.
    """
    units_by_id = units.set_index("unit_id")
    sensor_specs = model_cfg.sensor_specs
    cfg = model_cfg.trust

    readings_by_station = readings.groupby("station_id")
    checks_by_station = manual_checks.groupby("station_id")
    entries: list[PredictionLogEntry] = []

    for station in plant.stations:
        if station.instrumentation == InstrumentationTier.MANUAL:
            if station.id not in checks_by_station.groups:
                continue
            for row in checks_by_station.get_group(station.id).itertuples(index=False):
                probability = cfg.visit_manual_fail_probability if not row.check_pass else cfg.visit_manual_pass_probability
                entry = _build_entry(row.unit_id, station.id, units_by_id, row.timestamp_s, probability, "manual", cfg)
                if entry is not None:
                    entries.append(entry)
        else:
            if station.id not in readings_by_station.groups:
                continue
            group = readings_by_station.get_group(station.id)
            group = group[group["sensor_name"].isin(sensor_specs.keys())]
            for unit_id, unit_group in group.groupby("unit_id"):
                nominal = unit_group["sensor_name"].map(lambda s: sensor_specs[s].nominal)
                deviation = float((np.abs((unit_group["value"] - nominal) / nominal.abs().clip(lower=1e-9))).max())
                probability = 1.0 - math.exp(-deviation / cfg.visit_deviation_reference)
                timestamp_s = float(unit_group["timestamp_s"].iloc[0])
                entry = _build_entry(
                    unit_id, station.id, units_by_id, timestamp_s, probability, station.instrumentation.value, cfg
                )
                if entry is not None:
                    entries.append(entry)

    log_predictions_batch(conn, entries)
    print(f"  logged {len(entries)} per-visit predictions")


def _build_entry(unit_id, station_id, units_by_id, timestamp_s, probability, level, cfg) -> PredictionLogEntry | None:
    if unit_id not in units_by_id.index:
        return None
    return PredictionLogEntry(
        unit_id=unit_id, station_id=station_id, shift_id=str(units_by_id.loc[unit_id, "shift_id"]),
        predicted_at_s=timestamp_s, probability=probability, abstained=False, abstain_reason=None,
        instrumentation_level=level, alert_selected=probability > cfg.visit_alert_probability_threshold,
        logged_at_s=timestamp_s,
    )


def _dataframe_to_markdown(df: pd.DataFrame) -> str:
    header = "| " + " | ".join(df.columns) + " |"
    separator = "|" + "|".join("---" for _ in df.columns) + "|"
    rows = [
        "| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in row) + " |"
        for row in df.itertuples(index=False)
    ]
    return "\n".join([header, separator, *rows])


def _write_results_md(ablation_rows, score_df: pd.DataFrame, path: Path) -> None:
    lines = ["# TwinLine Backtest Results", "", "Simulated data — see Home page for assumptions.", ""]
    lines += ["## Ablation table", "", "| method | PR-AUC | precision@budget | recall@budget | mean lead time (s) | rework avoided | investigation cost | net benefit |", "|---|---|---|---|---|---|---|---|"]
    for r in ablation_rows:
        lines.append(
            f"| {r.method} | {r.pr_auc:.3f} | {r.precision_at_budget:.3f} | {r.recall_at_budget:.3f} | "
            f"{r.mean_lead_time_s:.0f} | {r.rework_avoided:.0f} | {r.investigation_cost:.0f} | {r.net_benefit:.0f} |"
        )
    lines += ["", "## Scorecard by instrumentation level", "", _dataframe_to_markdown(score_df), ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    cfg = load_app_config()
    features_cfg = load_features_config()
    ss_cfg = load_soft_sensors_config(plant=cfg.plant)
    detect_cfg = load_detect_config()

    units = load_units()
    readings = load_readings()
    manual_checks = load_manual_checks()
    defects = load_defects()

    print("rebuilding feature/detection pipeline...")
    station_features = build_station_features(readings, manual_checks, units, cfg.plant, features_cfg)
    unit_features = build_unit_features(readings, manual_checks, units, cfg.plant, station_features)
    soft_store = fit_soft_sensor_store(station_features, cfg.plant, cfg.model, ss_cfg)
    spc_signals = run_spc(readings, cfg.plant, features_cfg, detect_cfg, soft_store)
    anomaly_signals = run_anomaly_detection(station_features, cfg.plant, detect_cfg.anomaly, soft_store)

    print("running chronological ablation backtest...")
    ablation_rows = run_backtest(
        unit_features, station_features, units, readings, manual_checks, defects, cfg.plant, cfg.model, soft_store,
        spc_signals, anomaly_signals, features_cfg,
    )

    print("\n=== ABLATION TABLE ===")
    header = f"{'method':<22}{'PR-AUC':>8}{'precision':>11}{'recall':>8}{'lead(s)':>10}{'rework_avoided':>16}{'invest_cost':>13}{'net':>10}"
    print(header)
    for r in ablation_rows:
        print(
            f"{r.method:<22}{r.pr_auc:>8.3f}{r.precision_at_budget:>11.3f}{r.recall_at_budget:>8.3f}"
            f"{r.mean_lead_time_s:>10.0f}{r.rework_avoided:>16.0f}{r.investigation_cost:>13.0f}{r.net_benefit:>10.0f}"
        )

    print("\npopulating trust ledger with per-visit scorecard demonstration...")
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    if LEDGER_PATH.exists():
        LEDGER_PATH.unlink()
    conn = open_ledger(LEDGER_PATH)
    _log_visit_scorecard_predictions(conn, units, readings, manual_checks, cfg.plant, cfg.model)
    n_resolved = auto_resolve_pass(conn, units, defects, cfg.plant, as_of_time_s=float(units["start_time_s"].max() + 100_000))
    print(f"  resolved {n_resolved} predictions")

    score_df = scorecard(conn)
    conn.close()

    print("\n=== SCORECARD BY INSTRUMENTATION LEVEL ===")
    print(score_df.to_string(index=False))

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    _write_results_md(ablation_rows, score_df, RESULTS_PATH)
    print(f"\nwrote {RESULTS_PATH}")


if __name__ == "__main__":
    main()
