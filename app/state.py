"""Cached loaders for config/data/models/ledger, plus the replay clock and
scenario selector every page reads from st.session_state.
"""

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from components.alerts import AlertDetail, build_open_alerts
from components.analytics import traced_drivers
from twinline.config import (
    AppConfig,
    load_app_config,
    load_detect_config,
    load_features_config,
    load_soft_sensors_config,
)
from twinline.data_access import load_defects, load_manual_checks, load_readings, load_units
from twinline.detect.anomaly import run_anomaly_detection
from twinline.detect.spc import run_spc
from twinline.features.soft_sensor_data import build_archetype_dataset
from twinline.features.soft_sensor_validation import ArchetypeValidation, validate_soft_sensors
from twinline.features.soft_sensors import SoftSensorStore, fit_soft_sensor_store
from twinline.features.station_features import StationFeatureFrame
from twinline.features.store import UnitFeatureFrame, build_station_features, build_unit_features
from twinline.predict.calibration import Calibrator, predict_with_abstention
from twinline.predict.defect_risk import DefectRiskModel
from twinline.predict.journey_features import build_unit_journey_features
from twinline.schemas import AnomalySignal, DetectConfig, FeaturesConfig, SPCSignal, SoftSensorsConfig
from twinline.trust.backtest import AblationRow, run_backtest
from twinline.trust.ledger import open_ledger

DATA_DIR = Path("data/simulated")
MODELS_DIR = Path("data/models")
LEDGER_DB_PATH = MODELS_DIR / "trust_ledger.sqlite"

NOW_KEY = "twinline_now_s"
AUTO_ADVANCE_KEY = "twinline_auto_advance"
SCENARIO_KEY = "twinline_active_scenario"
SCENARIO_LIST_KEY = "twinline_scenarios"
BASELINE_SCENARIO = "Baseline (simulated run)"
AUTO_ADVANCE_TICK_SECONDS = 1.5


def data_available() -> bool:
    return (DATA_DIR / "units.csv").exists()


def models_available() -> bool:
    return (MODELS_DIR / "LATEST").exists()


def require_data_and_models() -> bool:
    """Call at the top of every page. Returns False (after showing a friendly
    error) if the simulated dataset or trained models haven't been built yet."""
    if data_available() and models_available():
        return True
    st.error(
        "TwinLine needs simulated data and trained models before this page can render.\n\n"
        "Run `make demo` from the project root (generates data, trains models, runs the backtest), "
        "then reload this page.",
        icon="⚠️",
    )
    return False


@dataclass(frozen=True)
class LineData:
    units: pd.DataFrame
    readings: pd.DataFrame
    manual_checks: pd.DataFrame
    defects: pd.DataFrame


@dataclass(frozen=True)
class IntelligencePipeline:
    station_features: StationFeatureFrame
    unit_features: UnitFeatureFrame
    soft_store: SoftSensorStore
    spc_signals: list[SPCSignal]
    anomaly_signals: list[AnomalySignal]


@dataclass(frozen=True)
class TrainedModels:
    version: str
    risk_model: DefectRiskModel
    calibrator: Calibrator
    metrics: dict


@st.cache_data(show_spinner="Loading plant configuration...")
def load_config() -> AppConfig:
    return load_app_config()


@st.cache_data(show_spinner="Loading feature/detection configuration...")
def load_side_configs() -> tuple[FeaturesConfig, SoftSensorsConfig, DetectConfig]:
    plant = load_config().plant
    return load_features_config(), load_soft_sensors_config(plant=plant), load_detect_config()


@st.cache_data(show_spinner="Loading simulated production data...")
def load_line_data() -> LineData:
    return LineData(
        units=load_units(), readings=load_readings(), manual_checks=load_manual_checks(), defects=load_defects()
    )


@st.cache_resource(show_spinner="Building the intelligence layer (features, soft sensors, detection)...")
def load_pipeline() -> IntelligencePipeline:
    cfg = load_config()
    features_cfg, ss_cfg, detect_cfg = load_side_configs()
    line = load_line_data()

    station_features = build_station_features(
        line.readings, line.manual_checks, line.units, cfg.plant, features_cfg
    )
    unit_features = build_unit_features(line.readings, line.manual_checks, line.units, cfg.plant, station_features)
    soft_store = fit_soft_sensor_store(station_features, cfg.plant, cfg.model, ss_cfg)
    spc_signals = run_spc(line.readings, cfg.plant, features_cfg, detect_cfg, soft_store)
    anomaly_signals = run_anomaly_detection(station_features, cfg.plant, detect_cfg.anomaly, soft_store)

    return IntelligencePipeline(
        station_features=station_features, unit_features=unit_features, soft_store=soft_store,
        spc_signals=spc_signals, anomaly_signals=anomaly_signals,
    )


@st.cache_data(show_spinner="Scoring every unit against the trained model...")
def load_unit_predictions() -> pd.DataFrame:
    """Full-journey calibrated probability + abstention per unit, from the already-
    trained model — computed once and cached, then sliced by the replay clock on
    each page rather than rebuilt per rerun.
    """
    cfg = load_config()
    line = load_line_data()
    pipeline = load_pipeline()
    trained = load_trained_models()
    if trained is None:
        return pd.DataFrame()

    journey = build_unit_journey_features(
        pipeline.unit_features, line.units, line.readings, line.manual_checks, cfg.model.sensor_specs, cfg.plant,
        pipeline.soft_store,
    )
    predictions = predict_with_abstention(
        trained.risk_model, trained.calibrator, journey, cfg.model.predict.calibration
    )
    predictions_df = pd.DataFrame([p.model_dump() for p in predictions]).set_index("unit_id")
    return predictions_df.join(line.units.set_index("unit_id")[["shift_id", "variant_id", "start_time_s"]])


ALERT_CACHE_BUCKET_SECONDS = 300.0


def bucket_now_for_cache(now_s: float) -> float:
    """Round the clock to a coarse bucket so dragging the replay slider or an
    auto-advance tick doesn't force a full alert-rebuild (trace_defect over every
    detected defect) on every single rerun."""
    return round(now_s / ALERT_CACHE_BUCKET_SECONDS) * ALERT_CACHE_BUCKET_SECONDS


@st.cache_data(show_spinner="Tracing open alerts...")
def get_open_alerts(now_bucket_s: float) -> tuple[list[AlertDetail], list[AlertDetail]]:
    cfg = load_config()
    line = load_line_data()
    pipeline = load_pipeline()
    return build_open_alerts(
        cfg, line.units, line.defects, line.manual_checks, pipeline.spc_signals, pipeline.anomaly_signals,
        now_bucket_s,
    )


@st.cache_data(show_spinner="Tracing root cause for every historical defect...")
def get_traced_drivers(now_bucket_s: float) -> pd.DataFrame:
    cfg = load_config()
    line = load_line_data()
    pipeline = load_pipeline()
    return traced_drivers(
        cfg, line.units, line.defects, line.manual_checks, pipeline.spc_signals, pipeline.anomaly_signals,
        now_bucket_s,
    )


@st.cache_data(show_spinner="Replaying the chronological backtest ablation...")
def get_ablation_table() -> list[AblationRow]:
    cfg = load_config()
    line = load_line_data()
    pipeline = load_pipeline()
    return run_backtest(
        pipeline.unit_features, pipeline.station_features, line.units, line.readings, line.manual_checks,
        line.defects, cfg.plant, cfg.model, pipeline.soft_store, pipeline.spc_signals, pipeline.anomaly_signals,
    )


@st.cache_data(show_spinner="Validating soft-sensor interval coverage...")
def get_soft_sensor_validation() -> list[ArchetypeValidation]:
    _, ss_cfg, _ = load_side_configs()
    pipeline = load_pipeline()
    datasets = {
        a.id: build_archetype_dataset(pipeline.station_features, load_config().plant, a) for a in ss_cfg.archetypes
    }
    return validate_soft_sensors(datasets)


@st.cache_resource(show_spinner="Loading trained models...")
def load_trained_models() -> TrainedModels | None:
    if not models_available():
        return None
    version = (MODELS_DIR / "LATEST").read_text(encoding="utf-8").strip()
    version_dir = MODELS_DIR / version

    risk_bundle = joblib.load(version_dir / "defect_risk_model.joblib")
    calibrator = joblib.load(version_dir / "calibrator.joblib")
    metrics = _read_metrics_json(version_dir)
    risk_metrics = metrics.get("defect_risk", {})

    risk_model = DefectRiskModel(
        model=risk_bundle["model"], feature_columns=risk_bundle["feature_columns"],
        train_medians=risk_bundle["train_medians"],
        pr_auc_test=risk_metrics.get("pr_auc_test", float("nan")),
        n_train=risk_metrics.get("n_train", 0), n_test=risk_metrics.get("n_test", 0),
        n_positive_test=risk_metrics.get("n_positive_test", 0),
    )
    return TrainedModels(version=version, risk_model=risk_model, calibrator=calibrator, metrics=metrics)


def _read_metrics_json(version_dir: Path) -> dict:
    metrics_path = version_dir / "metrics.json"
    return json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}


@st.cache_resource(show_spinner=False)
def get_ledger_connection() -> sqlite3.Connection:
    LEDGER_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    return open_ledger(LEDGER_DB_PATH)


def run_horizon_seconds(line: LineData) -> float:
    return max(float(line.readings["timestamp_s"].max()), float(line.manual_checks["timestamp_s"].max()))


def get_now_s() -> float:
    return float(st.session_state.get(NOW_KEY, 0.0))


def render_sidebar_clock(line: LineData, takt_seconds: float) -> float:
    horizon_s = run_horizon_seconds(line)
    if NOW_KEY not in st.session_state:
        st.session_state[NOW_KEY] = horizon_s

    st.sidebar.markdown("### Replay clock")
    auto_advance = st.sidebar.toggle("Auto-advance", key=AUTO_ADVANCE_KEY)

    now_hours = st.sidebar.slider(
        "Now (hours into the run)",
        min_value=0.0,
        max_value=horizon_s / 3600.0,
        value=st.session_state[NOW_KEY] / 3600.0,
        step=takt_seconds / 3600.0,
        key="_replay_clock_hours",
    )
    st.session_state[NOW_KEY] = now_hours * 3600.0
    st.sidebar.caption("Everything on screen reflects only what was knowable at this timestamp.")

    if auto_advance:
        _auto_advance_tick(horizon_s, takt_seconds)

    return st.session_state[NOW_KEY]


@st.fragment(run_every=AUTO_ADVANCE_TICK_SECONDS)
def _auto_advance_tick(horizon_s: float, takt_seconds: float) -> None:
    step_s = takt_seconds * 30
    next_now = st.session_state.get(NOW_KEY, 0.0) + step_s
    st.session_state[NOW_KEY] = next_now if next_now <= horizon_s else 0.0
    st.rerun()


def render_sidebar_scenario_selector() -> str:
    if SCENARIO_LIST_KEY not in st.session_state:
        st.session_state[SCENARIO_LIST_KEY] = [BASELINE_SCENARIO]
    if SCENARIO_KEY not in st.session_state:
        st.session_state[SCENARIO_KEY] = BASELINE_SCENARIO

    st.sidebar.markdown("### Scenario")
    scenario = st.sidebar.selectbox(
        "Active scenario", options=st.session_state[SCENARIO_LIST_KEY],
        index=st.session_state[SCENARIO_LIST_KEY].index(st.session_state[SCENARIO_KEY]),
    )
    st.session_state[SCENARIO_KEY] = scenario
    if scenario != BASELINE_SCENARIO:
        st.sidebar.caption("Non-baseline scenarios are explored on the Scenario Lab page.")
    return scenario


def register_scenario(name: str) -> None:
    if name not in st.session_state.get(SCENARIO_LIST_KEY, [BASELINE_SCENARIO]):
        st.session_state[SCENARIO_LIST_KEY].append(name)
    st.session_state[SCENARIO_KEY] = name
