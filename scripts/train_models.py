"""Train the full TwinLine intelligence stack and persist it to data/models/<version>/."""

import json
import time
from pathlib import Path

import joblib

from twinline.config import load_app_config, load_detect_config, load_features_config, load_soft_sensors_config
from twinline.data_access import load_manual_checks, load_readings, load_units
from twinline.detect.anomaly import run_anomaly_detection
from twinline.detect.spc import run_spc
from twinline.features.soft_sensors import coverage_report, fit_soft_sensor_store
from twinline.features.soft_sensor_data import build_archetype_dataset
from twinline.features.soft_sensor_validation import print_validation_report, validate_soft_sensors
from twinline.features.store import build_station_features, build_unit_features
from twinline.predict.calibration import calibration_error, fit_calibrator
from twinline.predict.defect_risk import build_labeled_dataset, time_based_split, train_defect_risk_model
from twinline.predict.journey_features import build_unit_journey_features
from twinline.rootcause.attribution import driver_importance_report

DEFAULT_MODELS_DIR = Path("data/models")


def main() -> None:
    version = time.strftime("v%Y%m%d_%H%M%S")
    out_dir = DEFAULT_MODELS_DIR / version
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"training version {version} -> {out_dir}/")

    cfg = load_app_config()
    features_cfg = load_features_config()
    ss_cfg = load_soft_sensors_config(plant=cfg.plant)
    detect_cfg = load_detect_config()

    units = load_units()
    readings = load_readings()
    manual_checks = load_manual_checks()

    print("building station/unit features...")
    station_features = build_station_features(readings, manual_checks, units, cfg.plant, features_cfg)
    unit_features = build_unit_features(readings, manual_checks, units, cfg.plant, station_features)

    print("fitting soft-sensor store...")
    soft_store = fit_soft_sensor_store(station_features, cfg.plant, cfg.model, ss_cfg)
    coverage = coverage_report(soft_store)

    archetype_datasets = {a.id: build_archetype_dataset(station_features, cfg.plant, a) for a in ss_cfg.archetypes}
    validation = validate_soft_sensors(archetype_datasets)
    print("\nsoft-sensor held-out validation:")
    print_validation_report(validation)

    print("\nrunning SPC + anomaly detection...")
    spc_signals = run_spc(readings, cfg.plant, features_cfg, detect_cfg, soft_store)
    anomaly_signals = run_anomaly_detection(station_features, cfg.plant, detect_cfg.anomaly, soft_store)
    print(f"  spc signals: {len(spc_signals)}, anomaly signals: {len(anomaly_signals)}")

    print("\nbuilding journey features and training defect-risk model...")
    journey = build_unit_journey_features(
        unit_features, units, readings, manual_checks, cfg.model.sensor_specs, cfg.plant, soft_store
    )
    labeled = build_labeled_dataset(journey)
    split = time_based_split(labeled, cfg.model.predict.defect_risk)
    risk_model = train_defect_risk_model(split, cfg.model.predict.defect_risk.hist_gbc)
    calibrator = fit_calibrator(risk_model, split.calibration)
    cal_error = calibration_error(calibrator, split.test, risk_model)
    print(f"  PR-AUC (test): {risk_model.pr_auc_test:.4f}  calibration error: {cal_error:.4f}")

    driver_report = driver_importance_report(risk_model, split.test)
    print("  driver importance:", {d.driver: round(d.importance_share, 3) for d in driver_report})

    print("\nsaving artifacts...")
    joblib.dump(
        {
            "model": risk_model.model, "feature_columns": risk_model.feature_columns,
            "train_medians": risk_model.train_medians,
        },
        out_dir / "defect_risk_model.joblib",
    )
    joblib.dump(calibrator, out_dir / "calibrator.joblib")
    joblib.dump(soft_store, out_dir / "soft_sensor_store.joblib")

    overall_coverage = (
        sum(r.coverage_fraction * r.n_test for r in validation if r.n_test > 0)
        / sum(r.n_test for r in validation if r.n_test > 0)
    )
    metrics = {
        "version": version,
        "defect_risk": {
            "pr_auc_test": risk_model.pr_auc_test,
            "n_train": risk_model.n_train,
            "n_test": risk_model.n_test,
            "n_positive_test": risk_model.n_positive_test,
            "calibration_error": cal_error,
        },
        "driver_importance": {d.driver: d.importance_share for d in driver_report},
        "soft_sensors": {
            "overall_interval_coverage": overall_coverage,
            "per_archetype": {r.archetype_id: r.coverage_fraction for r in validation},
        },
        "coverage_report": coverage.set_index("station_id")[["real_pct", "soft_pct", "blind_pct"]].to_dict("index"),
        "signal_counts": {"spc": len(spc_signals), "anomaly": len(anomaly_signals)},
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    latest_marker = DEFAULT_MODELS_DIR / "LATEST"
    latest_marker.write_text(version, encoding="utf-8")

    print(f"\ndone. version={version}")


if __name__ == "__main__":
    main()
