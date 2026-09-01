import sys
sys.path.insert(0, "app")

import pandas as pd

from twinline.config import load_app_config, load_detect_config, load_features_config, load_soft_sensors_config
from twinline.data_access import load_defects, load_manual_checks, load_readings, load_units
from twinline.detect.anomaly import run_anomaly_detection
from twinline.detect.spc import run_spc
from twinline.features.soft_sensors import fit_soft_sensor_store
from twinline.features.store import build_station_features, build_unit_features
from twinline.rootcause.trace import trace_defect, units_at_risk, _visit_time_s
from twinline.schemas import DefectType

cfg = load_app_config()
features_cfg = load_features_config()
ss_cfg = load_soft_sensors_config(plant=cfg.plant)
detect_cfg = load_detect_config()

units = load_units()
readings = load_readings()
manual_checks = load_manual_checks()
defects = load_defects()

station_features = build_station_features(readings, manual_checks, units, cfg.plant, features_cfg)
soft_store = fit_soft_sensor_store(station_features, cfg.plant, cfg.model, ss_cfg)
spc_signals = run_spc(readings, cfg.plant, features_cfg, detect_cfg, soft_store)
anomaly_signals = run_anomaly_detection(station_features, cfg.plant, detect_cfg.anomaly, soft_store)

horizon_s = max(float(readings["timestamp_s"].max()), float(manual_checks["timestamp_s"].max()))
now_s = horizon_s / 2.0
print(f"horizon_s={horizon_s:.0f}  now_s (mid-run)={now_s:.0f}")

detected = defects[defects["detected"] & (defects["detection_time_s"] <= now_s)]
print(f"detected defects as of now_s: {len(detected)}")

units_by_id = units.set_index("unit_id")

n_examined = 0
n_with_candidates = 0
n_nonzero_risk = 0
for row in detected.itertuples(index=False):
    n_examined += 1
    defect_type = DefectType(row.defect_type)
    trace = trace_defect(
        row.unit_id, defect_type, row.detection_station_id, row.detection_time_s, units, defects, manual_checks,
        cfg.plant, spc_signals, anomaly_signals, cfg.model.rootcause,
    )
    if not trace.candidates:
        continue
    n_with_candidates += 1
    top = trace.candidates[0]

    origin_station = cfg.plant.station_by_id(top.station_id)
    gate = cfg.plant.gate_for_defect_type(defect_type)
    gate_sequence = cfg.plant.station_by_id(gate.station_id).sequence
    takt = cfg.plant.takt_seconds

    origin_visit_time_s = units["start_time_s"] + origin_station.sequence * takt
    gate_visit_time_s = units["start_time_s"] + gate_sequence * takt

    in_window = (origin_visit_time_s >= top.window_start_s) & (origin_visit_time_s <= top.window_end_s)
    not_yet_inspected = gate_visit_time_s > now_s

    risk = units_at_risk(units, cfg.plant, top.station_id, top.window_start_s, top.window_end_s, defect_type, now_s)
    if risk.unit_ids:
        n_nonzero_risk += 1

    if n_examined <= 8:
        print(f"\n--- unit {row.unit_id}  defect={defect_type.value}  detected@{row.detection_time_s:.0f} ---")
        print(f"  top candidate station={top.station_id}  window=[{top.window_start_s:.0f}, {top.window_end_s:.0f}]  (span={top.window_end_s - top.window_start_s:.0f}s)")
        print(f"  gate={gate.station_id} (seq {gate_sequence}), origin seq={origin_station.sequence}, takt={takt}")
        print(f"  units in window at origin station: {int(in_window.sum())}")
        print(f"  of those, units whose gate-visit time is still in the future (> now_s={now_s:.0f}): {int((in_window & not_yet_inspected).sum())}")
        if int(in_window.sum()) > 0:
            sample = units.loc[in_window, ["unit_id", "start_time_s"]].head(5).copy()
            sample["origin_visit_time_s"] = sample["start_time_s"] + origin_station.sequence * takt
            sample["gate_visit_time_s"] = sample["start_time_s"] + gate_sequence * takt
            sample["gate_already_passed"] = sample["gate_visit_time_s"] <= now_s
            print(sample.to_string(index=False))
        print(f"  final units_at_risk.unit_ids: {risk.unit_ids}")

print(f"\n\nSUMMARY over {n_examined} detected defects:")
print(f"  traces with >=1 candidate: {n_with_candidates}")
print(f"  alerts with nonzero units_at_risk: {n_nonzero_risk}")
