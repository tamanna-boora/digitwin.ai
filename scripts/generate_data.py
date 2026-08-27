"""Generate TwinLine's simulated production dataset and write it to CSV."""

import argparse
from pathlib import Path

import pandas as pd

from twinline.config import DEFAULT_MODEL_CONFIG_PATH, DEFAULT_PLANT_CONFIG_PATH, load_app_config
from twinline.schemas import SimulationOutput
from twinline.sim.run_sim import run_simulation

DEFAULT_OUT_DIR = Path("data/simulated")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plant-config", type=Path, default=DEFAULT_PLANT_CONFIG_PATH)
    parser.add_argument("--model-config", type=Path, default=DEFAULT_MODEL_CONFIG_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def _write_output(output: SimulationOutput, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([u.model_dump() for u in output.units]).to_csv(out_dir / "units.csv", index=False)
    pd.DataFrame([r.model_dump() for r in output.readings]).to_csv(out_dir / "readings.csv", index=False)
    pd.DataFrame([m.model_dump() for m in output.manual_checks]).to_csv(
        out_dir / "manual_checks.csv", index=False
    )

    all_defects = pd.DataFrame([d.model_dump() for d in output.defects])
    all_defects["causes"] = all_defects["causes"].apply(lambda causes: ";".join(causes))

    # Split what an inspection gate actually observes from why the defect happened.
    # A real plant never knows origin_station_id/causes/created_time_s/gap_units at
    # inspection time — those only exist because this is a simulator with a God's-eye
    # view. Keeping them mixed into defects.csv would let downstream code cheat.
    inspection_columns = ["unit_id", "defect_type", "detected", "detection_station_id", "detection_time_s"]
    all_defects[inspection_columns].to_csv(out_dir / "defects.csv", index=False)

    ground_truth_columns = ["unit_id", "origin_station_id", "defect_type", "causes", "created_time_s", "gap_units"]
    all_defects[ground_truth_columns].to_csv(out_dir / "ground_truth.csv", index=False)


def _print_summary(output: SimulationOutput) -> None:
    n_units = len(output.units)
    n_defects = len(output.defects)
    detected = [d for d in output.defects if d.detected]
    gaps = [d.gap_units for d in detected if d.gap_units is not None]

    print(f"units simulated:      {n_units}")
    print(f"readings:              {len(output.readings)}")
    print(f"manual checks:         {len(output.manual_checks)}")
    print(f"defects created:       {n_defects} ({n_defects / n_units:.2%} of units)")
    print(f"defects detected:      {len(detected)} ({len(detected) / n_defects:.2%} of defects)")
    if gaps:
        print(f"origin-to-detection gap (units): mean={sum(gaps) / len(gaps):.1f}, "
              f"min={min(gaps):.1f}, max={max(gaps):.1f}")


def main() -> None:
    args = _parse_args()
    app_config = load_app_config(args.plant_config, args.model_config)
    output = run_simulation(app_config.plant, app_config.model)
    _write_output(output, args.out_dir)
    _print_summary(output)
    print(f"wrote dataset to {args.out_dir}/")


if __name__ == "__main__":
    main()
