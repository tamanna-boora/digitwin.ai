"""Central, guarded access to the simulated dataset on disk (data/simulated/*.csv).

load_ground_truth() is deliberately gated: a real plant never knows which fault
caused a defect or where it originated until root-cause analysis finds it —
that's the God's-eye view the simulator keeps for evaluation only. features/
and predict/ must reason solely from what is observable at prediction time
(sensor readings, manual checks, and inspection *outcomes*), never from
ground_truth.csv. Only rootcause/ and trust/ may legitimately read it.
"""

import inspect
from pathlib import Path

import pandas as pd

DATA_DIR = Path("data/simulated")

_GROUND_TRUTH_FORBIDDEN_PACKAGES = ("twinline.features", "twinline.predict")


def load_units(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    return pd.read_csv(data_dir / "units.csv")


def load_readings(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    return pd.read_csv(data_dir / "readings.csv")


def load_manual_checks(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    return pd.read_csv(data_dir / "manual_checks.csv")


def load_defects(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Inspection-observable outcomes only: unit, defect_type, detected, where/when caught."""
    return pd.read_csv(data_dir / "defects.csv")


def load_ground_truth(data_dir: Path = DATA_DIR) -> pd.DataFrame:
    _assert_ground_truth_access_allowed()
    return pd.read_csv(data_dir / "ground_truth.csv")


def _assert_ground_truth_access_allowed() -> None:
    for record in inspect.stack()[1:]:
        module_name = record.frame.f_globals.get("__name__")
        if not module_name:
            continue
        if any(module_name == pkg or module_name.startswith(pkg + ".") for pkg in _GROUND_TRUTH_FORBIDDEN_PACKAGES):
            raise RuntimeError(
                f"{module_name} attempted to read ground_truth.csv. features/ and predict/ must "
                "never see root-cause labels — only rootcause/ and trust/ may call load_ground_truth()."
            )
