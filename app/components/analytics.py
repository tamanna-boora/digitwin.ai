"""Presentation-layer analytics for Plant Manager: Pareto shares, a generic
two-proportion CI (for shift/operator/batch comparisons — distinct from
matched_cohort_comparison, which is specific to one traced incident's
suspected origin station), constraint history, and tool-wear trend
extrapolation. This composes backend outputs; it never re-derives root
cause or defect labels itself.
"""

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from twinline.config import AppConfig
from twinline.rootcause.attribution import infer_suspected_driver
from twinline.rootcause.trace import trace_defect
from twinline.schemas import AnomalySignal, DefectType, SPCSignal


def defect_type_to_area(defect_type: str) -> str:
    from twinline.rootcause.trace import AREA_OWNS_DEFECT_TYPE

    for area, dtype in AREA_OWNS_DEFECT_TYPE.items():
        if dtype.value == defect_type:
            return area
    return "UNKNOWN"


def pareto_shares(counts: pd.Series) -> pd.DataFrame:
    """counts indexed by category -> DataFrame(category, share_pct, cumulative_pct),
    sorted descending — everything indexed to a 0-100% axis so bars and the
    cumulative line can share one scale (never a second, raw-count axis)."""
    ordered = counts.sort_values(ascending=False)
    total = ordered.sum()
    share_pct = 100.0 * ordered / total if total > 0 else ordered * 0.0
    return pd.DataFrame({
        "category": ordered.index, "share_pct": share_pct.to_numpy(), "cumulative_pct": share_pct.cumsum().to_numpy(),
    })


@dataclass(frozen=True)
class ProportionComparison:
    label_a: str
    label_b: str
    rate_a: float
    rate_b: float
    n_a: int
    n_b: int
    diff: float
    ci_lo: float
    ci_hi: float
    sufficient_evidence: bool


def two_proportion_comparison(
    label_a: str, successes_a: int, n_a: int, label_b: str, successes_b: int, n_b: int, z_score: float,
    min_n: int = 10,
) -> ProportionComparison:
    rate_a = successes_a / n_a if n_a > 0 else float("nan")
    rate_b = successes_b / n_b if n_b > 0 else float("nan")
    sufficient = n_a >= min_n and n_b >= min_n
    if not sufficient:
        return ProportionComparison(label_a, label_b, rate_a, rate_b, n_a, n_b, float("nan"), float("nan"), float("nan"), False)

    diff = rate_a - rate_b
    se = math.sqrt(rate_a * (1 - rate_a) / n_a + rate_b * (1 - rate_b) / n_b)
    margin = z_score * se
    return ProportionComparison(label_a, label_b, rate_a, rate_b, n_a, n_b, diff, diff - margin, diff + margin, True)


def constraint_history(
    wide: pd.DataFrame,
    station_ids: list[str],
    takt_seconds: float,
    up_to_s: float,
    overrides: dict[str, pd.Series] | None = None,
) -> pd.DataFrame:
    """Per historical bucket, which station had the highest buffer_utilisation
    (the constraint), and how much throughput (in takt-equivalent units) that
    excess cost — bucket_end_s <= up_to_s only, respecting the replay clock.

    overrides lets a caller substitute one station's buffer_utilisation series
    (e.g. Scenario Lab's what-if-adjusted series) so the constraint is
    recomputed exactly against the other stations' real values, rather than
    approximated by a separate score-nudge formula."""
    overrides = overrides or {}
    rows = []
    for station_id in station_ids:
        if station_id in overrides:
            series = overrides[station_id]
            series = series[series.index <= up_to_s]
        else:
            if station_id not in wide.index.get_level_values("station_id"):
                continue
            df = wide.loc[station_id]
            df = df[df.index <= up_to_s]
            if "buffer_utilisation" not in df.columns:
                continue
            series = df["buffer_utilisation"]
        for bucket_end_s, value in series.items():
            if pd.notna(value):
                rows.append({"bucket_end_s": bucket_end_s, "station_id": station_id, "buffer_utilisation": float(value)})

    if not rows:
        return pd.DataFrame(columns=["station_id", "times_constraint", "units_lost"])

    long_df = pd.DataFrame(rows)
    constraint_per_bucket = long_df.loc[long_df.groupby("bucket_end_s")["buffer_utilisation"].idxmax()]

    summary = constraint_per_bucket.groupby("station_id").agg(
        times_constraint=("station_id", "count"),
        units_lost=("buffer_utilisation", lambda s: float(np.sum(np.clip(s - 1.0, 0.0, None)))),
    ).reset_index()
    return summary.sort_values("times_constraint", ascending=False)


def tool_wear_trajectory(
    wide: pd.DataFrame, station_id: str, sensor: str, nominal: float, defect_shift_frac: float, up_to_s: float
) -> dict | None:
    """Linear-extrapolate the sensor's EWMA trend to estimate time-to-threshold —
    a presentation-layer trend fit, not a read of the simulator's actual fault
    ramp parameters (that would be reading the answer key)."""
    column = f"{sensor}_ewma"
    if station_id not in wide.index.get_level_values("station_id"):
        return None
    df = wide.loc[station_id]
    df = df[df.index <= up_to_s][column].dropna() if column in df.columns else pd.Series(dtype=float)
    if len(df) < 3:
        return None

    x = df.index.to_numpy(dtype=float)
    y = df.to_numpy(dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    threshold = nominal * (1.0 + defect_shift_frac)

    if abs(slope) < 1e-9 or np.sign(threshold - y[-1]) != np.sign(slope):
        return {"slope": float(slope), "threshold": threshold, "history_x": x, "history_y": y, "time_to_threshold_s": None}

    time_to_threshold_s = (threshold - y[-1]) / slope
    return {
        "slope": float(slope), "threshold": threshold, "history_x": x, "history_y": y,
        "time_to_threshold_s": float(time_to_threshold_s) if time_to_threshold_s > 0 else None,
    }


def traced_drivers(
    cfg: AppConfig, units: pd.DataFrame, defects: pd.DataFrame, manual_checks: pd.DataFrame,
    spc_signals: list[SPCSignal], anomaly_signals: list[AnomalySignal], up_to_s: float,
) -> pd.DataFrame:
    """Infers a driver for every detected defect so far (not just the alarm-budgeted
    subset) — for the Pareto-of-drivers chart, which needs the full population."""
    detected = defects[defects["detected"] & (defects["detection_time_s"] <= up_to_s)]
    rows = []
    for row in detected.itertuples(index=False):
        defect_type = DefectType(row.defect_type)
        trace = trace_defect(
            row.unit_id, defect_type, row.detection_station_id, row.detection_time_s, units, defects, manual_checks,
            cfg.plant, spc_signals, anomaly_signals, cfg.model.rootcause,
        )
        driver = infer_suspected_driver(trace.candidates[0]) if trace.candidates else "unknown"
        rows.append({"unit_id": row.unit_id, "defect_type": row.defect_type, "driver": driver})
    return pd.DataFrame(rows)
