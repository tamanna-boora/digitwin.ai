"""Append-only SQLite prediction log. log_prediction() only ever inserts;
resolve()/auto_resolve_pass() only ever fill in the outcome columns of an
existing row once it's knowable — the original prediction fields (what was
predicted, when, from what evidence) are never rewritten. scorecard()
reports precision/recall/lead-time/calibration/abstention overall and split
by instrumentation level, because we expect (and should show, not hide)
worse performance at manual stations.
"""

import sqlite3
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from twinline.schemas import PlantLineConfig

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    unit_id TEXT NOT NULL,
    station_id TEXT,
    shift_id TEXT NOT NULL,
    predicted_at_s REAL NOT NULL,
    probability REAL,
    abstained INTEGER NOT NULL,
    abstain_reason TEXT,
    instrumentation_level TEXT NOT NULL,
    alert_selected INTEGER NOT NULL,
    logged_at_s REAL NOT NULL,
    resolved INTEGER NOT NULL DEFAULT 0,
    outcome_detected INTEGER,
    resolved_at_s REAL
);
"""


@dataclass(frozen=True)
class PredictionLogEntry:
    unit_id: str
    station_id: str | None
    shift_id: str
    predicted_at_s: float
    probability: float | None
    abstained: bool
    abstain_reason: str | None
    instrumentation_level: str
    alert_selected: bool
    logged_at_s: float


def open_ledger(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute(_CREATE_TABLE_SQL)
    conn.commit()
    return conn


def log_prediction(conn: sqlite3.Connection, entry: PredictionLogEntry, commit: bool = True) -> int:
    cursor = conn.execute(
        """INSERT INTO predictions
           (unit_id, station_id, shift_id, predicted_at_s, probability, abstained, abstain_reason,
            instrumentation_level, alert_selected, logged_at_s)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            entry.unit_id, entry.station_id, entry.shift_id, entry.predicted_at_s, entry.probability,
            int(entry.abstained), entry.abstain_reason, entry.instrumentation_level, int(entry.alert_selected),
            entry.logged_at_s,
        ),
    )
    if commit:
        conn.commit()
    return int(cursor.lastrowid)


def log_predictions_batch(conn: sqlite3.Connection, entries: list[PredictionLogEntry]) -> None:
    """Same append-only insert as log_prediction(), just committed once for the whole
    batch — thousands of individual commits (one per row) is a real bottleneck at scale.
    """
    conn.executemany(
        """INSERT INTO predictions
           (unit_id, station_id, shift_id, predicted_at_s, probability, abstained, abstain_reason,
            instrumentation_level, alert_selected, logged_at_s)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        [
            (
                e.unit_id, e.station_id, e.shift_id, e.predicted_at_s, e.probability, int(e.abstained),
                e.abstain_reason, e.instrumentation_level, int(e.alert_selected), e.logged_at_s,
            )
            for e in entries
        ],
    )
    conn.commit()


def resolve(conn: sqlite3.Connection, unit_id: str, detected: bool, resolved_at_s: float, commit: bool = True) -> int:
    cursor = conn.execute(
        "UPDATE predictions SET resolved = 1, outcome_detected = ?, resolved_at_s = ? WHERE unit_id = ? AND resolved = 0",
        (int(detected), resolved_at_s, unit_id),
    )
    if commit:
        conn.commit()
    return cursor.rowcount


def auto_resolve_pass(
    conn: sqlite3.Connection, units: pd.DataFrame, defects: pd.DataFrame, plant: PlantLineConfig, as_of_time_s: float
) -> int:
    final_gate_sequence = max(s.sequence for s in plant.stations)
    units_by_id = units.set_index("unit_id")
    detection_time_by_unit = (
        defects.loc[defects["detected"]].set_index("unit_id")["detection_time_s"]
        if defects["detected"].any() else pd.Series(dtype=float)
    )

    unresolved_unit_ids = [
        row[0] for row in conn.execute("SELECT DISTINCT unit_id FROM predictions WHERE resolved = 0").fetchall()
    ]

    n_resolved = 0
    for unit_id in unresolved_unit_ids:
        if unit_id not in units_by_id.index:
            continue
        gate_visit_time_s = units_by_id.loc[unit_id, "start_time_s"] + final_gate_sequence * plant.takt_seconds
        if gate_visit_time_s > as_of_time_s:
            continue
        # A defective unit's outcome becomes knowable at its actual detection time
        # (usually before the final gate, e.g. the paint gate); a clean unit's outcome
        # is only knowable once it clears the final gate — using one blanket timestamp
        # for both would make lead-time meaningless.
        detected = unit_id in detection_time_by_unit.index
        resolved_at_s = float(detection_time_by_unit.loc[unit_id]) if detected else float(gate_visit_time_s)
        n_resolved += resolve(conn, unit_id, detected, resolved_at_s, commit=False)
    conn.commit()
    return n_resolved


def scorecard(conn: sqlite3.Connection) -> pd.DataFrame:
    df = pd.read_sql_query("SELECT * FROM predictions", conn)
    if df.empty:
        return pd.DataFrame(columns=[
            "instrumentation_level", "n", "precision", "recall", "false_alarms", "mean_lead_time_s",
            "calibration_error", "abstention_rate",
        ])

    rows = [_score_group(df, "overall")]
    for level, group in df.groupby("instrumentation_level"):
        rows.append(_score_group(group, level))
    return pd.DataFrame(rows)


def _score_group(df: pd.DataFrame, label: str) -> dict[str, object]:
    n = len(df)
    abstention_rate = float(df["abstained"].mean()) if n else float("nan")

    active = df[df["abstained"] == 0]
    resolved_active = active[active["resolved"] == 1]

    alerted = resolved_active[resolved_active["alert_selected"] == 1]
    not_alerted = resolved_active[resolved_active["alert_selected"] == 0]

    tp = int((alerted["outcome_detected"] == 1).sum())
    fp = int((alerted["outcome_detected"] == 0).sum())
    fn = int((not_alerted["outcome_detected"] == 1).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall = tp / (tp + fn) if (tp + fn) > 0 else float("nan")

    tp_rows = alerted[alerted["outcome_detected"] == 1]
    lead_times = tp_rows["resolved_at_s"] - tp_rows["predicted_at_s"]
    mean_lead_time = float(lead_times.mean()) if len(lead_times) else float("nan")

    cal_error = _calibration_error(resolved_active) if len(resolved_active) else float("nan")

    return {
        "instrumentation_level": label, "n": n, "precision": precision, "recall": recall,
        "false_alarms": fp, "mean_lead_time_s": mean_lead_time, "calibration_error": cal_error,
        "abstention_rate": abstention_rate,
    }


def _calibration_error(df: pd.DataFrame, n_bins: int = 10) -> float:
    probs = df["probability"].to_numpy(dtype=float)
    outcomes = df["outcome_detected"].to_numpy(dtype=float)
    bins = pd.cut(probs, bins=n_bins, labels=False, include_lowest=True)

    gaps, weights = [], []
    for b in pd.unique(bins):
        if pd.isna(b):
            continue
        mask = bins == b
        gaps.append(abs(probs[mask].mean() - outcomes[mask].mean()))
        weights.append(mask.sum())
    return float((pd.Series(gaps) * pd.Series(weights)).sum() / sum(weights)) if gaps else float("nan")
