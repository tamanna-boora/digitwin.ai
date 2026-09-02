"""Floor Supervisor: KPIs, the live line map, alarm-budgeted alerts with
acknowledge/dismiss, the containment panel, and current abstentions.
"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import state
from components.alerts import current_station_id
from components.line_map import render_line_map
from components.theme import apply_theme, provenance_badge_html, status_color
from twinline.config import load_detect_config
from twinline.trust.ledger import OperatorAction, list_operator_actions, log_operator_action

st.set_page_config(page_title="Floor Supervisor — TwinLine", page_icon="🏭", layout="wide")
apply_theme()
state.render_sidebar_scenario_selector()

if not state.require_data_and_models():
    st.stop()

cfg = state.load_config()
line = state.load_line_data()
pipeline = state.load_pipeline()
ledger_conn = state.get_ledger_connection()

now_s = state.render_sidebar_clock(line, cfg.plant.takt_seconds)
units_by_id = line.units.set_index("unit_id")

st.title("Floor Supervisor")

# --- current shift + KPIs -------------------------------------------------
recent_units = line.units[line.units["start_time_s"] <= now_s]
current_shift_id = str(recent_units.iloc[-1]["shift_id"]) if not recent_units.empty else str(line.units.iloc[0]["shift_id"])
shift_cfg = next(s for s in cfg.plant.shifts if s.id.value == current_shift_id)
shift_duration_s = (shift_cfg.end_hour - shift_cfg.start_hour) * 3600.0
target_units_per_shift = shift_duration_s / cfg.plant.takt_seconds

units_this_shift = line.units[
    (line.units["shift_id"] == current_shift_id)
    & (line.units["start_time_s"] > now_s - shift_duration_s)
    & (line.units["start_time_s"] <= now_s)
]

wide = pipeline.station_features.wide


def _latest_value(station_id: str, column: str) -> float | None:
    if station_id not in wide.index.get_level_values("station_id"):
        return None
    station_df = wide.loc[station_id]
    eligible = station_df[station_df.index <= now_s]
    if eligible.empty or column not in eligible.columns:
        return None
    value = eligible.iloc[-1][column]
    return None if pd.isna(value) else float(value)


constraint_scores = {s.id: _latest_value(s.id, "buffer_utilisation") for s in cfg.plant.stations}
constraint_scores = {k: v for k, v in constraint_scores.items() if v is not None}
constraint_station = max(constraint_scores, key=constraint_scores.get) if constraint_scores else "n/a"

now_bucket_s = state.bucket_now_for_cache(now_s)
selected_alerts, digest_alerts = state.get_open_alerts(now_bucket_s)

actions_df = list_operator_actions(ledger_conn)
dismissed_ids = set(actions_df.loc[actions_df["action"] == "dismiss", "alert_id"]) if not actions_df.empty else set()
acknowledged_ids = set(actions_df.loc[actions_df["action"] == "acknowledge", "alert_id"]) if not actions_df.empty else set()
# Scoped to the current shift: the alarm budget itself is per shift, so this
# is the number that should read consistently against "N alerts / shift".
open_alerts = [
    a for a in selected_alerts if a.candidate.id not in dismissed_ids and a.candidate.shift_id == current_shift_id
]
digest_this_shift = [a for a in digest_alerts if a.candidate.shift_id == current_shift_id]

at_risk_union: set[str] = set()
for alert in open_alerts:
    at_risk_union.update(alert.risk.unit_ids)

kpi_cols = st.columns(4)
with kpi_cols[0]:
    st.markdown(
        f'<div class="tw-kpi-tile"><div class="tw-kpi-label">Units this shift</div>'
        f'<div class="tw-kpi-value">{len(units_this_shift)}</div>'
        f'<div class="tw-kpi-sub">target {target_units_per_shift:.0f} ({current_shift_id})</div></div>',
        unsafe_allow_html=True,
    )
with kpi_cols[1]:
    st.markdown(
        f'<div class="tw-kpi-tile"><div class="tw-kpi-label">Constraint station</div>'
        f'<div class="tw-kpi-value">{constraint_station}</div>'
        f'<div class="tw-kpi-sub">highest buffer utilisation right now</div></div>',
        unsafe_allow_html=True,
    )
with kpi_cols[2]:
    st.markdown(
        f'<div class="tw-kpi-tile"><div class="tw-kpi-label">Open alerts</div>'
        f'<div class="tw-kpi-value">{len(open_alerts)}</div>'
        f'<div class="tw-kpi-sub">budget {cfg.model.predict.alarm_budget.max_alerts_per_shift}/shift '
        f'&middot; {len(digest_this_shift)} more in the digest ({current_shift_id})</div></div>',
        unsafe_allow_html=True,
    )
with kpi_cols[3]:
    st.markdown(
        f'<div class="tw-kpi-tile"><div class="tw-kpi-label">Units at risk pending inspection</div>'
        f'<div class="tw-kpi-value">{len(at_risk_union)}</div>'
        f'<div class="tw-kpi-sub">across all open alerts</div></div>',
        unsafe_allow_html=True,
    )

st.markdown("")

# --- line map --------------------------------------------------------------
st.markdown('<div class="tw-section-title">Line status</div>', unsafe_allow_html=True)
detect_cfg = load_detect_config()
selected_station = render_line_map(
    cfg.plant, now_s, pipeline.spc_signals, pipeline.anomaly_signals, wide,
    cfg.model.predict.station_hazard, detect_cfg.anomaly, key="floor_supervisor_line_map",
)
if selected_station:
    station = cfg.plant.station_by_id(selected_station)
    st.markdown(
        f'<div class="tw-card"><b>{station.id}</b> — {station.name} · {station.area.value} · '
        f'{station.instrumentation.value} tier</div>',
        unsafe_allow_html=True,
    )

st.markdown("---")

# --- alert feed + containment ----------------------------------------------
col_alerts, col_containment = st.columns([3, 2])

with col_alerts:
    st.markdown('<div class="tw-section-title">Alert feed (alarm-budgeted)</div>', unsafe_allow_html=True)
    if not open_alerts:
        st.info("No open alerts at this point in the run.")
    for alert in open_alerts:
        top = alert.trace.candidates[0]
        rec = alert.recommendation
        acknowledged = alert.candidate.id in acknowledged_ids
        evidence_html = " ".join(
            f'{provenance_badge_html(e.provenance)} <span class="tw-caption">{e.detail}</span><br>'
            for e in top.evidence
        )
        occurrence_badge = (
            f' &nbsp;·&nbsp; <span class="tw-pill" style="background:{status_color("watch")}22;'
            f'color:{status_color("watch")};">&times;{alert.occurrence_count} occurrences</span>'
            if alert.occurrence_count > 1 else ""
        )
        st.markdown(
            f"""
            <div class="tw-card" style="border-left:3px solid {status_color('warn')};">
            <b>{alert.driver.replace('_', ' ').title()}</b> suspected at <b>{top.station_id}</b>
            &nbsp;·&nbsp; confidence {rec.confidence:.0%} &nbsp;·&nbsp; {len(alert.risk.unit_ids)} units at risk
            {occurrence_badge}
            {' &nbsp;·&nbsp; <span class="tw-pill" style="background:' + status_color('ok') + '22;color:' + status_color('ok') + ';">ACKNOWLEDGED</span>' if acknowledged else ''}
            <br><br>
            <span class="tw-caption">WHY</span><br>{evidence_html}
            <br><span class="tw-caption">RECOMMENDED ACTION</span><br>
            {rec.action} &nbsp;<i>— owner: {rec.owner_role.replace('_', ' ')}</i><br>
            <span class="tw-caption">Expected impact: {rec.expected_impact:,.0f} currency &nbsp;·&nbsp;
            monitoring: {rec.monitoring_plan}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        # [2, 3, 3] rather than [1, 2, 3]: at [1, 2, 3] the "Acknowledge" column
        # was too narrow for its own label at this card's typical width and the
        # button text wrapped to two lines.
        btn_cols = st.columns([2, 3, 3])
        with btn_cols[0]:
            if not acknowledged and st.button("Acknowledge", key=f"ack_{alert.candidate.id}"):
                log_operator_action(
                    ledger_conn, OperatorAction(alert.candidate.id, top.station_id, "acknowledge", None, now_s)
                )
                st.rerun()
        with btn_cols[1]:
            reason = st.text_input("Dismiss reason", key=f"reason_{alert.candidate.id}", label_visibility="collapsed",
                                    placeholder="Reason for dismissing...")
        with btn_cols[2]:
            if st.button("Dismiss", key=f"dismiss_{alert.candidate.id}"):
                log_operator_action(
                    ledger_conn,
                    OperatorAction(alert.candidate.id, top.station_id, "dismiss", reason or "no reason given", now_s),
                )
                st.rerun()

with col_containment:
    st.markdown('<div class="tw-section-title">Containment — units at risk</div>', unsafe_allow_html=True)
    if not open_alerts:
        st.info("No active trace.")
    elif not at_risk_union:
        st.caption("No units currently in flight from any open alert's window.")
    else:
        top_alert = max(open_alerts, key=lambda a: a.candidate.probability * a.candidate.units_at_risk)
        st.markdown(
            f'<div class="tw-card">Active trace: <b>{top_alert.driver.replace("_"," ").title()}</b> at '
            f'<b>{top_alert.trace.candidates[0].station_id}</b> — '
            f'{len(at_risk_union)} unit(s) across all open alerts passed through during an anomalous window and '
            f"haven't been inspected yet.</div>",
            unsafe_allow_html=True,
        )
        rows = [
            {"unit_id": uid, "currently_at": current_station_id(cfg, uid, units_by_id, now_s)}
            for uid in sorted(at_risk_union)
        ]
        st.dataframe(pd.DataFrame(rows), width='stretch', hide_index=True, height=280)

st.markdown("---")

# --- abstention strip --------------------------------------------------------
st.markdown('<div class="tw-section-title">Unsure here</div>', unsafe_allow_html=True)
predictions = state.load_unit_predictions()
if predictions.empty:
    st.caption("No prediction data available.")
else:
    final_gate_sequence = max(s.sequence for s in cfg.plant.stations)
    gate_visit_time = predictions["start_time_s"] + final_gate_sequence * cfg.plant.takt_seconds
    near_now = (gate_visit_time > now_s - shift_duration_s) & (gate_visit_time <= now_s)
    abstained_now = predictions[predictions["abstained"] & near_now]
    if abstained_now.empty:
        st.caption("No abstentions among units inspected this shift.")
    else:
        st.markdown(
            f'<div class="tw-card">The system abstained on <b>{len(abstained_now)}</b> unit(s) inspected '
            "this shift rather than force a probability — evidence was too thin or too soft to act on.</div>",
            unsafe_allow_html=True,
        )
        st.dataframe(
            abstained_now.reset_index()[["unit_id", "shift_id", "variant_id", "reason"]],
            width='stretch', hide_index=True,
        )
