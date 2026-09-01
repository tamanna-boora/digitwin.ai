"""Plant Manager: defect trends, throughput vs. takt, Pareto charts,
constraint history, operator/shift/batch comparisons, tool-wear trajectories.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import state
from components.analytics import (
    constraint_history,
    defect_type_to_area,
    pareto_shares,
    tool_wear_trajectory,
    two_proportion_comparison,
)
from components.theme import CATEGORICAL, apply_theme

st.set_page_config(page_title="Plant Manager — TwinLine", page_icon="🏭", layout="wide")
apply_theme()
state.render_sidebar_scenario_selector()

if not state.require_data_and_models():
    st.stop()

cfg = state.load_config()
line = state.load_line_data()
pipeline = state.load_pipeline()
now_s = state.render_sidebar_clock(line, cfg.plant.takt_seconds)
now_bucket_s = state.bucket_now_for_cache(now_s)

st.title("Plant Manager")

units_so_far = line.units[line.units["start_time_s"] <= now_s].copy()
detected = line.defects[line.defects["detected"] & (line.defects["detection_time_s"] <= now_s)].copy()
detected["area"] = detected["defect_type"].map(defect_type_to_area)
units_by_id = units_so_far.set_index("unit_id")
detected_in_scope = detected[detected["unit_id"].isin(units_by_id.index)]
defective_ids = set(detected_in_scope["unit_id"])

# --- defect rate trends ------------------------------------------------------
st.markdown('<div class="tw-section-title">Defect rate trends</div>', unsafe_allow_html=True)
units_so_far["hour"] = (units_so_far["start_time_s"] // 3600.0).astype(int)
units_so_far["is_defective"] = units_so_far["unit_id"].isin(defective_ids)

trend_cols = st.columns(3)
for col, (dim, dim_label) in zip(trend_cols, [("variant_id", "Variant"), ("shift_id", "Shift"), (None, "Zone")]):
    with col:
        fig = go.Figure()
        if dim is not None:
            for i, value in enumerate(sorted(units_so_far[dim].unique())):
                subset = units_so_far[units_so_far[dim] == value]
                by_hour = subset.groupby("hour")["is_defective"].mean() * 100.0
                fig.add_trace(go.Scatter(x=by_hour.index, y=by_hour.values, mode="lines", name=str(value),
                                          line=dict(color=CATEGORICAL[i % len(CATEGORICAL)], width=2)))
        else:
            for i, area in enumerate(sorted(detected_in_scope["area"].unique())):
                area_unit_ids = set(detected_in_scope.loc[detected_in_scope["area"] == area, "unit_id"])
                units_so_far["_flag"] = units_so_far["unit_id"].isin(area_unit_ids)
                by_hour = units_so_far.groupby("hour")["_flag"].mean() * 100.0
                fig.add_trace(go.Scatter(x=by_hour.index, y=by_hour.values, mode="lines", name=area,
                                          line=dict(color=CATEGORICAL[i % len(CATEGORICAL)], width=2)))
        fig.update_layout(title=f"By {dim_label}", height=280, yaxis_title="Defect rate (%)", xaxis_title="Hour")
        st.plotly_chart(fig, width='stretch', theme=None)

st.markdown("---")

# --- throughput vs takt -------------------------------------------------------
st.markdown('<div class="tw-section-title">Throughput vs. takt</div>', unsafe_allow_html=True)
target_per_hour = 3600.0 / cfg.plant.takt_seconds
throughput_by_hour = units_so_far.groupby("hour").size()
fig = go.Figure()
fig.add_trace(go.Scatter(x=throughput_by_hour.index, y=throughput_by_hour.values, mode="lines+markers",
                          name="Actual", line=dict(color=CATEGORICAL[0], width=2)))
fig.add_trace(go.Scatter(x=throughput_by_hour.index, y=[target_per_hour] * len(throughput_by_hour), mode="lines",
                          name="Takt target", line=dict(color=CATEGORICAL[7], width=2, dash="dash")))
fig.update_layout(height=280, yaxis_title="Units / hour", xaxis_title="Hour")
st.plotly_chart(fig, width='stretch', theme=None)

st.markdown("---")

# --- pareto charts -------------------------------------------------------------
pareto_cols = st.columns(2)
with pareto_cols[0]:
    st.markdown('<div class="tw-section-title">Defect type Pareto</div>', unsafe_allow_html=True)
    type_counts = detected_in_scope["defect_type"].value_counts()
    if type_counts.empty:
        st.caption("No defects detected yet at this point in the run.")
    else:
        pareto = pareto_shares(type_counts)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=pareto["category"], y=pareto["share_pct"], name="Share", marker_color=CATEGORICAL[0]))
        fig.add_trace(go.Scatter(x=pareto["category"], y=pareto["cumulative_pct"], mode="lines+markers",
                                  name="Cumulative", line=dict(color=CATEGORICAL[1], width=2)))
        fig.update_layout(height=320, yaxis_title="% of defects", yaxis_range=[0, 105])
        st.plotly_chart(fig, width='stretch', theme=None)

with pareto_cols[1]:
    st.markdown('<div class="tw-section-title">Root-cause driver Pareto</div>', unsafe_allow_html=True)
    drivers_df = state.get_traced_drivers(now_bucket_s)
    driver_counts = drivers_df["driver"].value_counts() if not drivers_df.empty else pd.Series(dtype=int)
    if driver_counts.empty:
        st.caption("No traced drivers yet at this point in the run.")
    else:
        pareto = pareto_shares(driver_counts)
        fig = go.Figure()
        fig.add_trace(go.Bar(x=pareto["category"], y=pareto["share_pct"], name="Share", marker_color=CATEGORICAL[2]))
        fig.add_trace(go.Scatter(x=pareto["category"], y=pareto["cumulative_pct"], mode="lines+markers",
                                  name="Cumulative", line=dict(color=CATEGORICAL[3], width=2)))
        fig.update_layout(height=320, yaxis_title="% of traced defects", yaxis_range=[0, 105])
        st.plotly_chart(fig, width='stretch', theme=None)

st.markdown("---")

# --- constraint history ---------------------------------------------------------
st.markdown('<div class="tw-section-title">Constraint history</div>', unsafe_allow_html=True)
station_ids = [s.id for s in cfg.plant.stations]
history = constraint_history(pipeline.station_features.wide, station_ids, cfg.plant.takt_seconds, now_s)
if history.empty:
    st.caption("Not enough history yet.")
else:
    hist_cols = st.columns(2)
    with hist_cols[0]:
        fig = go.Figure(go.Bar(x=history["station_id"], y=history["times_constraint"], marker_color=CATEGORICAL[0]))
        fig.update_layout(height=300, title="Times each station was the constraint", yaxis_title="Bucket count")
        st.plotly_chart(fig, width='stretch', theme=None)
    with hist_cols[1]:
        fig = go.Figure(go.Bar(x=history["station_id"], y=history["units_lost"], marker_color=CATEGORICAL[1]))
        fig.update_layout(height=300, title="Throughput lost (takt-equivalent units)", yaxis_title="Units lost")
        st.plotly_chart(fig, width='stretch', theme=None)

st.markdown("---")

# --- comparisons with confidence intervals ---------------------------------------
st.markdown('<div class="tw-section-title">Comparisons (95% CI)</div>', unsafe_allow_html=True)
comp_cols = st.columns(2)

with comp_cols[0]:
    st.markdown("**Shift comparison**")
    shift_ids = sorted(units_so_far["shift_id"].unique())
    if len(shift_ids) >= 2:
        a_id, b_id = shift_ids[0], shift_ids[1]
        a_units = units_so_far[units_so_far["shift_id"] == a_id]
        b_units = units_so_far[units_so_far["shift_id"] == b_id]
        comp = two_proportion_comparison(
            a_id, int(a_units["is_defective"].sum()), len(a_units), b_id, int(b_units["is_defective"].sum()),
            len(b_units), cfg.model.rootcause.cohort_ci_z_score,
        )
        if comp.sufficient_evidence:
            fig = go.Figure(go.Scatter(
                x=[comp.rate_a * 100, comp.rate_b * 100], y=[comp.label_a, comp.label_b], mode="markers",
                marker=dict(size=14, color=CATEGORICAL[0]),
                error_x=dict(type="data", array=[cfg.model.rootcause.cohort_ci_z_score * 100 * np.sqrt(comp.rate_a * (1 - comp.rate_a) / comp.n_a),
                                                  cfg.model.rootcause.cohort_ci_z_score * 100 * np.sqrt(comp.rate_b * (1 - comp.rate_b) / comp.n_b)]),
            ))
            fig.update_layout(height=220, xaxis_title="Defect rate (%)")
            st.plotly_chart(fig, width='stretch', theme=None)
            st.caption(f"Difference: {comp.diff:+.2%} (95% CI [{comp.ci_lo:+.2%}, {comp.ci_hi:+.2%}])"
                       + (" — not statistically significant, interval crosses zero." if comp.ci_lo < 0 < comp.ci_hi else ""))
        else:
            st.caption("Insufficient evidence for a shift comparison yet.")
    else:
        st.caption("Only one shift observed so far.")

with comp_cols[1]:
    st.markdown("**Operator check-fail rate (manual stations)**")
    checks_so_far = line.manual_checks[line.manual_checks["timestamp_s"] <= now_s]
    if checks_so_far.empty:
        st.caption("No manual checks yet.")
    else:
        by_operator = checks_so_far.groupby("operator_id").agg(n=("check_pass", "size"), fails=("check_pass", lambda s: int((~s).sum())))
        by_operator["fail_rate_pct"] = 100.0 * by_operator["fails"] / by_operator["n"]
        by_operator = by_operator.sort_values("fail_rate_pct", ascending=False).head(10)
        fig = go.Figure(go.Bar(x=by_operator.index, y=by_operator["fail_rate_pct"], marker_color=CATEGORICAL[4]))
        fig.update_layout(height=280, yaxis_title="Check-fail rate (%)")
        st.plotly_chart(fig, width='stretch', theme=None)

st.markdown("**Batch-position defect rate**")
batch_sources = [f for f in cfg.model.fault_sources if f.kind.value == "supplier_batch"]
batch_size = batch_sources[0].batch_size_units if batch_sources else 100
units_so_far["batch_block"] = units_so_far["sequence_number"] // batch_size
by_batch = units_so_far.groupby("batch_block")["is_defective"].mean() * 100.0
fig = go.Figure(go.Bar(x=[f"block {int(b)}" for b in by_batch.index], y=by_batch.values, marker_color=CATEGORICAL[5]))
fig.update_layout(height=260, yaxis_title="Defect rate (%)",
                   xaxis_title=f"Unit blocks of {batch_size} (the configured supplier lot size)")
st.plotly_chart(fig, width='stretch', theme=None)

st.markdown("---")

# --- tool-wear trajectories ---------------------------------------------------
st.markdown('<div class="tw-section-title">Predicted tool-wear trajectories</div>', unsafe_allow_html=True)
st.caption("Linear extrapolation of each station's own sensor trend — not a read of the simulator's fault model.")
wear_stations = sorted({sid for f in cfg.model.fault_sources if f.kind.value == "tool_wear" for sid in f.station_ids})
horizon_s = state.run_horizon_seconds(line)

wear_cols = st.columns(len(wear_stations)) if wear_stations else []
for col, station_id in zip(wear_cols, wear_stations):
    with col:
        station = cfg.plant.station_by_id(station_id)
        st.markdown(f"**{station_id}** — {station.name}")
        primary_sensor = next((s for s in station.sensors if s != "cycle_time_s"), None)
        if primary_sensor is None:
            st.caption("No sensor at this station — relies on the manual checklist only, so no trend can be fit.")
            continue
        spec = cfg.model.sensor_specs[primary_sensor]
        traj = tool_wear_trajectory(pipeline.station_features.wide, station_id, primary_sensor, spec.nominal,
                                     spec.defect_shift_frac, now_s)
        if traj is None:
            st.caption("Not enough history yet to fit a trend.")
            continue
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=traj["history_x"] / 3600.0, y=traj["history_y"], mode="lines",
                                  name=primary_sensor, line=dict(color=CATEGORICAL[0], width=2)))
        fig.add_hline(y=traj["threshold"], line_dash="dash", line_color=CATEGORICAL[7], annotation_text="threshold")
        fig.update_layout(height=260, yaxis_title=f"{primary_sensor} ({spec.unit})", xaxis_title="Hour")
        st.plotly_chart(fig, width='stretch', theme=None)
        if traj["time_to_threshold_s"] is None or traj["time_to_threshold_s"] > horizon_s:
            st.caption("No meaningful drift detected in the observed window.")
        else:
            st.caption(f"At the observed drift rate: **{traj['time_to_threshold_s'] / 3600.0:.1f}h** to threshold.")
