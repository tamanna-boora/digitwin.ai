"""Scenario Lab: what-if controls that re-score against changed conditions —
buffer capacity, station speed, variant mix, a bad supplier batch, and a
sensor retrofit onto a currently-blind station. These are analytical
transforms of already-observed data (recomputing derived metrics under a
changed assumption), not a full pipeline re-simulation — labeled as such.
"""

import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import state
from components.analytics import constraint_history
from components.scenario import (
    adjusted_buffer_utilisation,
    bad_batch_projected_rate,
    reweighted_defect_rate,
    retrofit_confidence,
    total_abstention_count,
)
from components.theme import CATEGORICAL, apply_theme, status_color

st.set_page_config(page_title="Scenario Lab — TwinLine", page_icon="🏭", layout="wide")
apply_theme()
state.render_sidebar_scenario_selector()

if not state.require_data_and_models():
    st.stop()

cfg = state.load_config()
line = state.load_line_data()
pipeline = state.load_pipeline()
horizon_s = state.run_horizon_seconds(line)
wide = pipeline.station_features.wide

st.title("Scenario Lab")
st.caption(
    "These what-ifs recompute derived metrics (buffer utilisation, defect rate, soft-sensor confidence) directly "
    "from the observed run under a changed assumption — a fast analytical approximation, not a full re-simulation."
)

station_ids = [s.id for s in cfg.plant.stations]
defective_ids = set(line.defects.loc[line.defects["detected"], "unit_id"])
baseline_defect_rate = line.units["unit_id"].isin(defective_ids).mean()
baseline_history = constraint_history(wide, station_ids, cfg.plant.takt_seconds, horizon_s)
baseline_constraint = baseline_history.iloc[0]["station_id"] if not baseline_history.empty else "n/a"
baseline_throughput = 3600.0 / cfg.plant.takt_seconds

st.markdown("---")
control_cols = st.columns(3)

with control_cols[0]:
    st.markdown("**Buffer & speed**")
    buffer_station = st.selectbox("Station to modify", station_ids, index=station_ids.index("FA-04") if "FA-04" in station_ids else 0)
    speed_multiplier = st.slider("Speed multiplier (>1 = slower)", 0.7, 1.3, 1.0, step=0.05)
    buffer_slots = st.slider("Buffer slots added after this station", 0, 10, 0)
    damping_per_slot = st.slider("Damping per slot (%)", 0, 25, 12) / 100.0

with control_cols[1]:
    st.markdown("**Variant mix**")
    variant_ids = [v.id for v in cfg.plant.variants]
    mix_pct = {}
    remaining = 100.0
    for i, vid in enumerate(variant_ids):
        if i == len(variant_ids) - 1:
            mix_pct[vid] = remaining
            st.caption(f"{vid}: {remaining:.0f}% (remainder)")
        else:
            default = next((v.mix_ratio * 100 for v in cfg.plant.variants if v.id == vid), 100.0 / len(variant_ids))
            mix_pct[vid] = st.slider(f"{vid} mix (%)", 0.0, 100.0, float(round(default)), step=5.0)
            remaining -= mix_pct[vid]

with control_cols[2]:
    st.markdown("**Bad supplier batch**")
    batch_capable = [s.id for s in cfg.plant.stations if s.can_cause_defect]
    batch_station = st.selectbox("Station receiving the bad batch", batch_capable)
    batch_added_rate = st.slider("Added defect probability during the batch", 0.0, 0.2, 0.05, step=0.01)
    existing_batch_sources = [f for f in cfg.model.fault_sources if f.kind.value == "supplier_batch"]
    default_batch_size = existing_batch_sources[0].batch_size_units if existing_batch_sources else 100
    batch_size = st.slider("Batch size (units)", 20, 300, default_batch_size, step=10)

st.markdown("---")
st.markdown('<div class="tw-section-title">Before / after</div>', unsafe_allow_html=True)

adjusted_series = adjusted_buffer_utilisation(wide, buffer_station, speed_multiplier, buffer_slots, damping_per_slot)
adjusted_mean_buffer = float(adjusted_series.mean()) if not adjusted_series.empty else float("nan")
baseline_mean_buffer = float(wide.loc[buffer_station]["buffer_utilisation"].dropna().mean()) if buffer_station in wide.index.get_level_values("station_id") else float("nan")
throughput_after = baseline_throughput / max(speed_multiplier, 1e-6) if buffer_station == baseline_constraint else baseline_throughput

variant_rate_after = reweighted_defect_rate(line.units, line.defects, mix_pct)
rate_after_batch = bad_batch_projected_rate(variant_rate_after, batch_added_rate, batch_size, len(line.units))

adjusted_history = constraint_history(
    wide, station_ids, cfg.plant.takt_seconds, horizon_s,
    overrides={buffer_station: adjusted_series} if not adjusted_series.empty else None,
)
constraint_after = adjusted_history.iloc[0]["station_id"] if not adjusted_history.empty else baseline_constraint

before_after_cols = st.columns(3)
with before_after_cols[0]:
    st.markdown("**Throughput (units/hour)**")
    fig = go.Figure(go.Bar(x=["Before", "After"], y=[baseline_throughput, throughput_after],
                            marker_color=[CATEGORICAL[6], CATEGORICAL[0]]))
    fig.update_layout(height=260, yaxis_title="Units / hour")
    st.plotly_chart(fig, width='stretch', theme=None)

with before_after_cols[1]:
    st.markdown("**Defect rate**")
    fig = go.Figure(go.Bar(x=["Before", "After"], y=[baseline_defect_rate * 100, rate_after_batch * 100],
                            marker_color=[CATEGORICAL[6], CATEGORICAL[0]]))
    fig.update_layout(height=260, yaxis_title="Defect rate (%)")
    st.plotly_chart(fig, width='stretch', theme=None)

with before_after_cols[2]:
    st.markdown("**Constraint station**")
    st.markdown(
        f'<div class="tw-card" style="text-align:center;padding-top:40px;">'
        f'<span class="tw-caption">Before</span><br><b style="font-size:28px;">{baseline_constraint}</b><br><br>'
        f'<span class="tw-caption">After</span><br><b style="font-size:28px;color:{CATEGORICAL[0]};">{constraint_after}</b>'
        f"</div>",
        unsafe_allow_html=True,
    )

st.markdown(f"**Adjusted buffer utilisation at {buffer_station}:** {baseline_mean_buffer:.2f} -> {adjusted_mean_buffer:.2f} "
            f"(mean over the observed run)")

st.markdown("---")

# --- sensor retrofit ROI ---------------------------------------------------------
st.markdown('<div class="tw-section-title">Sensor retrofit ROI</div>', unsafe_allow_html=True)
blind_stations = sorted(pipeline.soft_store.archetypes_by_station.keys())
retrofit_station = st.selectbox("Blind station to retrofit with a real sensor", blind_stations)

app_rows = pipeline.soft_store.datasets[pipeline.soft_store.archetypes_by_station[retrofit_station].id].application[retrofit_station]
sample_bucket = app_rows["bucket_end_s"].iloc[len(app_rows) // 2] if not app_rows.empty else None

if sample_bucket is None:
    st.caption("No data available for this station.")
else:
    result = retrofit_confidence(pipeline.soft_store, retrofit_station, float(sample_bucket))
    if result is None:
        st.caption("No estimate available at this station.")
    else:
        retro_cols = st.columns(3)
        with retro_cols[0]:
            before_display = f"{result.before_confidence:.0%}" if result.before_confidence is not None else "abstained (below floor)"
            st.markdown(
                f'<div class="tw-kpi-tile"><div class="tw-kpi-label">Soft-sensor confidence — before</div>'
                f'<div class="tw-kpi-value" style="font-size:26px;color:{status_color("watch")};">{before_display}</div></div>',
                unsafe_allow_html=True,
            )
        with retro_cols[1]:
            st.markdown(
                f'<div class="tw-kpi-tile"><div class="tw-kpi-label">Confidence — with a co-located rich sensor</div>'
                f'<div class="tw-kpi-value" style="font-size:26px;color:{status_color("ok")};">{result.after_confidence:.0%}</div></div>',
                unsafe_allow_html=True,
            )
        with retro_cols[2]:
            n_abstained = total_abstention_count(state.load_unit_predictions())
            st.markdown(
                f'<div class="tw-kpi-tile"><div class="tw-kpi-label">Plant-wide abstentions today</div>'
                f'<div class="tw-kpi-value" style="font-size:26px;">{n_abstained}</div>'
                f'<div class="tw-kpi-sub">a real sensor here removes the distance penalty on every future estimate</div></div>',
                unsafe_allow_html=True,
            )

        fig = go.Figure(go.Bar(
            x=["Interval width", "Distance to donor", "Variant support", "Donor support"],
            y=[result.interval_score, result.distance_score_before, result.variant_score, result.support_score],
            marker_color=CATEGORICAL[0],
        ))
        fig.add_hline(y=1.0, line_dash="dash", line_color=status_color("ok"), annotation_text="retrofit target (1.0)")
        fig.update_layout(height=280, yaxis_title="Score (0-1)", yaxis_range=[0, 1.05],
                           title="Confidence factors today — a retrofit fixes 'distance to donor' directly")
        st.plotly_chart(fig, width='stretch', theme=None)
        st.caption(
            "A real sensor at this station eliminates the need to borrow signal from a distant rich neighbour — "
            "the distance-to-donor factor jumps to 1.0, which is what drives the confidence improvement above. "
            "This is the retrofit ROI argument: instrumenting this station converts every future estimate here "
            "from a discounted guess to ground truth."
        )
