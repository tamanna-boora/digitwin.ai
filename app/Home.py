"""TwinLine — entry page: what this is, our assumptions, the architecture, and
navigation into the five operating views.
"""

import plotly.graph_objects as go
import streamlit as st

import state
from components.theme import CATEGORICAL, INK_MUTED, INK_PRIMARY, INK_SECONDARY, SURFACE, apply_theme

st.set_page_config(page_title="TwinLine", page_icon="🏭", layout="wide")
apply_theme()

state.render_sidebar_scenario_selector()

st.title("TwinLine")
st.markdown(
    "A digital twin for a mixed-model vehicle assembly line — where bottlenecks form and "
    "which units are likely to fail inspection, before they do."
)
st.warning(
    "**All data on every page of this app is simulated.** Nothing here reads from or writes "
    "to a real production system — this is a hackathon prototype running entirely in shadow mode.",
    icon="🧪",
)

if not state.require_data_and_models():
    st.stop()

cfg = state.load_config()

st.markdown("---")
col_left, col_right = st.columns([3, 2])

with col_left:
    st.markdown('<div class="tw-section-title">What this is</div>', unsafe_allow_html=True)
    n_rich = sum(1 for s in cfg.plant.stations if s.instrumentation.value == "rich")
    n_partial = sum(1 for s in cfg.plant.stations if s.instrumentation.value == "partial")
    n_manual = sum(1 for s in cfg.plant.stations if s.instrumentation.value == "manual")
    n_gates = len(cfg.plant.inspection_gates)
    variant_names = ", ".join(v.id for v in cfg.plant.variants)

    st.markdown(
        f"""
        <div class="tw-card">
        TwinLine simulates <b>{len(cfg.plant.stations)} stations</b> across body construction, paint,
        and final assembly ({cfg.plant.line_id}), running {variant_names} at a {cfg.plant.takt_seconds:g}s
        takt across {len(cfg.plant.shifts)} shifts over {cfg.model.simulation_days} simulated production days.
        Instrumentation is deliberately uneven: {n_rich} stations are richly sensored, {n_partial} report a
        partial signal, and {n_manual} rely on a manual checklist only — the twin has to stay useful there too.
        <br><br>
        Defects are only observable at {n_gates} inspection gates, well downstream of where they're created —
        so the system traces backwards from a caught defect to its likely origin, lists the units still in
        flight from that window, and recommends an advisory action to a named owner. Nothing here writes to
        the line.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<div class="tw-section-title">Assumptions we operate under</div>', unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="tw-card">
        <ul>
        <li><b>Uneven sensor coverage</b> — blind stations get a soft-sensor estimate with honest,
        degrading confidence, or the system abstains rather than guess.</li>
        <li><b>Delayed defect discovery</b> — a defect made upstream surfaces only at the next gate;
        root cause is traced backwards and units still in flight are listed.</li>
        <li><b>Multi-causal root causes</b> — tool wear, supplier batch, operator variation, and ambient
        conditions can overlap at the same station.</li>
        <li><b>False alarms destroy trust</b> — a hard alarm budget of
        {cfg.model.predict.alarm_budget.max_alerts_per_shift} per shift; the system abstains on
        thin evidence instead of forcing a number.</li>
        <li><b>Shadow mode only</b> — every output is advisory, addressed to a named human owner.
        No writes to the line.</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

def _build_architecture_figure() -> go.Figure:
    stages = [
        ("Simulator", "sim/"),
        ("Raw data", "data/simulated"),
        ("Features +\nsoft sensors", "features/"),
        ("Detect\n(SPC, anomaly)", "detect/"),
        ("Predict +\ncalibrate", "predict/"),
        ("Root cause +\ncontainment", "rootcause/"),
        ("Actions", "actions/"),
        ("Trust ledger +\nbacktest", "trust/"),
    ]
    fig = go.Figure()
    box_w, gap = 1.0, 0.35
    for i, (label, subtitle) in enumerate(stages):
        x0 = i * (box_w + gap)
        x1 = x0 + box_w
        color = CATEGORICAL[i % len(CATEGORICAL)]
        fig.add_shape(
            type="rect", x0=x0, x1=x1, y0=0, y1=1, fillcolor=f"{color}22",
            line=dict(color=color, width=2), layer="below",
        )
        fig.add_annotation(x=(x0 + x1) / 2, y=0.62, text=label.replace("\n", "<br>"), showarrow=False,
                            font=dict(size=11, color=INK_PRIMARY), align="center")
        fig.add_annotation(x=(x0 + x1) / 2, y=0.22, text=subtitle, showarrow=False,
                            font=dict(size=9, color=INK_MUTED), align="center")
        if i < len(stages) - 1:
            fig.add_annotation(
                x=x1 + gap, y=0.5, ax=x1, ay=0.5, xref="x", yref="y", axref="x", ayref="y",
                showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=1.5, arrowcolor=INK_SECONDARY,
            )
    fig.update_layout(
        height=360, showlegend=False, plot_bgcolor=SURFACE, paper_bgcolor=SURFACE,
        xaxis=dict(visible=False, range=[-0.2, len(stages) * (box_w + gap)], fixedrange=True),
        yaxis=dict(visible=False, range=[-0.1, 1.1], fixedrange=True),
        margin=dict(l=10, r=10, t=10, b=10),
    )
    return fig


with col_right:
    st.markdown('<div class="tw-section-title">Architecture</div>', unsafe_allow_html=True)
    st.plotly_chart(_build_architecture_figure(), width='stretch', theme=None)
    st.caption("Each stage reads only what the one before it produced — features/ and predict/ never see "
               "ground-truth cause labels, only rootcause/ and trust/ do, and only for evaluation.")

st.markdown("---")
st.markdown('<div class="tw-section-title">Go to</div>', unsafe_allow_html=True)

nav_cards = [
    ("1  Floor Supervisor", "Live line map, alarm-budgeted alerts, and the units-at-risk containment list.", "pages/1_Floor_Supervisor.py"),
    ("2  Plant Manager", "Defect trends, throughput vs. takt, constraint history, driver comparisons.", "pages/2_Plant_Manager.py"),
    ("3  Leadership", "The business case: rework/scrap avoided, payback, and a 3-year projection.", "pages/3_Leadership.py"),
    ("4  Trust & Validation", "Scorecard, reliability diagram, the ablation table, coverage by instrumentation.", "pages/4_Trust_and_Validation.py"),
    ("5  Scenario Lab", "What-if controls: buffers, station speed, variant mix, bad batches, sensor retrofits.", "pages/5_Scenario_Lab.py"),
]

cols = st.columns(len(nav_cards))
for col, (title, desc, target) in zip(cols, nav_cards):
    with col:
        st.markdown(f'<div class="tw-nav-card"><h4>{title}</h4><p>{desc}</p></div>', unsafe_allow_html=True)
        st.page_link(target, label="Open", icon="👉")
