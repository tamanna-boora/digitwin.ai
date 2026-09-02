"""Leadership: the business case from the backtest cost model, payback and a
3-year projection, and a trust summary pulled live from the ledger.
"""

import sys
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import state
from components.analytics import constraint_history
from components.theme import CATEGORICAL, apply_theme, status_color
from twinline.trust.ledger import scorecard

st.set_page_config(page_title="Leadership — TwinLine", page_icon="🏭", layout="wide")
apply_theme()
state.render_sidebar_scenario_selector()

if not state.require_data_and_models():
    st.stop()

cfg = state.load_config()
line = state.load_line_data()
pipeline = state.load_pipeline()
ledger_conn = state.get_ledger_connection()
ablation_rows = state.get_ablation_table()
ablation = {row.method: row for row in ablation_rows}
ml_row = ablation.get("ML-only")

st.title("Leadership")

# --- lead with the ablation finding -----------------------------------------
negative_rows = [r for r in ablation_rows if r.net_benefit < 0]
positive_rows = [r for r in ablation_rows if r.net_benefit > 0]

if negative_rows and positive_rows:
    negative_methods = ", ".join(r.method for r in negative_rows)
    negative_values = " / ".join(f"{r.net_benefit:+.0f}" for r in negative_rows)
    positive_methods = " and ".join(r.method for r in positive_rows)
    positive_values = " / ".join(f"{r.net_benefit:+.0f}" for r in positive_rows)
    verb = "is" if len(positive_rows) == 1 else "are"
    st.error(
        f"**At the current alarm budget, {negative_methods} all lose money** "
        f"(net {negative_values} currency in the backtest window) — investigation cost on false alarms outweighs "
        f"rework avoided. Only **{positive_methods}** {verb} net positive ({positive_values} currency). This is why "
        "alarm budgeting and calibrated abstention matter, not just having a model.",
        icon="📉",
    )

st.markdown("---")

# --- business case -----------------------------------------------------------
st.markdown('<div class="tw-section-title">Business case</div>', unsafe_allow_html=True)
st.caption("Computed from the ML-only backtest row — the only net-positive configuration at the current alarm "
           "budget. Adjust the assumptions below — the numbers move.")

assumption_cols = st.columns(4)
with assumption_cols[0]:
    rework_cost = st.number_input("Rework cost / unit", min_value=1.0, value=cfg.model.predict.alarm_budget.rework_cost_currency, step=10.0)
with assumption_cols[1]:
    investigation_cost = st.number_input("Investigation cost / false alarm", min_value=1.0, value=cfg.model.trust.investigation_cost_currency, step=5.0)
with assumption_cols[2]:
    scrap_rate_pct = st.slider("Scrap rate among caught defects (%)", 0, 100, 15)
with assumption_cols[3]:
    scrap_cost_multiplier = st.slider("Scrap cost, as a multiple of rework", 1.0, 10.0, 3.0, step=0.5)

throughput_margin_per_unit = st.slider(
    "Margin recovered per takt-equivalent unit of constraint time avoided", 0.0, 500.0, 120.0, step=10.0
)

if ml_row is None:
    st.warning("No ablation data available.")
else:
    tp_count = ml_row.rework_avoided / cfg.model.predict.alarm_budget.rework_cost_currency if cfg.model.predict.alarm_budget.rework_cost_currency else 0.0
    fp_count = ml_row.investigation_cost / cfg.model.trust.investigation_cost_currency if cfg.model.trust.investigation_cost_currency else 0.0

    rework_avoided = tp_count * rework_cost
    scrap_avoided = tp_count * (scrap_rate_pct / 100.0) * rework_cost * scrap_cost_multiplier
    cost_of_investigation = fp_count * investigation_cost

    station_ids = [s.id for s in cfg.plant.stations]
    history = constraint_history(pipeline.station_features.wide, station_ids, cfg.plant.takt_seconds, state.run_horizon_seconds(line))
    units_lost_total = float(history["units_lost"].sum()) if not history.empty else 0.0
    throughput_recovered = units_lost_total * throughput_margin_per_unit

    net_value = rework_avoided + scrap_avoided + throughput_recovered - cost_of_investigation

    value_cols = st.columns(4)
    for col, (label, value) in zip(value_cols, [
        ("Rework avoided", rework_avoided), ("Scrap avoided", scrap_avoided),
        ("Throughput recovered", throughput_recovered), ("Net value (backtest window)", net_value),
    ]):
        with col:
            st.markdown(
                f'<div class="tw-kpi-tile"><div class="tw-kpi-label">{label}</div>'
                f'<div class="tw-kpi-value">{value:,.0f}</div><div class="tw-kpi-sub">currency</div></div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # --- payback + 3-year projection -----------------------------------------
    st.markdown('<div class="tw-section-title">Payback & 3-year projection</div>', unsafe_allow_html=True)
    proj_cols = st.columns(3)
    with proj_cols[0]:
        n_lines = st.slider("Lines", 1, 20, 1)
    with proj_cols[1]:
        n_plants = st.slider("Plants", 1, 10, 1)
    with proj_cols[2]:
        deployment_cost_per_line = st.number_input("Deployment cost / line", min_value=0.0, value=15000.0, step=1000.0)

    test_fraction = 1.0 - cfg.model.predict.defect_risk.train_fraction - cfg.model.predict.defect_risk.calibration_fraction
    test_span_days = max(cfg.model.simulation_days * test_fraction, 0.01)
    daily_net_value_per_line = net_value / test_span_days
    annual_value = daily_net_value_per_line * 365.0 * n_lines * n_plants
    total_deployment_cost = deployment_cost_per_line * n_lines * n_plants

    monthly_value = annual_value / 12.0
    payback_months = total_deployment_cost / monthly_value if monthly_value > 0 else None

    payback_cols = st.columns(2)
    with payback_cols[0]:
        payback_text = f"{payback_months:.1f} months" if payback_months else "not reached"
        st.markdown(
            f'<div class="tw-kpi-tile"><div class="tw-kpi-label">Payback period</div>'
            f'<div class="tw-kpi-value">{payback_text}</div>'
            f'<div class="tw-kpi-sub">deployment cost {total_deployment_cost:,.0f} vs. {monthly_value:,.0f} / month</div></div>',
            unsafe_allow_html=True,
        )
    with payback_cols[1]:
        st.markdown(
            f'<div class="tw-kpi-tile"><div class="tw-kpi-label">Annualized value</div>'
            f'<div class="tw-kpi-value">{annual_value:,.0f}</div>'
            f'<div class="tw-kpi-sub">across {n_lines} line(s) x {n_plants} plant(s)</div></div>',
            unsafe_allow_html=True,
        )

    months = list(range(0, 37))
    cumulative = [monthly_value * m - total_deployment_cost for m in months]
    fig = go.Figure(go.Scatter(x=months, y=cumulative, mode="lines", line=dict(color=CATEGORICAL[0], width=2), fill="tozeroy"))
    fig.add_hline(y=0, line_color=status_color("watch"), line_dash="dash")
    fig.update_layout(height=300, xaxis_title="Month", yaxis_title="Cumulative net value (currency)")
    st.plotly_chart(fig, width='stretch', theme=None)

st.markdown("---")

# --- trust summary from the live ledger --------------------------------------
st.markdown('<div class="tw-section-title">Trust summary</div>', unsafe_allow_html=True)
score = scorecard(ledger_conn)
if score.empty:
    st.caption("No ledger data yet — run the backtest script to populate it.")
else:
    overall = score[score["instrumentation_level"] == "overall"]
    if not overall.empty:
        row = overall.iloc[0]
        trust_cols = st.columns(5)
        for col, (label, value, fmt) in zip(trust_cols, [
            ("Precision", row["precision"], ".1%"), ("Recall", row["recall"], ".1%"),
            ("False alarms", row["false_alarms"], ",.0f"), ("Mean lead time", row["mean_lead_time_s"] / 3600.0, ".1f"),
            ("Abstention rate", row["abstention_rate"], ".1%"),
        ]):
            with col:
                display = f"{value:{fmt}}" if value == value else "n/a"
                suffix = "h" if label == "Mean lead time" else ""
                st.markdown(
                    f'<div class="tw-kpi-tile"><div class="tw-kpi-label">{label}</div>'
                    f'<div class="tw-kpi-value" style="font-size:24px;">{display}{suffix}</div></div>',
                    unsafe_allow_html=True,
                )
    st.caption("Full scorecard, reliability diagram, and by-instrumentation breakdown: Trust & Validation page.")
