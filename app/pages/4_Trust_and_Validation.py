"""Trust & Validation: scorecard, reliability diagram, lead-time distribution,
alarm budget adherence, the ablation table, performance by instrumentation
level, the coverage map, soft-sensor interval coverage, and the operator
acknowledgement/dismissal log.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import state
from components.theme import CATEGORICAL, apply_theme, status_color
from twinline.features.soft_sensors import coverage_report
from twinline.trust.ledger import list_operator_actions, scorecard

st.set_page_config(page_title="Trust & Validation — TwinLine", page_icon="🏭", layout="wide")
apply_theme()
state.render_sidebar_scenario_selector()

if not state.require_data_and_models():
    st.stop()

cfg = state.load_config()
pipeline = state.load_pipeline()
ledger_conn = state.get_ledger_connection()

st.title("Trust & Validation")

# --- scorecard ------------------------------------------------------------------
st.markdown('<div class="tw-section-title">Scorecard</div>', unsafe_allow_html=True)
score = scorecard(ledger_conn)
if score.empty:
    st.info("No ledger data yet. Run `make demo` (or the backtest script) to populate it.")
else:
    display = score.copy()
    display["precision"] = display["precision"].map(lambda v: f"{v:.1%}" if v == v else "n/a")
    display["recall"] = display["recall"].map(lambda v: f"{v:.1%}" if v == v else "n/a")
    display["abstention_rate"] = display["abstention_rate"].map(lambda v: f"{v:.1%}" if v == v else "n/a")
    display["mean_lead_time_s"] = display["mean_lead_time_s"].map(lambda v: f"{v / 3600.0:.2f}h" if v == v else "n/a")
    display["calibration_error"] = display["calibration_error"].map(lambda v: f"{v:.3f}" if v == v else "n/a")
    st.dataframe(display, width='stretch', hide_index=True)
    st.caption(
        "This scorecard reflects a simple deterministic per-visit risk score used to exercise the ledger across "
        "every instrumentation tier — not the trained defect-risk model (see its PR-AUC on the Plant Manager / "
        "Home pages). **Performance by instrumentation level, stated plainly: manual-tier precision here is "
        "inflated by a near-binary heuristic (a failed check reads as high risk); partial-tier precision is "
        "genuinely the weakest — we are least reliable exactly where we are least instrumented.**"
    )

st.markdown("---")

# --- reliability diagram + lead-time distribution -------------------------------
diag_cols = st.columns(2)
predictions_df = pd.read_sql_query(
    "SELECT probability, outcome_detected FROM predictions WHERE abstained = 0 AND resolved = 1", ledger_conn
)

with diag_cols[0]:
    st.markdown('<div class="tw-section-title">Reliability diagram</div>', unsafe_allow_html=True)
    if predictions_df.empty:
        st.caption("No resolved, non-abstained predictions yet.")
    else:
        bins = np.linspace(0.0, 1.0, 11)
        predictions_df["bin"] = pd.cut(predictions_df["probability"], bins=bins, include_lowest=True)
        by_bin = predictions_df.groupby("bin", observed=True).agg(
            predicted=("probability", "mean"), observed=("outcome_detected", "mean"), n=("probability", "size")
        ).dropna()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Perfectly calibrated",
                                  line=dict(color=status_color("ok"), width=1, dash="dash")))
        fig.add_trace(go.Scatter(x=by_bin["predicted"], y=by_bin["observed"], mode="markers+lines",
                                  name="Observed", marker=dict(size=8, color=CATEGORICAL[0])))
        fig.update_layout(height=300, xaxis_title="Predicted probability", yaxis_title="Observed frequency",
                           xaxis_range=[0, 1], yaxis_range=[0, 1])
        st.plotly_chart(fig, width='stretch', theme=None)

with diag_cols[1]:
    st.markdown('<div class="tw-section-title">Lead-time distribution</div>', unsafe_allow_html=True)
    lead_df = pd.read_sql_query(
        "SELECT resolved_at_s - predicted_at_s AS lead_time_s FROM predictions "
        "WHERE alert_selected = 1 AND outcome_detected = 1 AND resolved = 1", ledger_conn
    )
    if lead_df.empty:
        st.caption("No true-positive alerts to measure lead time from yet.")
    else:
        fig = go.Figure(go.Histogram(x=lead_df["lead_time_s"] / 3600.0, marker_color=CATEGORICAL[1], nbinsx=20))
        fig.update_layout(height=300, xaxis_title="Lead time (hours)", yaxis_title="Count")
        st.plotly_chart(fig, width='stretch', theme=None)

st.markdown("---")

# --- alarm budget adherence -------------------------------------------------------
st.markdown('<div class="tw-section-title">Alarm budget adherence</div>', unsafe_allow_html=True)
line = state.load_line_data()
now_bucket_s = state.bucket_now_for_cache(state.run_horizon_seconds(line))
selected_alerts, digest_alerts = state.get_open_alerts(now_bucket_s)
by_shift_selected = pd.Series([a.candidate.shift_id for a in selected_alerts]).value_counts() if selected_alerts else pd.Series(dtype=int)
by_shift_digest = pd.Series([a.candidate.shift_id for a in digest_alerts]).value_counts() if digest_alerts else pd.Series(dtype=int)
all_shifts = sorted(set(by_shift_selected.index) | set(by_shift_digest.index))
if not all_shifts:
    st.caption("No alerts traced yet.")
else:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=all_shifts, y=[by_shift_selected.get(s, 0) for s in all_shifts], name="Alerted (within budget)", marker_color=CATEGORICAL[0]))
    fig.add_trace(go.Bar(x=all_shifts, y=[by_shift_digest.get(s, 0) for s in all_shifts], name="Digest (over budget)", marker_color=CATEGORICAL[6]))
    fig.update_layout(height=280, barmode="stack", yaxis_title="Alert candidates",
                       title=f"Budget: {cfg.model.predict.alarm_budget.max_alerts_per_shift} alerts / shift")
    st.plotly_chart(fig, width='stretch', theme=None, config={"displayModeBar": False})
    st.caption("The alarm budget is enforced by construction — alerted counts never exceed the budget; the digest "
               "bar shows how many additional candidates existed and were deliberately not surfaced.")

st.markdown("---")

# --- ablation table -----------------------------------------------------------
st.markdown('<div class="tw-section-title">Ablation table</div>', unsafe_allow_html=True)
ablation_rows = state.get_ablation_table()
ablation_df = pd.DataFrame([{
    "method": r.method, "PR-AUC": r.pr_auc, "precision@budget": r.precision_at_budget,
    "recall@budget": r.recall_at_budget, "mean lead time (h)": r.mean_lead_time_s / 3600.0,
    "rework avoided": r.rework_avoided, "investigation cost": r.investigation_cost, "net benefit": r.net_benefit,
} for r in ablation_rows])
st.dataframe(ablation_df.style.format({
    "PR-AUC": "{:.3f}", "precision@budget": "{:.3f}", "recall@budget": "{:.3f}",
    "mean lead time (h)": "{:.2f}", "rework avoided": "{:,.0f}", "investigation cost": "{:,.0f}", "net benefit": "{:+,.0f}",
}, na_rep="n/a"), width='stretch', hide_index=True)
st.caption("\"n/a\" mean lead time means that method caught zero true positives at the alarm budget — there's no lead time to average.")

st.markdown("---")

# --- coverage map + soft-sensor interval coverage -------------------------------
cov_cols = st.columns(2)
with cov_cols[0]:
    st.markdown('<div class="tw-section-title">Coverage map</div>', unsafe_allow_html=True)
    coverage = coverage_report(pipeline.soft_store)
    fig = go.Figure()
    for name, color in [("real_pct", CATEGORICAL[0]), ("soft_pct", CATEGORICAL[3]), ("blind_pct", CATEGORICAL[7])]:
        fig.add_trace(go.Bar(x=coverage["station_id"], y=coverage[name], name=name.replace("_pct", ""),
                              marker_color=color))
    fig.update_layout(height=340, barmode="stack", yaxis_title="% of time buckets")
    st.plotly_chart(fig, width='stretch', theme=None)

with cov_cols[1]:
    st.markdown('<div class="tw-section-title">Soft-sensor interval coverage</div>', unsafe_allow_html=True)
    validation = state.get_soft_sensor_validation()
    total_test = sum(v.n_test for v in validation)
    total_inside = sum(v.coverage_fraction * v.n_test for v in validation if v.n_test > 0)
    overall_coverage = total_inside / total_test if total_test else float("nan")

    fig = go.Figure()
    fig.add_trace(go.Bar(x=[v.archetype_id for v in validation], y=[v.coverage_fraction * 100 for v in validation],
                          marker_color=CATEGORICAL[2], name="Observed"))
    fig.add_hline(y=80.0, line_dash="dash", line_color=status_color("watch"), annotation_text="80% nominal")
    fig.update_layout(height=340, yaxis_title="10-90 interval coverage (%)", yaxis_range=[0, 100])
    st.plotly_chart(fig, width='stretch', theme=None)
    st.caption(f"Overall: {overall_coverage:.1%} against the 80% nominal target, n={total_test}.")

st.markdown("---")

# --- operator acknowledgements / dismissals -------------------------------------
st.markdown('<div class="tw-section-title">Operator acknowledgements & dismissals</div>', unsafe_allow_html=True)
actions_df = list_operator_actions(ledger_conn)
if actions_df.empty:
    st.caption("No operator actions logged yet — acknowledge or dismiss an alert on the Floor Supervisor page.")
else:
    st.dataframe(actions_df[["alert_id", "station_id", "action", "reason", "logged_at_s"]], width='stretch', hide_index=True)
    st.caption(
        "Dismissing an alert marks that specific incident closed for the Floor Supervisor feed — it does not "
        "retrain the model or suppress future alerts sharing the same driver or station. Each new incident is "
        "re-scored independently from the evidence available at the time, so a dismissal never silently reduces "
        "the twin's sensitivity."
    )
