"""One design system, applied everywhere: a dark industrial palette, a shared
Plotly template, status colors, and card CSS. Values here are presentation
constants (the design system's own parameters), not simulation/business
figures — those still come from configs/ wherever they appear on screen.
"""

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

PLOTLY_TEMPLATE_NAME = "twinline_dark"

# Dark-mode chart chrome + validated categorical/status steps (dataviz skill reference palette).
SURFACE = "#1a1a19"
PAGE_PLANE = "#0d0d0d"
INK_PRIMARY = "#ffffff"
INK_SECONDARY = "#c3c2b7"
INK_MUTED = "#898781"
GRIDLINE = "#2c2c2a"
AXIS_LINE = "#383835"
BORDER = "rgba(255,255,255,0.10)"

CATEGORICAL = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9", "#e66767"]
SEQUENTIAL_BLUE = ["#1a1a19", "#104281", "#184f95", "#1c5cab", "#256abf", "#2a78d6", "#3987e5", "#5598e7", "#86b6ef"]

STATUS_COLORS = {"ok": "#0ca30c", "watch": "#fab219", "warn": "#ec835a", "critical": "#d03b3b"}
PROVENANCE_COLORS = {"real": "#3987e5", "soft": "#c98500", "manual": "#898781"}

FONT_FAMILY = "system-ui, -apple-system, 'Segoe UI', sans-serif"
TYPE_SCALE = {"hero": 40, "title": 22, "subtitle": 16, "body": 14, "caption": 12}


def _build_template() -> go.layout.Template:
    template = go.layout.Template()
    template.layout = go.Layout(
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family=FONT_FAMILY, color=INK_SECONDARY, size=TYPE_SCALE["body"]),
        title=dict(font=dict(color=INK_PRIMARY, size=TYPE_SCALE["subtitle"])),
        colorway=CATEGORICAL,
        xaxis=dict(
            gridcolor=GRIDLINE, linecolor=AXIS_LINE, zerolinecolor=AXIS_LINE, tickfont=dict(color=INK_MUTED),
            title=dict(font=dict(color=INK_SECONDARY)),
        ),
        yaxis=dict(
            gridcolor=GRIDLINE, linecolor=AXIS_LINE, zerolinecolor=AXIS_LINE, tickfont=dict(color=INK_MUTED),
            title=dict(font=dict(color=INK_SECONDARY)),
        ),
        legend=dict(font=dict(color=INK_SECONDARY), bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor=SURFACE, bordercolor=AXIS_LINE, font=dict(color=INK_PRIMARY)),
        margin=dict(l=48, r=24, t=48, b=40),
    )
    return template


def register_plotly_template() -> None:
    if PLOTLY_TEMPLATE_NAME not in pio.templates:
        pio.templates[PLOTLY_TEMPLATE_NAME] = _build_template()
    pio.templates.default = PLOTLY_TEMPLATE_NAME


def apply_theme() -> None:
    register_plotly_template()
    st.markdown(_CARD_CSS, unsafe_allow_html=True)


def status_color(status: str) -> str:
    return STATUS_COLORS.get(status, INK_MUTED)


def provenance_color(provenance: str) -> str:
    return PROVENANCE_COLORS.get(provenance, INK_MUTED)


def provenance_badge_html(provenance: str) -> str:
    color = provenance_color(provenance)
    label = provenance.upper()
    return (
        f'<span style="background:{color}22;color:{color};border:1px solid {color}66;'
        f'border-radius:4px;padding:1px 6px;font-size:{TYPE_SCALE["caption"]}px;'
        f'font-weight:600;letter-spacing:0.03em;">{label}</span>'
    )


_CARD_CSS = f"""
<style>
:root {{
    --tw-surface: {SURFACE};
    --tw-page: {PAGE_PLANE};
    --tw-ink-primary: {INK_PRIMARY};
    --tw-ink-secondary: {INK_SECONDARY};
    --tw-ink-muted: {INK_MUTED};
    --tw-border: {BORDER};
}}

.stApp {{
    background-color: var(--tw-page);
}}

.tw-card {{
    background: var(--tw-surface);
    border: 1px solid var(--tw-border);
    border-radius: 10px;
    padding: 16px 18px;
    margin-bottom: 12px;
}}

.tw-kpi-tile {{
    background: var(--tw-surface);
    border: 1px solid var(--tw-border);
    border-left: 3px solid {CATEGORICAL[0]};
    border-radius: 10px;
    padding: 14px 16px;
}}

.tw-kpi-value {{
    font-size: {TYPE_SCALE["hero"]}px;
    font-weight: 700;
    color: var(--tw-ink-primary);
    line-height: 1.1;
}}

.tw-kpi-label {{
    font-size: {TYPE_SCALE["caption"]}px;
    color: var(--tw-ink-muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
}}

.tw-kpi-sub {{
    font-size: {TYPE_SCALE["caption"]}px;
    color: var(--tw-ink-secondary);
    margin-top: 4px;
}}

.tw-section-title {{
    font-size: {TYPE_SCALE["title"]}px;
    font-weight: 700;
    color: var(--tw-ink-primary);
    margin-bottom: 4px;
}}

.tw-caption {{
    font-size: {TYPE_SCALE["caption"]}px;
    color: var(--tw-ink-muted);
}}

.tw-pill {{
    display: inline-block;
    border-radius: 999px;
    padding: 2px 10px;
    font-size: {TYPE_SCALE["caption"]}px;
    font-weight: 600;
}}

.tw-nav-card {{
    background: var(--tw-surface);
    border: 1px solid var(--tw-border);
    border-radius: 10px;
    padding: 18px;
    height: 100%;
}}

.tw-nav-card h4 {{
    color: var(--tw-ink-primary);
    margin: 0 0 6px 0;
}}

.tw-nav-card p {{
    color: var(--tw-ink-secondary);
    font-size: {TYPE_SCALE["body"]}px;
}}
</style>
"""
