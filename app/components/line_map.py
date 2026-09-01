"""The horizontal 24-station schematic: fill = current status, border style =
instrumentation tier, badge = defect hazard, bars between stations = buffer
fill. Nodes are drawn as shapes (the only way to get a dashed/dotted border
in Plotly); invisible markers sit on top of each shape purely to catch hover
and click events, since shapes themselves aren't interactive.
"""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from components.theme import CATEGORICAL, GRIDLINE, INK_MUTED, INK_PRIMARY, INK_SECONDARY, SURFACE, status_color
from twinline.predict.station_hazard import station_hazard_now
from twinline.schemas import (
    AnomalyDetectConfig,
    AnomalySignal,
    PlantLineConfig,
    SPCSignal,
    StationConfig,
    StationHazardConfig,
)

_NODE_WIDTH = 0.62
_NODE_HEIGHT = 0.6
_ZONE_Y_PAD = 0.55
_BORDER_DASH = {"rich": "solid", "partial": "dash", "manual": "dot"}
_ZONE_ORDER = ["BODY", "PAINT", "FINAL_ASSEMBLY"]
_ZONE_LABELS = {"BODY": "Body Construction", "PAINT": "Paint", "FINAL_ASSEMBLY": "Final Assembly"}


def _severity_bucket(hazard: float, cfg: AnomalyDetectConfig) -> str:
    if hazard >= cfg.severity_critical_threshold:
        return "critical"
    if hazard >= cfg.severity_warn_threshold:
        return "warn"
    if hazard >= cfg.severity_watch_threshold:
        return "watch"
    return "ok"


def _latest_bucket_row(wide: pd.DataFrame, station_id: str, now_s: float) -> pd.Series | None:
    if station_id not in wide.index.get_level_values("station_id"):
        return None
    station_df = wide.loc[station_id]
    eligible = station_df[station_df.index <= now_s]
    if eligible.empty:
        return None
    return eligible.iloc[-1]


def render_line_map(
    plant: PlantLineConfig,
    now_s: float,
    spc_signals: list[SPCSignal],
    anomaly_signals: list[AnomalySignal],
    station_features_wide: pd.DataFrame,
    hazard_cfg: StationHazardConfig,
    severity_cfg: AnomalyDetectConfig,
    key: str = "line_map",
) -> str | None:
    stations = sorted(plant.stations, key=lambda s: s.sequence)

    hazards = {s.id: station_hazard_now(s.id, now_s, spc_signals, anomaly_signals, hazard_cfg) for s in stations}
    statuses = {sid: _severity_bucket(h, severity_cfg) for sid, h in hazards.items()}

    fig = go.Figure()
    _add_zone_bands(fig, stations)
    _add_buffers(fig, stations, station_features_wide, now_s)
    node_ids = _add_stations(fig, stations, hazards, statuses)
    _add_legend_traces(fig)

    fig.update_layout(
        height=300,
        xaxis=dict(range=[0, len(stations) + 1], visible=False, fixedrange=True),
        yaxis=dict(range=[-1.6, 1.6], visible=False, fixedrange=True),
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=11)),
        margin=dict(l=10, r=10, t=36, b=10),
        plot_bgcolor=SURFACE,
        paper_bgcolor=SURFACE,
    )

    event = st.plotly_chart(
        fig, width='stretch', on_select="rerun", selection_mode="points", key=key,
        config={"displayModeBar": False},
    )
    # theme=None (used on every other chart) breaks widget mounting here when combined
    # with on_select="rerun" (a Streamlit 1.62 quirk) — safe to omit, since every shape
    # and annotation already sets its own explicit color, so there's nothing left for
    # Streamlit's default theme to override.

    selected_station = None
    if event and event.get("selection", {}).get("points"):
        point = event["selection"]["points"][0]
        idx = point.get("point_index")
        if idx is not None and idx < len(node_ids):
            selected_station = node_ids[idx]
    return selected_station


def _add_zone_bands(fig: go.Figure, stations: list[StationConfig]) -> None:
    for zone in _ZONE_ORDER:
        zone_stations = [s for s in stations if s.area.value == zone]
        if not zone_stations:
            continue
        x0 = min(s.sequence for s in zone_stations) - 0.5
        x1 = max(s.sequence for s in zone_stations) + 0.5
        fig.add_shape(
            type="rect", x0=x0, x1=x1, y0=-_ZONE_Y_PAD, y1=_ZONE_Y_PAD,
            fillcolor="rgba(255,255,255,0.02)", line=dict(color=GRIDLINE, width=1), layer="below",
        )
        fig.add_annotation(
            x=(x0 + x1) / 2, y=_ZONE_Y_PAD + 0.28, text=_ZONE_LABELS[zone], showarrow=False,
            font=dict(size=12, color=INK_MUTED), yanchor="bottom",
        )


def _add_buffers(fig: go.Figure, stations: list[StationConfig], wide: pd.DataFrame, now_s: float) -> None:
    for prev_station, next_station in zip(stations, stations[1:]):
        if prev_station.area != next_station.area:
            continue
        row = _latest_bucket_row(wide, next_station.id, now_s)
        fill_frac = 0.0
        if row is not None and "buffer_utilisation" in row.index and pd.notna(row["buffer_utilisation"]):
            fill_frac = max(0.0, min(float(row["buffer_utilisation"]), 1.5)) / 1.5
        x_mid = (prev_station.sequence + next_station.sequence) / 2
        bar_width = 0.14
        fig.add_shape(
            type="rect", x0=x_mid - bar_width, x1=x_mid + bar_width, y0=-0.12, y1=0.12,
            fillcolor="rgba(255,255,255,0.06)", line=dict(color=GRIDLINE, width=1), layer="below",
        )
        if fill_frac > 0:
            fill_height = 0.12 * fill_frac
            fig.add_shape(
                type="rect", x0=x_mid - bar_width, x1=x_mid + bar_width, y0=-fill_height, y1=fill_height,
                fillcolor=CATEGORICAL[0], line=dict(width=0), layer="below",
            )


def _add_stations(
    fig: go.Figure, stations: list[StationConfig], hazards: dict[str, float], statuses: dict[str, str]
) -> list[str]:
    hover_texts, node_ids, xs, ys = [], [], [], []

    for station in stations:
        x, y = float(station.sequence), 0.0
        status = statuses[station.id]
        fill = status_color(status)
        dash = _BORDER_DASH[station.instrumentation.value]

        fig.add_shape(
            type="rect", x0=x - _NODE_WIDTH / 2, x1=x + _NODE_WIDTH / 2, y0=y - _NODE_HEIGHT / 2, y1=y + _NODE_HEIGHT / 2,
            fillcolor=fill, opacity=0.85, line=dict(color=INK_PRIMARY, width=2, dash=dash), layer="above",
        )
        fig.add_annotation(
            x=x, y=y - _NODE_HEIGHT / 2 - 0.16, text=station.id, showarrow=False,
            font=dict(size=10, color=INK_SECONDARY),
        )
        if station.is_inspection_gate:
            fig.add_annotation(x=x, y=y + _NODE_HEIGHT / 2 + 0.14, text="GATE", showarrow=False,
                                font=dict(size=9, color=INK_MUTED))

        hazard_pct = round(hazards[station.id] * 100)
        fig.add_annotation(
            x=x + _NODE_WIDTH / 2 - 0.05, y=y + _NODE_HEIGHT / 2 - 0.05, text=f"{hazard_pct}", showarrow=False,
            font=dict(size=9, color=INK_PRIMARY, family="system-ui"), bgcolor="rgba(0,0,0,0.55)",
            bordercolor=fill, borderwidth=1, borderpad=1, xanchor="right", yanchor="top",
        )

        hover_texts.append(
            f"<b>{station.id}</b> — {station.name}<br>"
            f"Area: {station.area.value}<br>Tier: {station.instrumentation.value}<br>"
            f"Status: {status.upper()}<br>Defect hazard: {hazard_pct}%"
        )
        node_ids.append(station.id)
        xs.append(x)
        ys.append(y)

    fig.add_trace(
        go.Scatter(
            x=xs, y=ys, mode="markers", marker=dict(size=42, opacity=0.0), hovertext=hover_texts,
            hoverinfo="text", showlegend=False,
        )
    )
    return node_ids


def _add_legend_traces(fig: go.Figure) -> None:
    for status in ("ok", "watch", "warn", "critical"):
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None], mode="markers", marker=dict(size=12, color=status_color(status)),
                name=status.upper(), showlegend=True,
            )
        )
    for tier, dash in _BORDER_DASH.items():
        fig.add_trace(
            go.Scatter(
                x=[None], y=[None], mode="lines",
                line=dict(color=INK_SECONDARY, width=2, dash=dash), name=f"border: {tier}", showlegend=True,
            )
        )
