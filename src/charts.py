"""Plotly figures for the Aedes Sentinel dashboard."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from . import config

RED = "#B3261E"
ORANGE = "#E65100"
AMBER = "#D99000"
GREEN = "#2E7D32"
INK = "#18212B"
MUTED = "#64748B"
GRID = "#E2E8F0"

TIER_SCALE = [
    [0.00, "#E8F5E9"],
    [0.25, "#FFF4D6"],
    [0.50, "#FBE9E1"],
    [0.75, "#F4A582"],
    [1.00, "#B3261E"],
]


def _style(fig: go.Figure, height: int = 340, margin: dict | None = None) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=margin or dict(l=10, r=10, t=42, b=52),
        font=dict(family="Inter, system-ui, -apple-system, Segoe UI, sans-serif", color=INK, size=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(bgcolor="white", font_size=12, bordercolor=GRID),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.16,
            x=0,
            xanchor="left",
        ),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def risk_gauge(score: float, tier: str) -> go.Figure:
    color = config.TIER_COLORS.get(tier, MUTED)
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=max(0.0, min(100.0, float(score))),
            number={"suffix": "/100", "font": {"size": 34, "color": INK}},
            title={"text": "Risk index", "font": {"size": 15, "color": MUTED}},
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickwidth": 0,
                    "tickcolor": MUTED,
                    "showticklabels": False,
                },
                "bar": {"color": color, "thickness": 0.32},
                "bgcolor": "white",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 25], "color": config.TIER_BG["Low"]},
                    {"range": [25, 50], "color": config.TIER_BG["Medium"]},
                    {"range": [50, 75], "color": config.TIER_BG["High"]},
                    {"range": [75, 100], "color": config.TIER_BG["Critical"]},
                ],
            },
            domain={"x": [0.10, 0.90], "y": [0.05, 0.85]},
        )
    )
    fig.update_layout(
        height=230,
        margin=dict(l=16, r=16, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, system-ui, sans-serif", color=INK),
    )
    return fig


def forecast_figure(preds: pd.DataFrame, as_of: pd.Timestamp) -> go.Figure:
    data = preds.sort_values("horizon").copy()
    x = pd.to_datetime(data["target_date"])
    fig = go.Figure()

    if data["high"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=x,
                y=data["high"],
                mode="lines",
                line=dict(width=0),
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=data["low"],
                mode="lines",
                fill="tonexty",
                fillcolor="rgba(179, 38, 30, 0.12)",
                line=dict(width=0),
                name="80% prediction interval",
                hoverinfo="skip",
            )
        )

    colors = [config.TIER_COLORS.get(t, RED) for t in data["tier"]]
    fig.add_trace(
        go.Scatter(
            x=x,
            y=data["cases"],
            mode="lines+markers",
            name="Predicted cases",
            line=dict(color=RED, width=3),
            marker=dict(size=12, color=colors, line=dict(color="white", width=2)),
            customdata=np.stack([data["tier"], data["low"], data["high"]], axis=-1),
            hovertemplate=(
                "<b>%{x|%d %b %Y}</b><br>"
                "Predicted: %{y:.1f} cases<br>"
                "Interval: %{customdata[1]:.1f}–%{customdata[2]:.1f}<br>"
                "Tier: %{customdata[0]}<extra></extra>"
            ),
        )
    )
    fig.add_vline(
        x=as_of,
        line=dict(color=MUTED, width=1, dash="dot"),
        annotation_text="as-of week",
        annotation_position="top left",
        annotation_font=dict(size=10, color=MUTED),
    )
    fig.update_layout(
        title=dict(text="Next 4 weeks", font=dict(size=15, color=INK)),
        xaxis_title=None,
        yaxis_title="Weekly cases",
        yaxis=dict(rangemode="tozero"),
    )
    return _style(fig, height=330)


def driver_figure(drivers: pd.DataFrame) -> go.Figure:
    data = drivers.sort_values("delta", ascending=True).copy()
    colors = [GREEN if value < 0 else RED for value in data["delta"]]
    labels = data["label"].tolist()
    values = data["delta"].tolist()
    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker_color=colors,
            marker_line_width=0,
            text=[f"{v:+.1f}" for v in values],
            textposition="outside",
            cliponaxis=False,
            customdata=np.stack([data["value"], data["reference"]], axis=-1),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Contribution: %{x:+.2f} cases<br>"
                "Current: %{customdata[0]:.2f}<br>"
                "Typical: %{customdata[1]:.2f}<extra></extra>"
            ),
        )
    )
    fig.add_vline(x=0, line=dict(color="#94A3B8", width=1))
    fig.update_layout(
        title=dict(text="What is moving the forecast", font=dict(size=15, color=INK)),
        xaxis_title="Change in predicted cases vs. typical conditions",
        yaxis_title=None,
        showlegend=False,
    )
    fig.update_xaxes(zeroline=False, showgrid=True, gridcolor=GRID)
    return _style(fig, height=340, margin=dict(l=10, r=30, t=42, b=10))


def rainfall_history_figure(history: pd.DataFrame, as_of: pd.Timestamp) -> go.Figure:
    data = history.tail(18).copy()
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(
            x=data[config.DATE_COL],
            y=data["rainfall_mm"],
            name="Rainfall (mm)",
            marker_color="#90CAF9",
            marker_line_width=0,
            hovertemplate="%{x|%d %b %Y}<br>Rainfall: %{y:.1f} mm<extra></extra>",
        ),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=data[config.DATE_COL],
            y=data[config.TARGET_COL],
            name="Reported cases",
            line=dict(color=RED, width=2.5),
            mode="lines+markers",
            marker=dict(size=5),
            hovertemplate="%{x|%d %b %Y}<br>Cases: %{y:.0f}<extra></extra>",
        ),
        secondary_y=True,
    )
    lag_start = as_of - pd.Timedelta(weeks=4)
    lag_end = as_of - pd.Timedelta(weeks=2)
    fig.add_vrect(
        x0=lag_start,
        x1=lag_end,
        fillcolor=AMBER,
        opacity=0.13,
        line_width=0,
        annotation_text="2–4 week lag window",
        annotation_position="top left",
        annotation_font=dict(size=10, color="#8A5A00"),
    )
    fig.add_vline(x=as_of, line=dict(color=MUTED, width=1, dash="dot"))
    fig.update_layout(
        title=dict(text="Rainfall, lag window and reported cases", font=dict(size=15, color=INK)),
        yaxis=dict(title="Rainfall (mm)", rangemode="tozero"),
        yaxis2=dict(title="Cases", rangemode="tozero", showgrid=False, tickformat=".0f"),
    )
    return _style(fig, height=330)


def ward_ranking_figure(batch: pd.DataFrame, top_n: int = 12) -> go.Figure:
    data = batch.dropna(subset=["cases"]).sort_values("cases", ascending=False).head(top_n)
    data = data.iloc[::-1]
    colors = [config.TIER_COLORS.get(t, RED) for t in data["tier"]]
    labels = [str(w).replace("_", " ") for w in data[config.WARD_COL]]
    fig = go.Figure(
        go.Bar(
            x=data["cases"],
            y=labels,
            orientation="h",
            marker_color=colors,
            marker_line_width=0,
            text=[f"{v:.1f}" for v in data["cases"]],
            textposition="outside",
            cliponaxis=False,
            customdata=np.stack([data["tier"], data["risk_index"]], axis=-1),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "Predicted: %{x:.1f} cases<br>"
                "Tier: %{customdata[0]}<br>"
                "Risk index: %{customdata[1]:.0f}/100<extra></extra>"
            ),
        )
    )
    fig.update_layout(
        title=dict(text="Highest-risk wards this week", font=dict(size=15, color=INK)),
        xaxis_title="Predicted weekly cases",
        yaxis_title=None,
        showlegend=False,
    )
    fig.update_xaxes(showgrid=True, gridcolor=GRID, rangemode="tozero")
    height = max(300, 60 + 28 * len(data))
    return _style(fig, height=height, margin=dict(l=10, r=40, t=42, b=10))


def heatmap_figure(matrix: pd.DataFrame) -> go.Figure:
    """matrix: index=ward_id, columns=target dates, values=predicted cases."""
    z = matrix.to_numpy(dtype=float)
    finite = z[np.isfinite(z)]
    zmax = max(12.0, float(finite.max())) if finite.size else 12.0
    x = [pd.Timestamp(c).strftime("%d %b") for c in matrix.columns]
    y = [str(w).replace("_", " ") for w in matrix.index]
    custom = np.empty(z.shape, dtype=object)
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            custom[i, j] = f"{z[i, j]:.1f}" if np.isfinite(z[i, j]) else "—"
    fig = go.Figure(
        go.Heatmap(
            z=z,
            x=x,
            y=y,
            zmin=0,
            zmax=zmax,
            colorscale=TIER_SCALE,
            colorbar=dict(title="Cases", thickness=12, len=0.7),
            text=custom,
            texttemplate="%{text}",
            textfont=dict(size=10),
            hovertemplate="<b>%{y}</b><br>Week of %{x}<br>Predicted: %{z:.1f} cases<extra></extra>",
            xgap=2,
            ygap=2,
        )
    )
    fig.update_layout(
        title=dict(text="Ward-level risk heatmap", font=dict(size=15, color=INK)),
        xaxis=dict(side="top", showgrid=False),
        yaxis=dict(autorange="reversed", showgrid=False),
    )
    height = max(280, min(620, 70 + 26 * len(matrix.index)))
    return _style(fig, height=height, margin=dict(l=10, r=10, t=60, b=10))


def validation_figure(test: pd.DataFrame, horizon: int | None = None) -> go.Figure:
    data = test.copy()
    if horizon is not None and config.HORIZON_COL in data.columns:
        data = data.loc[data[config.HORIZON_COL] == horizon]
    date_col = "target_date" if "target_date" in data.columns else config.DATE_COL
    if date_col not in data.columns:
        return _style(go.Figure(), height=320)
    agg = (
        data.groupby(date_col, as_index=False)[["y_true", "y_pred"]]
        .sum()
        .sort_values(date_col)
    )
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=agg[date_col],
            y=agg["y_true"],
            name="Actual cases",
            line=dict(color=INK, width=2),
            hovertemplate="%{x|%d %b %Y}<br>Actual: %{y:.0f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=agg[date_col],
            y=agg["y_pred"],
            name="Predicted cases",
            line=dict(color=RED, width=2, dash="dot"),
            hovertemplate="%{x|%d %b %Y}<br>Predicted: %{y:.0f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=dict(text="Chronological holdout: actual vs predicted", font=dict(size=15, color=INK)),
        yaxis_title="Weekly cases (all wards)",
    )
    return _style(fig, height=330)


def validation_error_figure(test: pd.DataFrame) -> go.Figure:
    if config.HORIZON_COL not in test.columns:
        return _style(go.Figure(), height=280)
    rows = []
    for horizon, group in test.groupby(config.HORIZON_COL):
        err = (group["y_true"] - group["y_pred"]).abs()
        rows.append(
            {
                "horizon": int(horizon),
                "mae": float(err.mean()),
                "rmse": float(np.sqrt(((group["y_true"] - group["y_pred"]) ** 2).mean())),
            }
        )
    data = pd.DataFrame(rows).sort_values("horizon")
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=[f"+{h} wk" for h in data["horizon"]],
            y=data["mae"],
            name="MAE",
            marker_color="#F4A582",
            text=[f"{v:.2f}" for v in data["mae"]],
            textposition="outside",
        )
    )
    fig.update_layout(
        title=dict(text="Error by forecast horizon", font=dict(size=15, color=INK)),
        yaxis_title="MAE (cases per ward-week)",
        showlegend=False,
    )
    return _style(fig, height=290)


def tier_bar(tier: str, label: str = "") -> str:
    """Small inline HTML chip used in cards and tables."""
    style = config.tier_style(tier)
    return (
        f"<span class='tier-chip' style='{style}'>{config.TIER_ICONS.get(tier, '●')} "
        f"{tier}{(' · ' + label) if label else ''}</span>"
    )
