"""
charts.py
Plotly visualization functions for the Mexico Decarbonization Scenario Explorer.
All functions return Plotly Figure objects for use in Streamlit or standalone.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List
import logging

logger = logging.getLogger(__name__)

SCENARIO_COLORS = {
    "BAU": "#c1121f",
    "NDC_unconditional": "#f4a261",
    "NDC_conditional": "#2196f3",
    "NZE_1.5C": "#1a7340",
}


def trajectory_chart(results: dict, selected_scenarios: List[str] = None,
                     show_uncertainty: bool = True) -> go.Figure:
    """
    Multi-scenario line chart with optional Monte Carlo confidence bands.
    
    Args:
        results: Dict of ScenarioResult from ScenarioEngine.run_all()
        selected_scenarios: List of scenario IDs to show (None = all)
        show_uncertainty: If True, adds shaded MC confidence bands
    """
    if selected_scenarios is None:
        selected_scenarios = list(results.keys())

    fig = go.Figure()

    for sid in selected_scenarios:
        r = results[sid]
        color = r.color

        # Confidence band
        if show_uncertainty:
            fig.add_trace(go.Scatter(
                x=r.years + r.years[::-1],
                y=r.mc_upper + r.mc_lower[::-1],
                fill="toself",
                fillcolor=color.replace("#", "rgba(").rstrip(")") + ",0.10)" if color.startswith("#") else color,
                line=dict(color="rgba(255,255,255,0)"),
                showlegend=False,
                hoverinfo="skip",
                name=f"{r.label} (banda MC)"
            ))
            # Use hex-to-rgba conversion properly
            r_hex = color.lstrip("#")
            rgb = tuple(int(r_hex[i:i+2], 16) for i in (0, 2, 4))
            fill_color = f"rgba({rgb[0]},{rgb[1]},{rgb[2]},0.12)"
            fig.data[-1].fillcolor = fill_color

        # Main line
        fig.add_trace(go.Scatter(
            x=r.years,
            y=r.total_emissions,
            mode="lines",
            name=r.label,
            line=dict(color=color, width=2.5),
            hovertemplate=(
                f"<b>{r.label}</b><br>"
                "Año: %{x}<br>"
                "Emisiones: %{y:.1f} MtCO₂e<extra></extra>"
            )
        ))

    # 2030 reference line
    fig.add_vline(x=2030, line_dash="dot", line_color="gray", opacity=0.5,
                  annotation_text="Meta 2030", annotation_position="top")

    # Baseline reference
    baseline_total = 750.3  # sum of BASELINE_EMISSIONS_2020
    fig.add_hline(y=baseline_total, line_dash="dash", line_color="gray", opacity=0.4,
                  annotation_text="Línea base 2020", annotation_position="right")

    fig.update_layout(
        title="Trayectorias de Emisiones GEI — México 2020–2050",
        xaxis_title="Año",
        yaxis_title="Emisiones totales (MtCO₂e)",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=500
    )
    return fig


def sector_waterfall(result, year: int = 2030) -> go.Figure:
    """
    Waterfall chart showing sector-level contribution to emissions in a given year.
    Compares a scenario vs BAU to show where reductions come from.
    """
    if year not in result.sector_emissions.index:
        year = result.sector_emissions.index[-1]

    sector_data = result.sector_emissions.loc[year].sort_values(ascending=False)

    fig = go.Figure(go.Waterfall(
        name=result.label,
        orientation="v",
        measure=["relative"] * len(sector_data) + ["total"],
        x=list(sector_data.index) + ["Total"],
        y=list(sector_data.values) + [sector_data.sum()],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        decreasing={"marker": {"color": "#52b788"}},
        increasing={"marker": {"color": "#c1121f"}},
        totals={"marker": {"color": "#264653"}},
        text=[f"{v:.1f}" for v in list(sector_data.values) + [sector_data.sum()]],
        textposition="outside"
    ))

    fig.update_layout(
        title=f"Descomposición sectorial — {result.label} ({year})",
        yaxis_title="MtCO₂e",
        xaxis_tickangle=-30,
        template="plotly_white",
        height=480,
        showlegend=False
    )
    return fig


def sector_stacked_area(result) -> go.Figure:
    """
    Stacked area chart showing sector composition of total emissions over time.
    """
    sector_colors = px.colors.qualitative.Set2
    df = result.sector_emissions

    fig = go.Figure()
    for i, sector in enumerate(df.columns):
        fig.add_trace(go.Scatter(
            x=df.index.tolist(),
            y=df[sector].tolist(),
            name=sector,
            stackgroup="one",
            mode="lines",
            line=dict(width=0.5, color=sector_colors[i % len(sector_colors)]),
            fillcolor=sector_colors[i % len(sector_colors)],
            hovertemplate=f"<b>{sector}</b><br>Año: %{{x}}<br>%{{y:.1f}} MtCO₂e<extra></extra>"
        ))

    fig.update_layout(
        title=f"Composición sectorial — {result.label}",
        xaxis_title="Año",
        yaxis_title="Emisiones (MtCO₂e)",
        template="plotly_white",
        hovermode="x unified",
        height=480,
    )
    return fig


def robustness_radar(robustness_df: pd.DataFrame) -> go.Figure:
    """
    Radar chart comparing all scenarios across robustness metrics.
    """
    metrics = ["Reducción vs 2020 (%)", "Presupuesto C 2020-2050 (GtCO₂e)"]
    available = [m for m in metrics if m in robustness_df.columns]
    if not available:
        return go.Figure()

    fig = go.Figure()
    colors = list(SCENARIO_COLORS.values())

    for i, row in robustness_df.iterrows():
        values = []
        for m in available:
            v = row.get(m, 0)
            try:
                values.append(float(v))
            except Exception:
                values.append(0.0)

        fig.add_trace(go.Scatterpolar(
            r=values + [values[0]],
            theta=available + [available[0]],
            fill="toself",
            name=row.get("Escenario", f"Scenario {i}"),
            line_color=colors[i % len(colors)],
            opacity=0.7
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        title="Análisis de Robustez por Escenario",
        template="plotly_white",
        height=450
    )
    return fig


def cumulative_reductions_chart(results: dict) -> go.Figure:
    """
    Cumulative GHG reductions relative to BAU for each non-BAU scenario.
    Answers: how many tonnes are avoided cumulatively under each pathway?
    """
    fig = go.Figure()

    for sid, r in results.items():
        if sid == "BAU":
            continue
        fig.add_trace(go.Scatter(
            x=r.years,
            y=r.cumulative_reductions,
            mode="lines",
            name=r.label,
            line=dict(color=r.color, width=2),
            fill="tozeroy",
            fillcolor=r.color.replace("#", "rgba(").rstrip(")") + ",0.08)" if r.color.startswith("#") else r.color,
        ))

    # proper fill color
    for trace in fig.data:
        c = trace.line.color
        if c.startswith("#"):
            hex_c = c.lstrip("#")
            rgb = tuple(int(hex_c[i:i+2], 16) for i in (0, 2, 4))
            trace.fillcolor = f"rgba({rgb[0]},{rgb[1]},{rgb[2]},0.12)"

    fig.update_layout(
        title="Reducciones Acumuladas vs BAU (MtCO₂e acumuladas)",
        xaxis_title="Año",
        yaxis_title="MtCO₂e evitadas (acumulado)",
        template="plotly_white",
        hovermode="x unified",
        height=430
    )
    return fig


def scenario_comparison_bar(robustness_df: pd.DataFrame) -> go.Figure:
    """
    Horizontal bar chart comparing 2030 emissions across scenarios.
    """
    col = "Emisiones 2030 (MtCO₂e)"
    if col not in robustness_df.columns:
        return go.Figure()

    df_sorted = robustness_df.sort_values(col, ascending=False)
    colors_map = {
        "Business-As-Usual": "#c1121f",
        "NDC Incondicional (2030)": "#f4a261",
        "NDC Condicional (2030)": "#2196f3",
        "Carbono Neutro (1.5°C)": "#1a7340",
    }

    fig = px.bar(
        df_sorted,
        x=col,
        y="Escenario",
        orientation="h",
        text=col,
        color="Escenario",
        color_discrete_map=colors_map,
        title=f"Emisiones GEI totales en 2030 por escenario",
        template="plotly_white",
    )
    fig.update_traces(texttemplate="%{text:.0f} MtCO₂e", textposition="outside")
    fig.update_layout(showlegend=False, height=300)
    return fig
