"""
visualizer.py
-------------
Plotly-based charts for the Mexico Decarbonization Scenario Explorer.
All figures are interactive (hover, zoom, legend toggle).
"""

import logging
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yaml
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent.parent / "config" / "scenarios.yaml"

SCENARIO_COLORS = {
    "Historical": "#78909c",
    "BAU": "#e53935",
    "NDC_unconditional": "#fb8c00",
    "NDC_conditional": "#43a047",
    "Pathway_15C": "#1565c0",
}

SECTOR_COLORS = {
    "Energía": "#ef5350",
    "Transporte": "#ab47bc",
    "AFOLU": "#66bb6a",
    "Procesos Industriales": "#ff7043",
    "Residuos": "#78909c",
}


def plot_national_trajectories(
    combined_df: pd.DataFrame,
    ndc_targets: Optional[dict] = None,
) -> go.Figure:
    """
    National-level GHG trajectory: historical + all scenario projections.
    Includes NDC target markers and uncertainty band for BAU.
    """
    # Aggregate to national totals
    national = (
        combined_df.groupby(["year", "scenario", "scenario_label"])["emissions_MtCO2e"]
        .sum()
        .reset_index()
        .rename(columns={"emissions_MtCO2e": "total_MtCO2e"})
    )

    fig = go.Figure()

    scenario_order = ["Historical", "BAU", "NDC_unconditional", "NDC_conditional", "Pathway_15C"]
    linestyles = {
        "Historical": "solid",
        "BAU": "dash",
        "NDC_unconditional": "solid",
        "NDC_conditional": "solid",
        "Pathway_15C": "solid",
    }

    for scenario_id in scenario_order:
        subset = national[national["scenario"] == scenario_id]
        if subset.empty:
            continue

        label = subset["scenario_label"].iloc[0]
        color = SCENARIO_COLORS.get(scenario_id, "#333")
        dash = linestyles.get(scenario_id, "solid")
        width = 2.5 if scenario_id != "Historical" else 2.0

        fig.add_trace(
            go.Scatter(
                x=subset["year"],
                y=subset["total_MtCO2e"],
                mode="lines",
                name=label,
                line=dict(color=color, dash=dash, width=width),
                hovertemplate=(
                    f"<b>{label}</b><br>Year: %{{x}}<br>Emissions: %{{y:.1f}} MtCO₂e<extra></extra>"
                ),
            )
        )

    # NDC target markers (2030)
    targets = ndc_targets or {
        "NDC Unconditional (−22% vs BAU)": (2030, None),  # calculated from BAU
    }

    # Add vertical reference line at 2030
    fig.add_vline(
        x=2030, line_dash="dot", line_color="gray", opacity=0.6,
        annotation_text="2030 NDC Target", annotation_position="top right",
    )
    fig.add_vline(
        x=2022, line_dash="dot", line_color="gray", opacity=0.3,
    )

    # Add 2019 reference point annotation
    hist_2019 = national[(national["scenario"] == "Historical") & (national["year"] == 2019)]
    if not hist_2019.empty:
        val = hist_2019["total_MtCO2e"].iloc[0]
        fig.add_annotation(
            x=2019, y=val,
            text=f"2019: {val:.0f} MtCO₂e",
            showarrow=True, arrowhead=2,
            xanchor="left", yanchor="bottom",
            font=dict(size=10, color="#546e7a"),
        )

    fig.update_layout(
        title=dict(
            text="Mexico GHG Trajectories by Scenario (1990–2050)",
            font=dict(size=16),
        ),
        xaxis_title="Year",
        yaxis_title="National GHG Emissions (MtCO₂e)",
        plot_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        font=dict(family="Inter, Arial", size=12),
        hovermode="x unified",
    )
    fig.update_xaxes(showgrid=True, gridcolor="#f0f0f0")
    fig.update_yaxes(showgrid=True, gridcolor="#f0f0f0")

    return fig


def plot_sector_breakdown(
    combined_df: pd.DataFrame,
    scenario_id: str = "NDC_unconditional",
) -> go.Figure:
    """
    Stacked area chart: sectoral composition of emissions for one scenario.
    """
    subset = combined_df[combined_df["scenario"] == scenario_id].copy()
    if subset.empty:
        logger.warning(f"No data for scenario: {scenario_id}")
        return go.Figure()

    scenario_label = subset["scenario_label"].iloc[0]
    sectors = subset["sector"].unique()

    fig = go.Figure()

    for sector in sectors:
        sector_data = subset[subset["sector"] == sector].sort_values("year")
        color = SECTOR_COLORS.get(sector, "#90a4ae")

        fig.add_trace(
            go.Scatter(
                x=sector_data["year"],
                y=sector_data["emissions_MtCO2e"],
                name=sector,
                stackgroup="one",
                fillcolor=color + "cc",
                line=dict(color=color),
                hovertemplate=(
                    f"<b>{sector}</b><br>Year: %{{x}}<br>%{{y:.1f}} MtCO₂e<extra></extra>"
                ),
            )
        )

    fig.add_vline(x=2022, line_dash="dot", line_color="#78909c", opacity=0.5)

    fig.update_layout(
        title=f"Sectoral Composition — {scenario_label}",
        xaxis_title="Year",
        yaxis_title="GHG Emissions (MtCO₂e)",
        plot_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        font=dict(family="Inter, Arial", size=12),
    )

    return fig


def plot_mitigation_wedges(
    results: dict,
    reference_scenario: str = "BAU",
    compare_scenarios: Optional[list] = None,
    target_year: int = 2030,
) -> go.Figure:
    """
    Bar chart comparing total mitigation gap between BAU and each scenario,
    disaggregated by sector (mitigation wedges).
    """
    if compare_scenarios is None:
        compare_scenarios = [k for k in results.keys() if k != reference_scenario]

    bau_df = results[reference_scenario]
    bau_2030 = bau_df[bau_df["year"] == target_year].set_index("sector")["emissions_MtCO2e"]

    fig = go.Figure()

    for scenario_id in compare_scenarios:
        df = results[scenario_id]
        scenario_label = df["scenario_label"].iloc[0] if "scenario_label" in df.columns else scenario_id
        df_2030 = df[df["year"] == target_year].set_index("sector")["emissions_MtCO2e"]

        for sector in bau_2030.index:
            bau_val = bau_2030.get(sector, 0)
            scen_val = df_2030.get(sector, bau_val)
            wedge = bau_val - scen_val  # positive = mitigation achieved

            fig.add_trace(
                go.Bar(
                    name=f"{scenario_label} — {sector}",
                    x=[scenario_label],
                    y=[wedge],
                    marker_color=SECTOR_COLORS.get(sector, "#90a4ae"),
                    legendgroup=sector,
                    legendgrouptitle_text=sector,
                    hovertemplate=(
                        f"<b>{sector}</b><br>Mitigation in {target_year}: %{{y:.1f}} MtCO₂e<extra></extra>"
                    ),
                )
            )

    fig.update_layout(
        barmode="stack",
        title=f"Mitigation Wedges by Sector vs BAU ({target_year})",
        xaxis_title="Scenario",
        yaxis_title=f"GHG Mitigation vs BAU in {target_year} (MtCO₂e)",
        plot_bgcolor="white",
        font=dict(family="Inter, Arial", size=12),
    )

    return fig


def plot_robustness_scatter(
    performance_df: pd.DataFrame,
    x_col: str = "gdp_growth_annual",
    y_col: str = "reduction_vs_bau_2030",
    color_col: str = "meets_ndc_unconditional",
) -> go.Figure:
    """
    Scatter plot of strategy performance across uncertainty space.
    Points colored by whether NDC threshold is met.
    """
    plot_df = performance_df.copy()
    plot_df["outcome"] = np.where(
        plot_df[color_col], "✅ Meets NDC", "❌ Misses NDC"
    )

    fig = px.scatter(
        plot_df,
        x=x_col,
        y=y_col,
        color="outcome",
        color_discrete_map={"✅ Meets NDC": "#43a047", "❌ Misses NDC": "#e53935"},
        opacity=0.6,
        title=f"Robustness Scatter: Performance across {len(plot_df)} Futures",
        labels={
            x_col: x_col.replace("_", " ").title(),
            y_col: "2030 Reduction vs BAU (%)",
        },
    )

    # Add NDC threshold line
    fig.add_hline(
        y=-0.22, line_dash="dash", line_color="#fb8c00",
        annotation_text="NDC Unconditional (−22%)",
        annotation_position="right",
    )
    fig.add_hline(
        y=-0.36, line_dash="dash", line_color="#43a047",
        annotation_text="NDC Conditional (−36%)",
        annotation_position="right",
    )

    fig.update_layout(
        plot_bgcolor="white",
        font=dict(family="Inter, Arial", size=12),
    )

    return fig


def plot_sensitivity_tornado(discovery_result: dict) -> go.Figure:
    """
    Tornado chart of uncertainty dimension importance for NDC success.
    """
    sensitivity = discovery_result.get("sensitivity", {})
    if not sensitivity:
        return go.Figure()

    dims = []
    deltas = []
    for dim, vals in sensitivity.items():
        dims.append(dim.replace("_", " ").title())
        deltas.append(abs(vals.get("delta", 0)))

    df = pd.DataFrame({"Dimension": dims, "Impact": deltas}).sort_values("Impact")

    fig = go.Figure(
        go.Bar(
            x=df["Impact"],
            y=df["Dimension"],
            orientation="h",
            marker_color="#1565c0",
            text=[f"{v:.3f}" for v in df["Impact"]],
            textposition="outside",
        )
    )

    fig.update_layout(
        title="Uncertainty Sensitivity: Impact on NDC Success Rate",
        xaxis_title="Δ Success Rate (High vs Low Half)",
        plot_bgcolor="white",
        font=dict(family="Inter, Arial", size=12),
    )

    return fig
