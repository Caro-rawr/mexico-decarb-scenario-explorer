"""
streamlit_app.py
----------------
Interactive dashboard for Mexico Decarbonization Scenario Explorer.

Launch: streamlit run app/streamlit_app.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import numpy as np

from src.data_loader import load_inegei_data, get_national_totals
from src.scenario_model import run_all_scenarios, compute_scenario_metrics, merge_historical_and_projections
from src.robustness import sample_uncertainty_space, evaluate_strategy_performance, compute_robustness_metrics, scenario_discovery, identify_no_regret_measures
from src.visualizer import (
    plot_national_trajectories,
    plot_sector_breakdown,
    plot_mitigation_wedges,
    plot_robustness_scatter,
    plot_sensitivity_tornado,
)

st.set_page_config(
    page_title="Mexico Decarbonization Scenario Explorer",
    page_icon="🇲🇽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# SIDEBAR — Controls
# ============================================================

st.sidebar.title("🇲🇽 Scenario Controls")
st.sidebar.markdown("---")

st.sidebar.subheader("Scenario Selection")
show_bau = st.sidebar.checkbox("BAU", value=True)
show_ndc_u = st.sidebar.checkbox("NDC Unconditional (−22%)", value=True)
show_ndc_c = st.sidebar.checkbox("NDC Conditional (−36%)", value=True)
show_15c = st.sidebar.checkbox("1.5°C Pathway (−51%)", value=True)

st.sidebar.markdown("---")
st.sidebar.subheader("Robustness Analysis")
n_samples = st.sidebar.slider("Monte Carlo Samples (SOW)", 100, 2000, 500, step=100)
robustness_scenario = st.sidebar.selectbox(
    "Evaluate Robustness For",
    ["NDC_conditional", "NDC_unconditional", "Pathway_15C"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.subheader("Visualization")
sector_scenario = st.sidebar.selectbox(
    "Sector Breakdown Scenario",
    ["NDC_unconditional", "NDC_conditional", "Pathway_15C", "BAU"],
    index=0,
)
target_year = st.sidebar.select_slider("Mitigation Wedge Year", options=[2030, 2035, 2040, 2050], value=2030)

# ============================================================
# DATA LOADING & COMPUTATION (cached)
# ============================================================

@st.cache_data
def load_data():
    return load_inegei_data(data_dir=Path("data/raw"))

@st.cache_data
def run_scenarios(_historical_df):
    return run_all_scenarios(_historical_df)

@st.cache_data
def run_robustness(_sow_df, scenario_id, baseline):
    return evaluate_strategy_performance(_sow_df, scenario_id, baseline)

historical_df = load_data()
results = run_scenarios(historical_df)

# Filter active scenarios
active_scenarios = {"Historical": historical_df}
if show_bau:
    active_scenarios["BAU"] = results.get("BAU")
if show_ndc_u:
    active_scenarios["NDC_unconditional"] = results.get("NDC_unconditional")
if show_ndc_c:
    active_scenarios["NDC_conditional"] = results.get("NDC_conditional")
if show_15c:
    active_scenarios["Pathway_15C"] = results.get("Pathway_15C")

combined_df = merge_historical_and_projections(historical_df, results)

# ============================================================
# MAIN LAYOUT
# ============================================================

st.title("🌎 Mexico Decarbonization Scenario Explorer")
st.markdown(
    """
    Interactive exploration of Mexico's GHG emission trajectories under BAU, NDC commitments, and 1.5°C-consistent pathways.  
    Robustness analysis uses DMDU methods to identify which futures lead to NDC achievement or failure.
    
    **Data:** INECC INEGEI synthetic demo (structure mirrors official 1990–2021 data) · **Units:** MtCO₂e (GWP-100, AR5)
    """
)

# --- Key Metrics Row ---
col1, col2, col3, col4 = st.columns(4)

total_2019 = historical_df[historical_df["year"] == 2019]["emissions_MtCO2e"].sum()
bau_2030 = results["BAU"][results["BAU"]["year"] == 2030]["emissions_MtCO2e"].sum()
ndc_u_2030 = results["NDC_unconditional"][results["NDC_unconditional"]["year"] == 2030]["emissions_MtCO2e"].sum()
pathway_15c_2030 = results["Pathway_15C"][results["Pathway_15C"]["year"] == 2030]["emissions_MtCO2e"].sum()

with col1:
    st.metric("2019 Baseline", f"{total_2019:.0f} MtCO₂e", help="Pre-COVID national total")
with col2:
    st.metric("BAU 2030", f"{bau_2030:.0f} MtCO₂e",
              delta=f"+{(bau_2030/total_2019-1)*100:.0f}% vs 2019",
              delta_color="inverse")
with col3:
    st.metric("NDC Unconditional 2030", f"{ndc_u_2030:.0f} MtCO₂e",
              delta=f"{(ndc_u_2030/bau_2030-1)*100:.0f}% vs BAU",
              delta_color="normal")
with col4:
    st.metric("1.5°C Pathway 2030", f"{pathway_15c_2030:.0f} MtCO₂e",
              delta=f"{(pathway_15c_2030/bau_2030-1)*100:.0f}% vs BAU",
              delta_color="normal")

st.divider()

# --- Tab Layout ---
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Trajectories", "🏭 Sectors", "📊 Robustness", "📋 Summary"
])

with tab1:
    st.subheader("National GHG Trajectories (1990–2050)")
    fig_traj = plot_national_trajectories(combined_df)
    st.plotly_chart(fig_traj, use_container_width=True)

    st.subheader(f"Mitigation Wedges vs BAU ({target_year})")
    compare = ["NDC_unconditional", "NDC_conditional", "Pathway_15C"]
    fig_wedge = plot_mitigation_wedges(results, target_year=target_year, compare_scenarios=compare)
    st.plotly_chart(fig_wedge, use_container_width=True)

with tab2:
    st.subheader(f"Sectoral Composition — {sector_scenario.replace('_', ' ')}")
    fig_sector = plot_sector_breakdown(combined_df, scenario_id=sector_scenario)
    st.plotly_chart(fig_sector, use_container_width=True)

    # Sector share table
    st.subheader("Sector Shares — 2030 vs 2019")
    from src.data_loader import get_sector_shares
    shares_2019 = get_sector_shares(historical_df, year=2019)
    scen_df = results.get(sector_scenario, pd.DataFrame())
    if not scen_df.empty:
        shares_2030 = get_sector_shares(scen_df, year=2030)
        share_table = pd.DataFrame({
            "Sector": shares_2019.index,
            "2019 Share": shares_2019.values,
            f"2030 Share ({sector_scenario.replace('_',' ')})": [shares_2030.get(s, 0) for s in shares_2019.index],
        })
        share_table["Change (pp)"] = share_table.iloc[:, 2] - share_table["2019 Share"]
        st.dataframe(share_table.set_index("Sector").style.format("{:.1%}", subset=["2019 Share", share_table.columns[2]]).format("{:+.1%}", subset="Change (pp)"))

with tab3:
    st.subheader(f"Robustness Analysis — {robustness_scenario.replace('_', ' ')}")
    st.markdown(
        f"Evaluating **{n_samples:,}** states of the world (Monte Carlo). "
        "Each point represents a combination of uncertain future conditions."
    )

    with st.spinner("Running robustness analysis..."):
        sow_df = sample_uncertainty_space(n_samples=n_samples)
        perf_df = run_robustness(sow_df, robustness_scenario, total_2019)
        metrics_df = compute_robustness_metrics(perf_df)

    # Robustness metrics
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric(
            "Meets NDC Unconditional",
            f"{metrics_df['pct_meets_ndc_unconditional'].iloc[0]:.0%}",
            help="% of futures where −22% vs BAU is achieved"
        )
    with c2:
        st.metric(
            "Meets NDC Conditional",
            f"{metrics_df['pct_meets_ndc_conditional'].iloc[0]:.0%}",
        )
    with c3:
        st.metric(
            "Meets 1.5°C",
            f"{metrics_df['pct_meets_15C'].iloc[0]:.0%}",
        )

    # Scatter: performance across SOW
    x_dim = st.selectbox(
        "X-Axis Uncertainty Dimension",
        ["gdp_growth_annual", "energy_technology_cost_factor",
         "carbon_price_2030_usd_tco2", "afolu_policy_stringency",
         "climate_finance_availability"],
        index=2,
    )

    fig_scatter = plot_robustness_scatter(perf_df, x_col=x_dim)
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Sensitivity analysis
    st.subheader("Uncertainty Sensitivity (Impact on NDC Success)")
    discovery = scenario_discovery(perf_df)
    fig_tornado = plot_sensitivity_tornado(discovery)
    st.plotly_chart(fig_tornado, use_container_width=True)

    # No-regret measures
    st.subheader("No-Regret Conditions")
    no_regret_df = identify_no_regret_measures(perf_df)
    no_regret_display = no_regret_df.copy()
    no_regret_display["NDC Success Rate"] = no_regret_display["NDC Success Rate"].apply(lambda x: f"{x:.1%}")
    no_regret_display["No-Regret"] = no_regret_display["No-Regret"].apply(lambda x: "✅ Yes" if x else "❌ No")
    st.dataframe(no_regret_display, use_container_width=True)

    if "tree_rules" in discovery and "scikit-learn" in str(discovery.get("tree_rules", "")):
        st.info("Install scikit-learn for decision tree scenario discovery: `pip install scikit-learn`")
    elif "tree_rules" in discovery:
        with st.expander("🌳 Scenario Discovery — Decision Tree Rules"):
            st.code(discovery["tree_rules"])

with tab4:
    st.subheader("Scenario Summary Metrics")
    metrics = compute_scenario_metrics(results, historical_df)
    st.dataframe(metrics.set_index("Scenario"), use_container_width=True)

    st.markdown("---")
    st.subheader("Methodological Notes")
    st.markdown(
        """
        **Scenario construction:**
        - BAU extrapolates 2010–2019 sectoral trends using OLS on log-emissions
        - Mitigation scenarios use logistic S-curves for technology adoption, calibrated to 2030 targets
        - 1.5°C pathway allows AFOLU net sinks after 2040 (restoration potential: −30 MtCO₂e)
        
        **DMDU approach:**
        - 5 deep uncertainty dimensions sampled uniformly (Monte Carlo, {n_samples} SOW)
        - Robustness metric: % of futures meeting NDC unconditional threshold (−22% vs BAU by 2030)
        - Scenario discovery uses CART decision tree to identify failure-triggering conditions
        - No-regret: conditions where success rate > 75% across the uncertainty space
        
        **Data:** Synthetic data mirroring INECC INEGEI (1990–2021) structure and approximate values.  
        Replace with official INEGEI Excel file in `data/raw/` for production use.
        
        **References:** INECC INEGEI (2023), Mexico NDC (2020 update), IPCC AR6, DMDU Society (2023), Walker et al. (2013).
        """
    )

# Footer
st.markdown("---")
st.caption("Carolina G. Cruz Núñez · M.Sc. Sustainability Sciences, UNAM · carostrepto@gmail.com · [LinkedIn](https://www.linkedin.com/in/carostrepto)")
