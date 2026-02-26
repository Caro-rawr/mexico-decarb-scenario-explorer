"""
app.py
Streamlit dashboard — Mexico Decarbonization Scenario Explorer

Run with:
    streamlit run app.py
"""

import streamlit as st
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.scenario_engine import ScenarioEngine, SCENARIO_LABELS
from src.charts import (
    trajectory_chart,
    sector_waterfall,
    sector_stacked_area,
    cumulative_reductions_chart,
    scenario_comparison_bar,
)

# ── Page configuration ───────────────────────────────────────────────────────
st.set_page_config(
    page_title="Mexico Decarb Scenario Explorer",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Cached computation ───────────────────────────────────────────────────────
@st.cache_data(show_spinner="Ejecutando simulaciones...")
def run_scenarios(n_mc: int = 500, rate_uncertainty: float = 0.20):
    engine = ScenarioEngine(n_mc_runs=n_mc, rate_uncertainty=rate_uncertainty)
    results = engine.run_all()
    robustness = engine.robustness_table(results)
    return results, robustness

# ── Sidebar controls ─────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Parámetros")
    st.markdown("---")

    selected_scenarios = st.multiselect(
        "Escenarios a comparar",
        options=list(SCENARIO_LABELS.keys()),
        default=list(SCENARIO_LABELS.keys()),
        format_func=lambda x: SCENARIO_LABELS[x]
    )

    show_uncertainty = st.toggle("Mostrar bandas de incertidumbre (MC)", value=True)

    st.markdown("---")
    st.markdown("**Parámetros Monte Carlo**")
    n_mc = st.slider("Corridas de simulación", min_value=100, max_value=2000, value=500, step=100)
    rate_uncertainty = st.slider(
        "Incertidumbre en tasas de reducción (±%)",
        min_value=5, max_value=50, value=20, step=5
    ) / 100

    st.markdown("---")
    sector_view = st.selectbox(
        "Escenario para vista sectorial",
        options=list(SCENARIO_LABELS.keys()),
        format_func=lambda x: SCENARIO_LABELS[x],
        index=2  # NDC conditional as default
    )

    st.markdown("---")
    st.markdown(
        """
        **Fuentes de datos**
        - INECC — Inventario Nacional GEI 2020
        - Sexta Comunicación Nacional UNFCCC
        - NDC actualizada México (2022)
        - IEA Net Zero Roadmap (LAC)
        """
    )

# ── Load data ─────────────────────────────────────────────────────────────────
results, robustness_df = run_scenarios(n_mc=n_mc, rate_uncertainty=rate_uncertainty)

if not selected_scenarios:
    st.warning("Selecciona al menos un escenario para visualizar.")
    st.stop()

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🇲🇽 Mexico Decarbonization Scenario Explorer")
st.markdown(
    """
    Análisis de trayectorias de descarbonización sectorial bajo múltiples escenarios de política climática.
    Los rangos de incertidumbre reflejan variación en tasas de reducción mediante simulación Monte Carlo
    — enfoque de Toma de Decisiones bajo Incertidumbre Profunda (DMDU).
    """
)

# ── KPI row ───────────────────────────────────────────────────────────────────
kpi_cols = st.columns(4)

kpi_data = {
    sid: robustness_df[robustness_df["Escenario"] == SCENARIO_LABELS[sid]].iloc[0]
    for sid in ["BAU", "NDC_conditional", "NZE_1.5C"]
    if sid in results and SCENARIO_LABELS[sid] in robustness_df["Escenario"].values
}

with kpi_cols[0]:
    st.metric("Emisiones base 2020", "750 MtCO₂e", help="Inventario INECC 2020")
with kpi_cols[1]:
    if "NDC_unconditional" in results:
        r = [r for r in robustness_df["Escenario"] if "Incondicional" in r]
        if r:
            val = robustness_df.loc[robustness_df["Escenario"] == r[0], "Emisiones 2030 (MtCO₂e)"].values[0]
            st.metric("NDC Incondicional 2030", f"{val:.0f} MtCO₂e",
                      delta=f"-{100-val/750*100:.0f}% vs 2020", delta_color="inverse")
with kpi_cols[2]:
    if "NDC_conditional" in results:
        val = results["NDC_conditional"].total_2030
        st.metric("NDC Condicional 2030", f"{val:.0f} MtCO₂e",
                  delta=f"-{100-val/750*100:.0f}% vs 2020", delta_color="inverse")
with kpi_cols[3]:
    if "NZE_1.5C" in results:
        val = results["NZE_1.5C"].total_2030
        st.metric("Carbono Neutro 2030", f"{val:.0f} MtCO₂e",
                  delta=f"-{100-val/750*100:.0f}% vs 2020", delta_color="inverse")

st.markdown("---")

# ── Main trajectory chart ──────────────────────────────────────────────────────
st.subheader("Trayectorias de emisiones GEI (2020–2050)")
fig_traj = trajectory_chart(results, selected_scenarios, show_uncertainty)
st.plotly_chart(fig_traj, use_container_width=True)

# ── Cumulative reductions ──────────────────────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    st.subheader("Reducciones acumuladas vs BAU")
    filtered_results = {k: v for k, v in results.items() if k in selected_scenarios}
    st.plotly_chart(cumulative_reductions_chart(filtered_results), use_container_width=True)

with col2:
    st.subheader("Emisiones en 2030 por escenario")
    rob_filtered = robustness_df[
        robustness_df["Escenario"].isin(
            [SCENARIO_LABELS[s] for s in selected_scenarios if s in SCENARIO_LABELS]
        )
    ]
    st.plotly_chart(scenario_comparison_bar(rob_filtered), use_container_width=True)

# ── Sector decomposition ────────────────────────────────────────────────────────
st.markdown("---")
st.subheader(f"Análisis sectorial — {SCENARIO_LABELS.get(sector_view, sector_view)}")

col3, col4 = st.columns(2)
with col3:
    st.plotly_chart(sector_stacked_area(results[sector_view]), use_container_width=True)
with col4:
    st.plotly_chart(sector_waterfall(results[sector_view], year=2030), use_container_width=True)

# ── Robustness table ────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📊 Tabla de Robustez — Comparación entre escenarios")
st.markdown(
    """
    Métricas clave para evaluar robustez de cada escenario ante incertidumbre:
    presupuesto de carbono acumulado, reducción relativa y factibilidad de carbono neutro.
    """
)
st.dataframe(
    robustness_df.style.background_gradient(
        subset=["Reducción vs 2020 (%)"],
        cmap="RdYlGn"
    ),
    use_container_width=True
)

# ── Download button ──────────────────────────────────────────────────────────────
st.markdown("---")
all_trajectories = []
for sid, r in results.items():
    if sid not in selected_scenarios:
        continue
    tmp = r.sector_emissions.copy()
    tmp["total"] = tmp.sum(axis=1)
    tmp["scenario"] = r.label
    tmp["year"] = tmp.index
    all_trajectories.append(tmp)

if all_trajectories:
    export_df = pd.concat(all_trajectories).reset_index(drop=True)
    csv_bytes = export_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Descargar datos de trayectorias (CSV)",
        data=csv_bytes,
        file_name="mexico_decarb_trajectories.csv",
        mime="text/csv"
    )

st.markdown(
    """
    ---
    *Datos: INECC 2020, Sexta Comunicación Nacional UNFCCC, NDC México 2022, IEA NZE Roadmap*  
    *Metodología: Monte Carlo sobre tasas de reducción sectoriales | Enfoque DMDU*
    """
)
