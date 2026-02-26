# Mexico Decarbonization Scenario Explorer 🇲🇽

An interactive Streamlit dashboard for analyzing Mexico's GHG emission reduction pathways under multiple decarbonization scenarios — from BAU to 1.5°C alignment — with Monte Carlo uncertainty quantification and DMDU-compatible robustness analysis.

## What It Does

This tool models Mexico's sectoral GHG trajectories from 2020 to 2050 under four scenarios:

| Scenario | Description |
|---|---|
| BAU | Business-As-Usual — no additional mitigation |
| NDC Incondicional | Mexico's unconditional NDC commitment (2030) |
| NDC Condicional | Mexico's conditional NDC (subject to international support) |
| Carbono Neutro 1.5°C | Net-zero compatible pathway |

Emission trajectories are computed for 8 sectors (Electricity, Transport, Industry, Residential, Industrial Processes, AFOLU, Waste, O&G Upstream) using INECC 2020 baseline data.

**Uncertainty is explicit**: Monte Carlo simulation over sector-specific reduction rates produces 10th–90th percentile confidence bands, enabling Decision-Making Under Deep Uncertainty (DMDU) analysis.

## Features

- 📈 Multi-scenario trajectory chart with MC confidence bands
- 🏭 Sector-level stacked area and waterfall decomposition
- 📊 Robustness table comparing scenarios on 5 metrics
- 📥 One-click CSV export of all trajectory data
- ⚙️ Interactive sidebar controls (MC runs, uncertainty range, scenario selection)

## Installation & Run

```bash
git clone https://github.com/yourusername/mexico-decarb-scenario-explorer.git
cd mexico-decarb-scenario-explorer
pip install -r requirements.txt
streamlit run app.py
```

## Data Sources

- **INECC** — Inventario Nacional de Gases y Compuestos de Efecto Invernadero 2020
- **SEMARNAT/INECC** — Sexta Comunicación Nacional de México ante la UNFCCC
- **NDC actualizada de México** (2022)
- **IEA** — Net Zero Emissions by 2050 Roadmap (Mexico chapter)

## Methodological Notes

Annual sector-level reduction rates are derived from INECC's published projections for the NDC scenarios and adapted from IEA's NZE Roadmap for the 1.5°C pathway. Monte Carlo perturbation applies ±20% uniform uncertainty on each sector's rate, producing realistic confidence intervals around deterministic trajectories.

The robustness table follows DMDU principles: rather than optimizing for a single future, it compares scenarios across multiple performance metrics to identify low-regret strategies.

## Project Structure

```
mexico-decarb-scenario-explorer/
├── app.py                  # Streamlit dashboard (main entry point)
├── src/
│   ├── scenario_engine.py  # Trajectory modeling + Monte Carlo
│   └── charts.py           # Plotly visualization functions
├── data/sample/            # Reference datasets
├── notebooks/              # Standalone analysis
├── tests/                  # Unit tests
└── requirements.txt
```

## Author

Carolina Cruz Núñez | M.Sc. Sustainability Sciences, UNAM  
Thesis: *Evaluation of Land-Use and Vegetation Strategies for Territorial Decarbonization in Yucatán*  
[linkedin.com/in/carostrepto](https://linkedin.com/in/carostrepto)
