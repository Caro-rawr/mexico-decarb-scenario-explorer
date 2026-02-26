"""
scenario_engine.py
Core engine for Mexico decarbonization scenario modeling.

Implements three emission trajectory scenarios per sector:
  - BAU:   Business-As-Usual (no additional policy)
  - NDC:   Aligned with Mexico's 2030 NDC targets (conditional + unconditional)
  - NZE:   Net-Zero-compatible pathway (1.5°C)

Each scenario applies sector-specific annual reduction rates to 2020
baseline emissions (sourced from Mexico's Sixth National Communication
to UNFCCC and INECC's 2020 National GHG Inventory).

Uncertainty is modeled via Monte Carlo simulation on key parameters
(reduction rate, baseline uncertainty), producing confidence bands.
This enables DMDU-style robustness analysis across plausible futures.
"""

import numpy as np
import pandas as pd
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

# ── Mexico 2020 baseline emissions by sector (MtCO₂e)
# Source: INECC — Inventario Nacional de Gases y Compuestos de Efecto Invernadero 2020
BASELINE_EMISSIONS_2020 = {
    "Energía - Generación Eléctrica": 127.3,
    "Energía - Transporte": 170.5,
    "Energía - Industria": 82.4,
    "Energía - Residencial y Comercial": 38.1,
    "Procesos Industriales": 45.2,
    "AFOLU": 92.7,
    "Residuos": 25.8,
    "Petróleo y Gas (upstream)": 68.3,
}

# ── Annual reduction rates by scenario and sector (fraction per year)
# Source: INECC 2030 projections, IEA Net Zero Roadmap for Mexico (adaptation)
SCENARIO_RATES = {
    "BAU": {
        "Energía - Generación Eléctrica": -0.005,   # slight increase in absolute
        "Energía - Transporte": -0.010,
        "Energía - Industria": -0.005,
        "Energía - Residencial y Comercial": -0.003,
        "Procesos Industriales": -0.003,
        "AFOLU": -0.008,
        "Residuos": -0.005,
        "Petróleo y Gas (upstream)": -0.003,
    },
    "NDC_unconditional": {
        "Energía - Generación Eléctrica": 0.025,
        "Energía - Transporte": 0.015,
        "Energía - Industria": 0.020,
        "Energía - Residencial y Comercial": 0.018,
        "Procesos Industriales": 0.015,
        "AFOLU": 0.025,
        "Residuos": 0.020,
        "Petróleo y Gas (upstream)": 0.018,
    },
    "NDC_conditional": {
        "Energía - Generación Eléctrica": 0.042,
        "Energía - Transporte": 0.028,
        "Energía - Industria": 0.032,
        "Energía - Residencial y Comercial": 0.030,
        "Procesos Industriales": 0.025,
        "AFOLU": 0.040,
        "Residuos": 0.032,
        "Petróleo y Gas (upstream)": 0.030,
    },
    "NZE_1.5C": {
        "Energía - Generación Eléctrica": 0.080,
        "Energía - Transporte": 0.055,
        "Energía - Industria": 0.060,
        "Energía - Residencial y Comercial": 0.050,
        "Procesos Industriales": 0.045,
        "AFOLU": 0.065,
        "Residuos": 0.055,
        "Petróleo y Gas (upstream)": 0.070,
    },
}

SCENARIO_LABELS = {
    "BAU": "Business-As-Usual",
    "NDC_unconditional": "NDC Incondicional (2030)",
    "NDC_conditional": "NDC Condicional (2030)",
    "NZE_1.5C": "Carbono Neutro (1.5°C)",
}

SCENARIO_COLORS = {
    "BAU": "#c1121f",
    "NDC_unconditional": "#f4a261",
    "NDC_conditional": "#2196f3",
    "NZE_1.5C": "#1a7340",
}


@dataclass
class ScenarioResult:
    """Container for a single scenario's trajectory data."""
    scenario_id: str
    label: str
    color: str
    years: List[int]
    total_emissions: List[float]          # MtCO₂e per year (deterministic)
    sector_emissions: pd.DataFrame         # Year × Sector matrix
    cumulative_reductions: List[float]    # Cumulative vs BAU
    mc_lower: List[float]                 # 10th percentile (Monte Carlo)
    mc_upper: List[float]                 # 90th percentile (Monte Carlo)
    total_2030: float
    pct_reduction_vs_2020: float


class ScenarioEngine:
    """
    Runs Mexico decarbonization scenarios from 2020 to 2050.
    
    Supports:
    - Deterministic trajectory for each scenario
    - Monte Carlo confidence bands (uncertainty on reduction rates)
    - Sector-level decomposition
    - Robustness metrics for DMDU analysis
    """

    def __init__(
        self,
        start_year: int = 2020,
        end_year: int = 2050,
        n_mc_runs: int = 500,
        rate_uncertainty: float = 0.20   # ±20% on annual reduction rates
    ):
        self.start_year = start_year
        self.end_year = end_year
        self.years = list(range(start_year, end_year + 1))
        self.n_mc_runs = n_mc_runs
        self.rate_uncertainty = rate_uncertainty
        self.sectors = list(BASELINE_EMISSIONS_2020.keys())
        self.baseline = BASELINE_EMISSIONS_2020.copy()

    def run_all(self) -> Dict[str, ScenarioResult]:
        """
        Runs all four scenarios and returns a dict of ScenarioResult objects.
        """
        results = {}
        bau_trajectory = None

        for scenario_id in ["BAU", "NDC_unconditional", "NDC_conditional", "NZE_1.5C"]:
            result = self._run_scenario(scenario_id, bau_trajectory)
            results[scenario_id] = result
            if scenario_id == "BAU":
                bau_trajectory = result.total_emissions

        logger.info(f"Ran {len(results)} scenarios for {len(self.years)} years")
        return results

    def _run_scenario(
        self,
        scenario_id: str,
        bau_trajectory: List[float] = None
    ) -> ScenarioResult:
        """
        Computes deterministic trajectory + MC confidence bands for one scenario.
        """
        rates = SCENARIO_RATES[scenario_id]

        # ── Deterministic trajectory ────────────────────────────────────
        sector_matrix = {}
        for sector in self.sectors:
            baseline = self.baseline[sector]
            rate = rates[sector]
            trajectory = [baseline]
            for _ in self.years[1:]:
                trajectory.append(trajectory[-1] * (1 - rate))
            sector_matrix[sector] = trajectory

        sector_df = pd.DataFrame(sector_matrix, index=self.years)
        total_emissions = sector_df.sum(axis=1).tolist()

        # Cumulative reductions vs BAU
        if bau_trajectory is not None:
            cumulative_reductions = [
                sum(bau_trajectory[:i+1]) - sum(total_emissions[:i+1])
                for i in range(len(self.years))
            ]
        else:
            cumulative_reductions = [0.0] * len(self.years)

        # ── Monte Carlo uncertainty bands ────────────────────────────────
        mc_totals = []
        for _ in range(self.n_mc_runs):
            run_total = np.zeros(len(self.years))
            for sector in self.sectors:
                baseline = self.baseline[sector]
                base_rate = rates[sector]
                # Perturb rate by ±rate_uncertainty (uniform distribution)
                perturbed_rate = base_rate * np.random.uniform(
                    1 - self.rate_uncertainty,
                    1 + self.rate_uncertainty
                )
                traj = [baseline]
                for _ in self.years[1:]:
                    traj.append(traj[-1] * (1 - perturbed_rate))
                run_total += np.array(traj)
            mc_totals.append(run_total)

        mc_array = np.array(mc_totals)
        mc_lower = np.percentile(mc_array, 10, axis=0).tolist()
        mc_upper = np.percentile(mc_array, 90, axis=0).tolist()

        # Key metrics
        baseline_total = sum(self.baseline.values())
        total_2030 = sector_df.loc[2030].sum() if 2030 in sector_df.index else total_emissions[-1]
        pct_reduction = (baseline_total - total_2030) / baseline_total * 100

        return ScenarioResult(
            scenario_id=scenario_id,
            label=SCENARIO_LABELS[scenario_id],
            color=SCENARIO_COLORS[scenario_id],
            years=self.years,
            total_emissions=total_emissions,
            sector_emissions=sector_df,
            cumulative_reductions=cumulative_reductions,
            mc_lower=mc_lower,
            mc_upper=mc_upper,
            total_2030=round(total_2030, 1),
            pct_reduction_vs_2020=round(pct_reduction, 1),
        )

    def robustness_table(self, results: Dict[str, ScenarioResult]) -> pd.DataFrame:
        """
        Generates a DMDU-style robustness summary comparing scenarios on
        multiple performance metrics.
        
        Metrics:
          - 2030 emissions (MtCO₂e)
          - % reduction vs 2020
          - Cumulative 2020-2050 emissions (carbon budget)
          - MC spread at 2030 (90th - 10th pct)
          - Net-zero feasibility year estimate
        """
        rows = []
        for sid, r in results.items():
            year_idx_2030 = r.years.index(2030) if 2030 in r.years else -1
            spread_2030 = (
                r.mc_upper[year_idx_2030] - r.mc_lower[year_idx_2030]
            ) if year_idx_2030 >= 0 else None

            # Estimate net-zero year
            nz_year = None
            for i, (y, e) in enumerate(zip(r.years, r.total_emissions)):
                if e <= sum(self.baseline.values()) * 0.05:
                    nz_year = y
                    break

            rows.append({
                "Escenario": r.label,
                "Emisiones 2030 (MtCO₂e)": r.total_2030,
                "Reducción vs 2020 (%)": r.pct_reduction_vs_2020,
                "Presupuesto C 2020-2050 (GtCO₂e)": round(sum(r.total_emissions) / 1000, 2),
                "Incertidumbre 2030 (MtCO₂e)": round(spread_2030, 1) if spread_2030 else "—",
                "Año Carbono Neutro (est.)": nz_year if nz_year else ">2050",
            })

        return pd.DataFrame(rows)
