"""
src/scenarios.py
────────────────────────────────────────────────────────────────
Motor de generación de trayectorias de descarbonización para México.

Implementa cuatro escenarios de referencia:
    1. BAU  — Tendencial sin política adicional
    2. NDC_U — NDC no condicionada (esfuerzo propio)
    3. NDC_C — NDC condicionada (con apoyo internacional)
    4. NZ2050 — Carbono neutro 2050 (compatible con 1.5°C)

También implementa análisis de sensibilidad y generación de
escenarios de Monte Carlo para cuantificar incertidumbre.

Uso:
    from src.scenarios import ScenarioEngine
    engine = ScenarioEngine()
    traj = engine.build_trajectory("ndc_unconditional", start=2015, end=2050)
    print(traj.head())
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

_DATA_PATH = Path(__file__).parent.parent / "data" / "emission_factors" / "mexico_sectoral_baseline.yaml"

SCENARIOS = ["bau", "ndc_unconditional", "ndc_conditional", "net_zero_2050"]
SECTORS = ["energy", "transport", "electricity", "agriculture", "waste", "industrial", "lulucf"]

SCENARIO_LABELS = {
    "bau": "Tendencial (BAU)",
    "ndc_unconditional": "NDC No Condicionada",
    "ndc_conditional": "NDC Condicionada",
    "net_zero_2050": "Carbono Neutro 2050",
}

SCENARIO_COLORS = {
    "bau": "#d32f2f",
    "ndc_unconditional": "#f57c00",
    "ndc_conditional": "#1976d2",
    "net_zero_2050": "#388e3c",
}


def _load_config(path: Path = _DATA_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


class ScenarioEngine:
    """Genera y analiza trayectorias de descarbonización para México.

    Parameters
    ----------
    config_path : Path, optional
        Ruta al YAML de parámetros sectoriales.
    base_year : int
        Año de referencia para emisiones base. Default: 2015.
    """

    def __init__(
        self,
        config_path: Optional[Path] = None,
        base_year: int = 2015,
    ) -> None:
        cfg = _load_config(config_path or _DATA_PATH)
        self.baseline = cfg["baseline_2015"]
        self.scenario_params = cfg["scenario_parameters"]
        self.ndc = cfg["ndc_2030"]
        self.measures = cfg["mitigation_measures"]
        self.base_year = base_year

        # Emisiones sectoriales del año base (MtCO2e)
        self.base_emissions: Dict[str, float] = self._extract_base_emissions()

    def _extract_base_emissions(self) -> Dict[str, float]:
        """Extrae las emisiones base por sector principal del YAML."""
        b = self.baseline
        return {
            "energy": b["energy"]["subsectors"]["industrial_energy"]
                      + b["energy"]["subsectors"]["residential_commercial"]
                      + b["energy"]["subsectors"]["oil_gas"],
            "transport": b["energy"]["subsectors"]["transport"],
            "electricity": b["energy"]["subsectors"]["electricity_generation"],
            "agriculture": b["agriculture"]["total"],
            "waste": b["waste"]["total"],
            "industrial": b["industrial_processes"]["total"],
            "lulucf": b["lulucf"]["subsectors"]["deforestation_emissions"]
                      + b["lulucf"]["subsectors"]["forest_degradation"],
        }

    # ─── Trayectorias base ─────────────────────────────────────────────────

    def build_trajectory(
        self,
        scenario: str,
        start: int = 2015,
        end: int = 2050,
    ) -> pd.DataFrame:
        """Genera la trayectoria de emisiones por sector para un escenario.

        Parameters
        ----------
        scenario : str
            Uno de: 'bau', 'ndc_unconditional', 'ndc_conditional', 'net_zero_2050'.
        start : int
            Año inicial de la trayectoria.
        end : int
            Año final (incluido).

        Returns
        -------
        pd.DataFrame
            Columnas: year, sector, emissions_mtco2e, scenario, cumulative_mtco2e
        """
        if scenario not in SCENARIOS:
            raise ValueError(f"Escenario '{scenario}' no válido. Opciones: {SCENARIOS}")

        rates = self.scenario_params["annual_reduction_rates"][scenario]
        records = []
        years = list(range(start, end + 1))

        for sector in SECTORS:
            base = self.base_emissions.get(sector, 0.0)
            rate = rates.get(sector, 0.0) / 100.0  # convertir % a decimal

            cum = 0.0
            prev_emissions = base
            for year in years:
                if year == self.base_year:
                    emissions = base
                else:
                    t = year - self.base_year
                    # Descarbonización compuesta: E(t) = E0 × (1 - r)^t
                    emissions = base * ((1 - rate) ** t)
                    # LULUCF puede volverse negativo (sumidero)
                    if sector != "lulucf":
                        emissions = max(0.0, emissions)

                cum += emissions
                records.append({
                    "year": year,
                    "sector": sector,
                    "emissions_mtco2e": round(emissions, 3),
                    "scenario": scenario,
                    "scenario_label": SCENARIO_LABELS[scenario],
                    "cumulative_mtco2e": round(cum, 3),
                })

        return pd.DataFrame(records)

    def build_all_scenarios(
        self, start: int = 2015, end: int = 2050
    ) -> pd.DataFrame:
        """Genera trayectorias para los 4 escenarios en un solo DataFrame."""
        dfs = [self.build_trajectory(sc, start, end) for sc in SCENARIOS]
        return pd.concat(dfs, ignore_index=True)

    def annual_totals(self, trajectory_df: pd.DataFrame) -> pd.DataFrame:
        """Agrega emisiones totales nacionales por año desde trayectorias sectoriales.

        Parameters
        ----------
        trajectory_df : pd.DataFrame
            Output de build_trajectory() o build_all_scenarios().

        Returns
        -------
        pd.DataFrame
            Columnas: year, scenario, total_mtco2e, cumulative_total
        """
        totals = (
            trajectory_df
            .groupby(["year", "scenario", "scenario_label"])["emissions_mtco2e"]
            .sum()
            .reset_index()
            .rename(columns={"emissions_mtco2e": "total_mtco2e"})
        )
        totals["cumulative_total"] = totals.groupby("scenario")["total_mtco2e"].cumsum()
        return totals

    # ─── Análisis de incertidumbre (DMDU) ─────────────────────────────────

    def monte_carlo_trajectories(
        self,
        scenario: str,
        n_simulations: int = 500,
        start: int = 2015,
        end: int = 2050,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Genera distribución de trayectorias bajo incertidumbre profunda.

        Usa variación aleatoria en las tasas de reducción anual dentro del
        rango de incertidumbre definido en el YAML, siguiendo el enfoque
        DMDU (Deep Uncertainty): en lugar de asignar probabilidades a estados
        del mundo, se exploran múltiples futuros plausibles.

        Parameters
        ----------
        scenario : str
            Escenario base para la variación.
        n_simulations : int
            Número de trayectorias a generar.
        start : int
            Año inicial.
        end : int
            Año final.
        seed : int
            Semilla para reproducibilidad.

        Returns
        -------
        pd.DataFrame
            Columnas: year, simulation_id, total_mtco2e
            Para calcular percentiles de la distribución.
        """
        rng = np.random.default_rng(seed)
        unc = self.scenario_params["uncertainty_ranges"].get(
            scenario, {"low": -0.20, "high": 0.10}
        )
        base_rates = self.scenario_params["annual_reduction_rates"][scenario]
        years = list(range(start, end + 1))
        records = []

        for sim_id in range(n_simulations):
            # Perturbación aleatoria uniforme en el rango de incertidumbre
            perturbation = rng.uniform(unc["low"], unc["high"])

            total_by_year: Dict[int, float] = {y: 0.0 for y in years}

            for sector in SECTORS:
                base = self.base_emissions.get(sector, 0.0)
                rate_base = base_rates.get(sector, 0.0) / 100.0
                # Perturbación afecta la tasa de reducción del sector
                rate = rate_base * (1 + perturbation)
                rate = max(-0.05, min(0.25, rate))  # clamping de límites físicos

                for year in years:
                    t = max(0, year - self.base_year)
                    em = base * ((1 - rate) ** t)
                    if sector != "lulucf":
                        em = max(0.0, em)
                    total_by_year[year] += em

            for year, total in total_by_year.items():
                records.append({
                    "year": year,
                    "simulation_id": sim_id,
                    "total_mtco2e": round(total, 3),
                    "scenario": scenario,
                })

        return pd.DataFrame(records)

    def uncertainty_bands(
        self,
        mc_df: pd.DataFrame,
        percentiles: Tuple[float, float, float] = (10, 50, 90),
    ) -> pd.DataFrame:
        """Calcula bandas de percentiles desde la distribución de Monte Carlo.

        Parameters
        ----------
        mc_df : pd.DataFrame
            Output de monte_carlo_trajectories().
        percentiles : tuple
            Percentiles a calcular. Default: (10, 50, 90).

        Returns
        -------
        pd.DataFrame
            Columnas: year, scenario, p10, p50, p90 (u otros percentiles).
        """
        p_low, p_mid, p_high = percentiles
        result = (
            mc_df.groupby(["year", "scenario"])["total_mtco2e"]
            .agg(
                p_low=lambda x: np.percentile(x, p_low),
                p_mid=lambda x: np.percentile(x, p_mid),
                p_high=lambda x: np.percentile(x, p_high),
            )
            .reset_index()
        )
        result.columns = ["year", "scenario", f"p{p_low}", f"p{p_mid}", f"p{p_high}"]
        return result

    # ─── Análisis de brecha y robustez ───────────────────────────────────

    def ndc_gap_analysis(self, trajectory_df: pd.DataFrame) -> pd.DataFrame:
        """Calcula la brecha entre trayectorias y metas NDC al 2030.

        Parameters
        ----------
        trajectory_df : pd.DataFrame
            Output de build_all_scenarios().

        Returns
        -------
        pd.DataFrame
            Brecha en MtCO2e y % para cada escenario al 2030.
        """
        totals = self.annual_totals(trajectory_df)
        year_2030 = totals[totals["year"] == 2030].copy()

        target_uncond = self.ndc["unconditional_target_mtco2e"]
        target_cond = self.ndc["conditional_target_mtco2e"]

        year_2030["ndc_unconditional_target"] = target_uncond
        year_2030["ndc_conditional_target"] = target_cond
        year_2030["gap_vs_uncond_mt"] = year_2030["total_mtco2e"] - target_uncond
        year_2030["gap_vs_cond_mt"] = year_2030["total_mtco2e"] - target_cond
        year_2030["gap_pct_uncond"] = (
            year_2030["gap_vs_uncond_mt"] / target_uncond * 100
        ).round(2)

        return year_2030[
            ["scenario", "scenario_label", "total_mtco2e",
             "ndc_unconditional_target", "gap_vs_uncond_mt", "gap_pct_uncond"]
        ].reset_index(drop=True)

    def robustness_ranking(
        self,
        sector: str,
        measure_key: str,
        n_worlds: int = 200,
        seed: int = 0,
    ) -> Dict[str, float]:
        """Evalúa la robustez de una medida de mitigación bajo múltiples futuros.

        Una medida es 'robusta' si reduce emisiones en el sector objetivo
        a través del rango de futuros plausibles, no solo en el caso central.

        Implementa el principio de decisión 'Satisficing' de la DMDU:
        una medida pasa si alcanza el objetivo en ≥ umbral % de los mundos.

        Parameters
        ----------
        sector : str
            Sector al que aplica la medida.
        measure_key : str
            Clave de la medida en el YAML (ej. 'zero_deforestation').
        n_worlds : int
            Número de futuros a evaluar.
        seed : int
            Semilla.

        Returns
        -------
        dict
            {'measure': str, 'robustness_pct': float, 'mean_reduction_mt': float}
        """
        if measure_key not in self.measures:
            raise KeyError(f"Medida '{measure_key}' no encontrada en el YAML.")

        measure = self.measures[measure_key]
        potential_central = measure["potential_mt_2030"]
        rng = np.random.default_rng(seed)

        successes = 0
        reductions = []

        for _ in range(n_worlds):
            # Variación del potencial de la medida bajo incertidumbre
            uncertainty_factor = rng.uniform(0.5, 1.3)  # ±30-50% de variación
            realized_reduction = potential_central * uncertainty_factor
            reductions.append(realized_reduction)
            # Condición de éxito: reducción ≥ 60% del potencial central
            if realized_reduction >= 0.60 * potential_central:
                successes += 1

        return {
            "measure": measure["label"],
            "sector": sector,
            "potential_central_mt": potential_central,
            "robustness_pct": round(successes / n_worlds * 100, 1),
            "mean_reduction_mt": round(np.mean(reductions), 2),
            "std_reduction_mt": round(np.std(reductions), 2),
            "cost_usd_tco2e": measure["abatement_cost_usd_tco2e"],
            "co_benefits": measure.get("co_benefits", []),
        }

    def all_measures_robustness(self, n_worlds: int = 200, seed: int = 0) -> pd.DataFrame:
        """Evalúa la robustez de todas las medidas de mitigación definidas."""
        results = []
        for key, measure_data in self.measures.items():
            sector = measure_data.get("sector", "unknown")
            result = self.robustness_ranking(sector, key, n_worlds, seed)
            result["measure_key"] = key
            results.append(result)

        df = pd.DataFrame(results)
        return df.sort_values("robustness_pct", ascending=False).reset_index(drop=True)

    # ─── Resumen ejecutivo ─────────────────────────────────────────────────

    def scenario_summary(self) -> pd.DataFrame:
        """Tabla comparativa de los 4 escenarios al 2030 y 2050."""
        all_traj = self.build_all_scenarios()
        totals = self.annual_totals(all_traj)

        rows = []
        for scenario in SCENARIOS:
            sc_data = totals[totals["scenario"] == scenario]
            base = sc_data[sc_data["year"] == self.base_year]["total_mtco2e"].values
            val_2030 = sc_data[sc_data["year"] == 2030]["total_mtco2e"].values
            val_2050 = sc_data[sc_data["year"] == 2050]["total_mtco2e"].values

            base_val = base[0] if len(base) > 0 else None
            val30 = val_2030[0] if len(val_2030) > 0 else None
            val50 = val_2050[0] if len(val_2050) > 0 else None

            rows.append({
                "scenario": scenario,
                "label": SCENARIO_LABELS[scenario],
                f"emissions_{self.base_year}_mtco2e": base_val,
                "emissions_2030_mtco2e": val30,
                "emissions_2050_mtco2e": val50,
                "reduction_vs_base_2030_pct": (
                    round((1 - val30 / base_val) * 100, 1)
                    if (base_val and val30 and base_val > 0) else None
                ),
                "reduction_vs_base_2050_pct": (
                    round((1 - val50 / base_val) * 100, 1)
                    if (base_val and val50 and base_val > 0) else None
                ),
            })

        return pd.DataFrame(rows)
