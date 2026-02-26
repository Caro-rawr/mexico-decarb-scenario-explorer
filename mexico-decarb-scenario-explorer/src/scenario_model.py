"""
scenario_model.py
-----------------
Generates sector-level GHG emission trajectories for each scenario
from 2022 to 2050, constrained by Mexico's NDC and IPCC pathways.

Approach:
- BAU: sector-specific linear + trend extrapolation from 2019 baseline
- NDC/1.5°C: top-down allocation of national targets, distributed to
  sectors proportionally to their mitigation potential and policy levers
- Mitigation policies are modeled as logistic adoption curves (S-curves)
  which reflect realistic technology diffusion dynamics

Emission units: MtCO2e (GWP-100, AR5, consistent with INEGEI)
"""

import logging
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent.parent / "config" / "scenarios.yaml"
PROJECTION_YEARS = list(range(2022, 2051))


def run_all_scenarios(
    historical_df: pd.DataFrame,
    config_path: Path = CONFIG_PATH,
) -> dict:
    """
    Generate all configured scenario trajectories.

    Parameters
    ----------
    historical_df : pd.DataFrame
        Long-format historical emissions (year, sector, emissions_MtCO2e)
    config_path : Path
        Configuration file.

    Returns
    -------
    dict
        Keys are scenario names, values are long-format DataFrames with
        columns: year, sector, emissions_MtCO2e, scenario
    """
    cfg = yaml.safe_load(open(config_path))
    scenarios_cfg = cfg["scenarios"]
    sector_params = cfg["sector_parameters"]

    # Establish 2019 reference baseline
    baseline_2019 = _get_baseline(historical_df, year=2019)

    results = {}
    for scenario_id, scenario_cfg in scenarios_cfg.items():
        logger.info(f"Modeling scenario: {scenario_id}")
        df = _project_scenario(
            scenario_id=scenario_id,
            scenario_cfg=scenario_cfg,
            baseline_2019=baseline_2019,
            sector_params=sector_params,
            historical_df=historical_df,
        )
        df["scenario"] = scenario_id
        df["scenario_label"] = scenario_cfg["label"]
        results[scenario_id] = df
        logger.info(
            f"  {scenario_id}: 2030 total = {df[df['year'] == 2030]['emissions_MtCO2e'].sum():.1f} MtCO2e"
        )

    return results


def _get_baseline(df: pd.DataFrame, year: int) -> pd.Series:
    """Extract per-sector emissions for the baseline year."""
    base = df[df["year"] == year].copy()
    if base.empty:
        raise ValueError(f"No data found for baseline year {year}")
    return base.set_index("sector")["emissions_MtCO2e"]


def _project_scenario(
    scenario_id: str,
    scenario_cfg: dict,
    baseline_2019: pd.Series,
    sector_params: dict,
    historical_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Project sector-level emissions from 2022 to 2050 for one scenario.

    Strategy:
    1. BAU: extrapolate historical trends per sector
    2. Mitigation scenarios: start from BAU, then apply policy levers
       using logistic adoption curves calibrated to target year constraints
    """
    years = PROJECTION_YEARS

    if scenario_id == "BAU":
        return _project_bau(baseline_2019, sector_params, historical_df, years)
    else:
        # Get BAU as reference for mitigation calculations
        bau_df = _project_bau(baseline_2019, sector_params, historical_df, years)
        return _project_mitigation(
            scenario_id, scenario_cfg, baseline_2019, sector_params, bau_df, years
        )


def _project_bau(
    baseline_2019: pd.Series,
    sector_params: dict,
    historical_df: pd.DataFrame,
    years: list,
) -> pd.DataFrame:
    """
    BAU projection: sector-specific trend extrapolation.
    Uses OLS on 2010–2019 (pre-COVID) to estimate sectoral growth rates.
    """
    records = []

    for sector in baseline_2019.index:
        # Estimate trend from 2010–2019 (exclude COVID)
        hist_sector = historical_df[
            (historical_df["sector"] == sector)
            & (historical_df["year"].between(2010, 2019))
        ].copy()

        if len(hist_sector) >= 3:
            # Linear regression on log(emissions) → exponential growth rate
            x = hist_sector["year"].values - 2019
            y = np.log(hist_sector["emissions_MtCO2e"].values.clip(1))
            cagr = np.polyfit(x, y, 1)[0]  # slope = annual log-growth rate
        else:
            # Fallback to config CAGR for sector
            cagr = sector_params.get(sector, {}).get("cagr", 0.02) if "cagr" in sector_params.get(sector, {}) else 0.02

        base = baseline_2019[sector]

        for year in years:
            t = year - 2019
            emissions = base * np.exp(cagr * t)
            # Gradual structural efficiency improvement dampens growth after 2030
            if year > 2030:
                efficiency_discount = 0.995 ** (year - 2030)
                emissions *= efficiency_discount
            records.append({"year": year, "sector": sector, "emissions_MtCO2e": max(0, emissions)})

    return pd.DataFrame(records)


def _project_mitigation(
    scenario_id: str,
    scenario_cfg: dict,
    baseline_2019: pd.Series,
    sector_params: dict,
    bau_df: pd.DataFrame,
    years: list,
) -> pd.DataFrame:
    """
    Mitigation scenario projection.

    Method:
    - National target for 2030 (% vs BAU) is distributed to sectors
      proportionally to their max mitigation potential
    - Sectoral trajectories are shaped as logistic curves, ensuring
      gradual ramp-up to 2030 and continued reduction to 2050
    - AFOLU can go net-negative after 2040 if scenario implies it
    """
    target_2030_pct = scenario_cfg.get("target_2030_pct_vs_bau", -0.22)
    target_2050_pct = scenario_cfg.get("target_2050_pct_vs_1990", -0.50)
    assumptions = scenario_cfg.get("assumptions", {})

    # BAU totals by year
    bau_totals = bau_df.groupby("year")["emissions_MtCO2e"].sum()
    bau_2030 = bau_totals.get(2030, bau_totals.iloc[-1])

    # Target absolute 2030 emission level
    target_2030_abs = bau_2030 * (1 + target_2030_pct)

    # 2019 national total (reference for 2050 target)
    total_2019 = baseline_2019.sum()
    target_2050_abs = total_2019 * (1 + target_2050_pct)

    records = []

    for sector in baseline_2019.index:
        sp = sector_params.get(sector, {})
        max_mitigation_pct = sp.get("max_mitigation_potential_2030_pct", 0.25)

        # Sector-level BAU trajectory
        bau_sector = bau_df[bau_df["sector"] == sector].set_index("year")["emissions_MtCO2e"]

        for year in years:
            bau_val = bau_sector.get(year, bau_sector.iloc[-1])

            # Effective mitigation at this year: logistic ramp to 2030, then continued
            mitigation_2030 = max_mitigation_pct * abs(target_2030_pct) / 0.22
            mitigation_2030 = min(mitigation_2030, max_mitigation_pct)

            if year <= 2030:
                # Logistic S-curve: slow start, accelerate, plateau near 2030
                t_normalized = (year - 2022) / (2030 - 2022)  # 0 to 1
                mitigation_fraction = mitigation_2030 * _logistic(t_normalized, k=5, x0=0.5)
            else:
                # Post-2030: continued reduction toward 2050 target
                t_post = (year - 2030) / (2050 - 2030)
                extra_mitigation = (
                    max_mitigation_pct * min(target_2050_pct + 1.0, 0.95) * t_post
                )
                mitigation_fraction = mitigation_2030 + (max_mitigation_pct - mitigation_2030) * t_post
                mitigation_fraction = min(mitigation_fraction, 0.98)

            emissions = bau_val * (1 - mitigation_fraction)

            # AFOLU net sink: allow negative emissions post-2040 in 1.5°C scenario
            if sector == "AFOLU" and assumptions.get("afolu_net_sink_2040") and year >= 2040:
                sink_potential = sp.get("net_sink_potential_2040", -30.0)
                t_sink = (year - 2040) / 10
                emissions = max(emissions + sink_potential * t_sink, sink_potential)

            records.append(
                {"year": year, "sector": sector, "emissions_MtCO2e": round(emissions, 2)}
            )

    return pd.DataFrame(records)


def _logistic(x: float, k: float = 5, x0: float = 0.5) -> float:
    """Logistic (sigmoid) function. Returns value in (0, 1)."""
    return 1 / (1 + np.exp(-k * (x - x0)))


def compute_scenario_metrics(
    results: dict,
    historical_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute key summary metrics across all scenarios.

    Metrics:
    - 2030 national total (MtCO2e)
    - 2030 change vs BAU (%)
    - 2030 change vs 2019 (%)
    - 2050 national total (MtCO2e)
    - Year of net-zero (if achieved)
    - Cumulative 2022–2050 emissions (MtCO2e) — carbon budget relevance

    Parameters
    ----------
    results : dict
        Output of run_all_scenarios().
    historical_df : pd.DataFrame
        Historical INEGEI data.

    Returns
    -------
    pd.DataFrame
        One row per scenario with summary metrics.
    """
    bau_2030 = results["BAU"][results["BAU"]["year"] == 2030]["emissions_MtCO2e"].sum()
    total_2019 = historical_df[historical_df["year"] == 2019]["emissions_MtCO2e"].sum()

    rows = []
    for scenario_id, df in results.items():
        total_2030 = df[df["year"] == 2030]["emissions_MtCO2e"].sum()
        total_2050 = df[df["year"] == 2050]["emissions_MtCO2e"].sum()
        cumulative = df["emissions_MtCO2e"].sum()

        # Find net-zero year (first year national total ≤ 0)
        annual = df.groupby("year")["emissions_MtCO2e"].sum()
        nz_years = annual[annual <= 0].index.tolist()
        net_zero_year = nz_years[0] if nz_years else "Not achieved by 2050"

        rows.append(
            {
                "Scenario": df["scenario_label"].iloc[0],
                "2030 Total (MtCO2e)": round(total_2030, 1),
                "2030 vs BAU (%)": round((total_2030 / bau_2030 - 1) * 100, 1),
                "2030 vs 2019 (%)": round((total_2030 / total_2019 - 1) * 100, 1),
                "2050 Total (MtCO2e)": round(total_2050, 1),
                "Cumulative 2022–2050 (GtCO2e)": round(cumulative / 1000, 2),
                "Net-Zero Year": net_zero_year,
            }
        )

    return pd.DataFrame(rows)


def merge_historical_and_projections(
    historical_df: pd.DataFrame,
    results: dict,
) -> pd.DataFrame:
    """
    Combine historical INEGEI data with all scenario projections
    into a unified long-format DataFrame for visualization.
    """
    frames = []

    # Historical data (all scenarios share the same history)
    hist = historical_df.copy()
    hist["scenario"] = "Historical"
    hist["scenario_label"] = "Historical (INEGEI)"
    frames.append(hist)

    # Scenario projections
    for df in results.values():
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    # National totals
    combined["national_total"] = combined.groupby(["year", "scenario"])["emissions_MtCO2e"].transform("sum")

    return combined
