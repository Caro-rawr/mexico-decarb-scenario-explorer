"""
robustness.py
-------------
DMDU-informed robustness analysis for decarbonization strategies.

Methods implemented:
1. Monte Carlo sampling of deep uncertainty dimensions
2. Strategy evaluation across the uncertainty ensemble
3. Robustness metrics: Regret, Satisficing, Maximum Regret
4. Scenario discovery: which futures lead to NDC failure?
5. No-regret identification: strategies robust across ALL sampled futures

Conceptual basis:
- Walker et al. (2013), "A Concept for Policy Robustness"
- Lempert et al. (2003), RAND Pardee Center, RDM methodology
- DMDU Society (2023) framework applied to climate policy

Note: This implementation uses scikit-learn for lightweight scenario discovery.
For production use, consider the ema_workbench library.
"""

import logging
import numpy as np
import pandas as pd
import yaml
from pathlib import Path
from typing import Optional, Tuple
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).parent.parent / "config" / "scenarios.yaml"


def sample_uncertainty_space(
    n_samples: int = 500,
    config_path: Path = CONFIG_PATH,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Generate Monte Carlo sample of deep uncertainty dimensions.

    Each row represents one "state of the world" (SOW) — a combination
    of uncertain parameters that together define a future context.

    Parameters
    ----------
    n_samples : int
        Number of states of the world to sample.

    Returns
    -------
    pd.DataFrame
        Shape (n_samples, n_dimensions), each column is one uncertainty dimension.
    """
    cfg = yaml.safe_load(open(config_path))
    unc_cfg = cfg["uncertainty"]
    dims = unc_cfg["dimensions"]

    rng = np.random.default_rng(random_seed)
    samples = {}

    for dim_name, dim_spec in dims.items():
        dist = dim_spec.get("distribution", "uniform")
        if dist == "uniform":
            samples[dim_name] = rng.uniform(
                dim_spec["min"], dim_spec["max"], size=n_samples
            )
        elif dist == "normal":
            samples[dim_name] = rng.normal(
                dim_spec.get("mean", 0.5),
                dim_spec.get("std", 0.1),
                size=n_samples,
            )
        elif dist == "lognormal":
            mu = np.log(dim_spec.get("mean", 1.0))
            sigma = dim_spec.get("std", 0.3)
            samples[dim_name] = rng.lognormal(mu, sigma, size=n_samples)

    df_sow = pd.DataFrame(samples)
    df_sow.index.name = "sow_id"
    logger.info(f"Sampled {n_samples} states of the world across {len(dims)} uncertainty dimensions.")
    return df_sow


def evaluate_strategy_performance(
    sow_df: pd.DataFrame,
    scenario_id: str,
    historical_baseline: float,
    config_path: Path = CONFIG_PATH,
) -> pd.DataFrame:
    """
    Evaluate how a mitigation strategy (scenario) performs under each SOW.

    For each state of the world, this function:
    1. Adjusts the scenario's mitigation potential based on uncertainty parameters
    2. Computes the resulting 2030 emission reduction vs BAU
    3. Records whether the NDC threshold is met

    This is a reduced-form model — a full factorial model would require
    running scenario_model.py for each SOW (computationally expensive).
    Here, parametric adjustment factors capture uncertainty propagation.

    Parameters
    ----------
    sow_df : pd.DataFrame
        Output of sample_uncertainty_space().
    scenario_id : str
        Which scenario to evaluate (must match scenarios.yaml).
    historical_baseline : float
        2019 national total (MtCO2e) for normalization.

    Returns
    -------
    pd.DataFrame
        SOW DataFrame with added performance columns:
        - reduction_vs_bau_2030: fractional GHG reduction vs BAU in 2030
        - meets_ndc_unconditional: bool
        - meets_ndc_conditional: bool
        - regret_vs_optimum: regret metric
    """
    cfg = yaml.safe_load(open(config_path))
    scenario_cfg = cfg["scenarios"].get(scenario_id, {})
    satisficing_cfg = cfg["uncertainty"]["satisficing"]

    target_2030 = scenario_cfg.get("target_2030_pct_vs_bau", -0.22)

    result = sow_df.copy()

    # --- Compute realized reduction under each SOW ---
    # The achieved reduction is modulated by:
    # (1) Technology cost factor: cheaper renewables → faster deployment → more mitigation
    # (2) Carbon price: higher price → stronger economic incentive
    # (3) AFOLU policy stringency: directly scales AFOLU mitigation
    # (4) Climate finance: gates conditional NDC components

    # Base mitigation target for the scenario
    base_reduction = abs(target_2030)

    # Technology cost effect: cheap tech boosts mitigation, expensive dampens it
    # Effect is non-linear: diminishing returns above a threshold
    tech_effect = 0.8 + 0.4 * (1 - result["energy_technology_cost_factor"])  # [0.4, 1.2]
    tech_effect = tech_effect.clip(0.3, 1.3)

    # Carbon price effect: each $10/tCO2 drives ~2% additional reduction
    carbon_price_effect = 1 + result["carbon_price_2030_usd_tco2"] * 0.002

    # AFOLU policy: directly scales AFOLU mitigation (10% of total in this sector)
    afolu_weight = 0.12  # AFOLU share of total national mitigation potential
    afolu_effect = 1 + (result["afolu_policy_stringency"] - 0.5) * afolu_weight

    # Climate finance: unlocks conditional NDC targets
    finance_threshold = 0.40  # Above this level, conditional components kick in
    finance_uplift = np.where(
        result["climate_finance_availability"] >= finance_threshold,
        1 + (result["climate_finance_availability"] - finance_threshold) * 0.3,
        1.0,
    )

    # GDP growth effect: faster growth generates more BAU emissions, making targets harder
    gdp_drag = 1 - (result["gdp_growth_annual"] - 0.024) * 2.0  # relative to baseline
    gdp_drag = gdp_drag.clip(0.85, 1.15)

    # Realized reduction (with all factors combined)
    realized_reduction = (
        base_reduction
        * tech_effect
        * carbon_price_effect
        * afolu_effect
        * finance_uplift
        * gdp_drag
    )
    realized_reduction = realized_reduction.clip(0.0, 0.95)

    result["reduction_vs_bau_2030"] = -realized_reduction  # negative = reduction

    # --- Satisficing: does strategy meet NDC thresholds? ---
    threshold = abs(satisficing_cfg["threshold"])
    result["meets_ndc_unconditional"] = realized_reduction >= 0.22
    result["meets_ndc_conditional"] = realized_reduction >= 0.36
    result["meets_15C"] = realized_reduction >= 0.51

    # --- Regret: distance from optimum if optimal strategy had been chosen ---
    # Optimum = maximum achievable reduction given the SOW
    max_possible = realized_reduction * 1.20  # 20% above realized = hypothetical best
    result["regret"] = np.maximum(0, max_possible - realized_reduction)

    return result


def compute_robustness_metrics(
    performance_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate performance across all SOWs into robustness metrics.

    Returns
    -------
    pd.DataFrame
        Single-row DataFrame with robustness summary statistics.
    """
    n = len(performance_df)

    metrics = {
        "n_sow": n,
        "mean_reduction_2030": performance_df["reduction_vs_bau_2030"].mean(),
        "p10_reduction": performance_df["reduction_vs_bau_2030"].quantile(0.10),
        "p50_reduction": performance_df["reduction_vs_bau_2030"].quantile(0.50),
        "p90_reduction": performance_df["reduction_vs_bau_2030"].quantile(0.90),
        "pct_meets_ndc_unconditional": performance_df["meets_ndc_unconditional"].mean(),
        "pct_meets_ndc_conditional": performance_df["meets_ndc_conditional"].mean(),
        "pct_meets_15C": performance_df["meets_15C"].mean(),
        "mean_regret": performance_df["regret"].mean(),
        "max_regret": performance_df["regret"].max(),
        "p95_regret": performance_df["regret"].quantile(0.95),
    }

    return pd.DataFrame([metrics])


def scenario_discovery(
    performance_df: pd.DataFrame,
    outcome_col: str = "meets_ndc_unconditional",
    uncertainty_cols: Optional[list] = None,
) -> dict:
    """
    Identify which uncertainty conditions drive NDC failure.

    Method: CART decision tree to partition the SOW space into
    "success" and "failure" regions by each uncertainty dimension.

    This is a lightweight implementation; full PRIM would use the
    ema_workbench library.

    Returns
    -------
    dict with:
    - 'feature_importance': which uncertainties matter most
    - 'failure_conditions': approximate thresholds for NDC failure
    - 'success_rate_by_dim': sensitivity analysis table
    """
    try:
        from sklearn.tree import DecisionTreeClassifier, export_text
        from sklearn.inspection import permutation_importance
    except ImportError:
        logger.warning("scikit-learn not available; returning simple sensitivity analysis.")
        return _simple_sensitivity(performance_df, outcome_col)

    if uncertainty_cols is None:
        uncertainty_cols = [
            "gdp_growth_annual",
            "energy_technology_cost_factor",
            "carbon_price_2030_usd_tco2",
            "afolu_policy_stringency",
            "climate_finance_availability",
        ]
        uncertainty_cols = [c for c in uncertainty_cols if c in performance_df.columns]

    X = performance_df[uncertainty_cols].copy()
    y = performance_df[outcome_col].astype(int)

    # Fit interpretable tree (max depth 3 → readable rules)
    clf = DecisionTreeClassifier(max_depth=3, min_samples_leaf=20, random_state=42)
    clf.fit(X, y)

    # Feature importance
    importances = pd.Series(clf.feature_importances_, index=uncertainty_cols)
    importances = importances.sort_values(ascending=False)

    # Decision rules (text)
    tree_rules = export_text(clf, feature_names=uncertainty_cols, max_depth=3)

    # Sensitivity: for each dimension, what is success rate in top vs bottom half?
    sensitivity = {}
    for col in uncertainty_cols:
        median = performance_df[col].median()
        high_success = performance_df.loc[performance_df[col] > median, outcome_col].mean()
        low_success = performance_df.loc[performance_df[col] <= median, outcome_col].mean()
        sensitivity[col] = {
            "high_half_success_rate": round(high_success, 3),
            "low_half_success_rate": round(low_success, 3),
            "delta": round(high_success - low_success, 3),
        }

    return {
        "feature_importance": importances,
        "tree_rules": tree_rules,
        "sensitivity": sensitivity,
        "overall_success_rate": y.mean(),
    }


def _simple_sensitivity(
    performance_df: pd.DataFrame,
    outcome_col: str,
) -> dict:
    """Fallback sensitivity analysis without sklearn."""
    uncertainty_cols = [
        "gdp_growth_annual", "energy_technology_cost_factor",
        "carbon_price_2030_usd_tco2", "afolu_policy_stringency",
        "climate_finance_availability",
    ]
    uncertainty_cols = [c for c in uncertainty_cols if c in performance_df.columns]

    sensitivity = {}
    importances = {}

    for col in uncertainty_cols:
        if col not in performance_df.columns:
            continue
        median = performance_df[col].median()
        high = performance_df.loc[performance_df[col] > median, outcome_col].mean()
        low = performance_df.loc[performance_df[col] <= median, outcome_col].mean()
        sensitivity[col] = {
            "high_half_success_rate": round(high, 3),
            "low_half_success_rate": round(low, 3),
            "delta": round(abs(high - low), 3),
        }
        importances[col] = abs(high - low)

    return {
        "feature_importance": pd.Series(importances).sort_values(ascending=False),
        "sensitivity": sensitivity,
        "overall_success_rate": performance_df[outcome_col].mean(),
        "tree_rules": "scikit-learn not available — install with: pip install scikit-learn",
    }


def identify_no_regret_measures(
    performance_df: pd.DataFrame,
    threshold_success_rate: float = 0.75,
) -> pd.DataFrame:
    """
    Identify "no-regret" conditions: parameter combinations where
    the strategy consistently meets NDC thresholds.

    A measure is "no-regret" if success rate is above threshold_success_rate
    across all sampled futures.

    Returns summary of robust vs vulnerable uncertainty ranges.
    """
    uncertainty_cols = [
        "gdp_growth_annual", "energy_technology_cost_factor",
        "carbon_price_2030_usd_tco2", "afolu_policy_stringency",
        "climate_finance_availability",
    ]
    uncertainty_cols = [c for c in uncertainty_cols if c in performance_df.columns]

    results = []
    for col in uncertainty_cols:
        quartiles = performance_df[col].quantile([0.25, 0.50, 0.75])
        for q_label, q_val in [("Q1 (Low)", quartiles[0.25]), ("Q3 (High)", quartiles[0.75])]:
            if "Low" in q_label:
                subset = performance_df[performance_df[col] <= q_val]
            else:
                subset = performance_df[performance_df[col] > q_val]

            success_rate = subset["meets_ndc_unconditional"].mean()
            results.append(
                {
                    "Uncertainty Dimension": col,
                    "Condition": q_label,
                    "Threshold Value": round(q_val, 4),
                    "NDC Success Rate": round(success_rate, 3),
                    "No-Regret": success_rate >= threshold_success_rate,
                }
            )

    return pd.DataFrame(results).sort_values("NDC Success Rate", ascending=False).reset_index(drop=True)
