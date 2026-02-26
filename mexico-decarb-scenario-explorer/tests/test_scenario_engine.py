"""
tests/test_scenario_engine.py
Unit tests for the ScenarioEngine and scenario output validation.
"""

import pytest
import pandas as pd
import numpy as np
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.scenario_engine import ScenarioEngine, BASELINE_EMISSIONS_2020


@pytest.fixture
def engine():
    return ScenarioEngine(start_year=2020, end_year=2040, n_mc_runs=50)


@pytest.fixture
def results(engine):
    return engine.run_all()


class TestScenarioEngine:

    def test_all_scenarios_present(self, results):
        assert set(results.keys()) == {"BAU", "NDC_unconditional", "NDC_conditional", "NZE_1.5C"}

    def test_trajectory_length(self, results, engine):
        expected_years = len(range(engine.start_year, engine.end_year + 1))
        for sid, r in results.items():
            assert len(r.total_emissions) == expected_years, f"{sid} trajectory length mismatch"

    def test_baseline_year_matches_input(self, results):
        baseline_total = sum(BASELINE_EMISSIONS_2020.values())
        for sid, r in results.items():
            first_val = r.total_emissions[0]
            assert abs(first_val - baseline_total) < 1.0, (
                f"{sid}: 2020 emissions {first_val:.1f} != baseline {baseline_total:.1f}"
            )

    def test_nze_lower_than_ndc_at_2030(self, results):
        idx_2030 = results["NZE_1.5C"].years.index(2030)
        nze_2030 = results["NZE_1.5C"].total_emissions[idx_2030]
        ndc_cond_2030 = results["NDC_conditional"].total_emissions[idx_2030]
        assert nze_2030 < ndc_cond_2030

    def test_ndc_lower_than_bau(self, results):
        idx_2030 = results["BAU"].years.index(2030)
        bau = results["BAU"].total_emissions[idx_2030]
        ndc = results["NDC_conditional"].total_emissions[idx_2030]
        assert ndc < bau

    def test_mc_bands_contain_deterministic(self, results):
        for sid, r in results.items():
            for lo, det, hi in zip(r.mc_lower, r.total_emissions, r.mc_upper):
                assert lo <= det + 0.5, f"{sid}: deterministic below MC lower"
                assert hi >= det - 0.5, f"{sid}: deterministic above MC upper"

    def test_pct_reduction_positive_for_mitigation(self, results):
        for sid in ["NDC_unconditional", "NDC_conditional", "NZE_1.5C"]:
            assert results[sid].pct_reduction_vs_2020 > 0, (
                f"{sid} should show positive reduction vs 2020"
            )

    def test_sector_emissions_dataframe_shape(self, results, engine):
        n_years = engine.end_year - engine.start_year + 1
        n_sectors = len(BASELINE_EMISSIONS_2020)
        for sid, r in results.items():
            assert r.sector_emissions.shape == (n_years, n_sectors), (
                f"{sid} sector matrix shape mismatch"
            )

    def test_sector_sums_match_total(self, results):
        for sid, r in results.items():
            sector_total = r.sector_emissions.sum(axis=1).values
            for i, (st, tt) in enumerate(zip(sector_total, r.total_emissions)):
                assert abs(st - tt) < 0.01, (
                    f"{sid} year index {i}: sector sum {st:.2f} != total {tt:.2f}"
                )

    def test_robustness_table_structure(self, results, engine):
        rob = engine.robustness_table(results)
        assert isinstance(rob, pd.DataFrame)
        assert "Escenario" in rob.columns
        assert "Emisiones 2030 (MtCO₂e)" in rob.columns
        assert len(rob) == 4


class TestScenarioResultMetrics:

    def test_cumulative_reductions_non_negative_for_mitigation(self, results):
        for sid in ["NDC_unconditional", "NDC_conditional", "NZE_1.5C"]:
            # By 2030, cumulative reductions vs BAU should be positive
            idx_2030 = results[sid].years.index(2030)
            cum_red = results[sid].cumulative_reductions[idx_2030]
            assert cum_red >= 0, f"{sid}: negative cumulative reductions at 2030"

    def test_bau_zero_cumulative_reductions(self, results):
        for v in results["BAU"].cumulative_reductions:
            assert v == 0.0
