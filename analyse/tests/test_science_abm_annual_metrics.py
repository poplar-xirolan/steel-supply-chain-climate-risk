"""Regression fixtures exercise annual functions without running the sealed grid."""

import ast
from collections import defaultdict
from datetime import date
from pathlib import Path
import sys

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1] / 'library'
sys.path.insert(0, str(ROOT))


def metric_functions():
    from steel_supply import abm_annual_metrics as metrics
    return {name:getattr(metrics,name) for name in ["overlap_pieces","annualize","add_loss"]}


def test_endpoint_year_filter_keeps_cross_boundary_mass_only_in_requested_years():
    annualize = metric_functions()["annualize"]
    steps = pd.DataFrame({
        "step": [0, 1], "start_date": ["2023-12-31", "2099-12-27"],
        "end_date_exclusive": ["2024-01-07", "2100-01-03"],
        "step_days": [7, 7], "mass": [70.0, 140.0],
    })
    annual = annualize(steps, years={2023, 2100}).set_index("year")
    assert annual.index.tolist() == [2023, 2100]
    assert annual.annual_mass.tolist() == [10.0, 40.0]
    assert annual.coverage_days.tolist() == [1, 2]
    assert annualize(steps).annual_mass.sum() == 210.0


def test_leap_year_loss_matches_figure_daily_rate_without_changing_annual_mass():
    from steel_supply.abm_figure_utils import add_baseline_loss

    annual = pd.DataFrame({
        "year": [2023, 2024, 2100], "coverage_days": [365, 366, 365],
        "annual_mass": [3650.0, 3660.0, 2920.0],
    })
    result = metric_functions()["add_loss"](annual)
    assert result.annual_mass.tolist() == [3650.0, 3660.0, 2920.0]
    assert result.loss_percent.tolist() == pytest.approx([0.0, 0.0, 20.0])
    figure = annual.assign(sensitivity_id="run", mean_step_mass=[70.0, 70.0, 56.0])
    assert result.loss_percent.tolist() == pytest.approx(add_baseline_loss(figure).loss_percent.tolist())


def test_sparse_country_flow_uses_the_observed_simulation_calendar():
    annualize = metric_functions()["annualize"]
    steps = pd.DataFrame({
        "step": [0, 0, 1], "country": ["A", "B", "B"],
        "start_date": ["2023-01-01", "2023-01-01", "2023-01-08"],
        "end_date_exclusive": ["2023-01-08", "2023-01-08", "2023-01-15"],
        "step_days": [7, 7, 7], "mass": [70.0, 140.0, 140.0],
    })
    annual = annualize(steps, groups=["country"]).set_index("country")
    assert annual.coverage_days.tolist() == [14, 14]
    assert annual.annual_mass.tolist() == [70.0, 280.0]
