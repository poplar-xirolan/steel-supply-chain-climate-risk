"""Small numerical checks for the Fig. 6 statistical supplement."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
from scipy import optimize, special, stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'library'))
from steel_supply.postprocessing.fig6_statistics import (
    bootstrap_ci, fractional_logit, permutation_spearman,
    prepare_data, weighted_pearson,
)


def fixture_data():
    return (
        pd.DataFrame({"country": ["AAA", "BBB", "CCC"], "E2ECI": [0., 50., 100.],
                      "fulfilment_percent": [75., 99.99999999999999, 100.00000000000003]}),
        pd.DataFrame({"Country": ["CCC", "AAA", "BBB"], "E2ECI": [100., 0., 50.],
                      "final_demand": [30., 10., 20.]}),
    )


def test_endpoint_ties_and_weights_are_aligned_without_mutating_raw_data():
    raw, scores = fixture_data()
    before = raw.copy(deep=True)
    result = prepare_data(raw, scores)
    assert result.country.tolist() == ["AAA", "BBB", "CCC"]
    assert result.satisfaction_pct.tolist() == [75., 100., 100.]
    assert result.final_demand.tolist() == [10., 20., 30.]
    assert stats.spearmanr(result.E2ECI, result.satisfaction_pct).statistic == pytest.approx(0.8660254037844387)
    pd.testing.assert_frame_equal(raw, before)


@pytest.mark.parametrize("problem", ["duplicate", "missing_weight", "score_mismatch", "outside", "nan", "zero_weight"])
def test_invalid_input_is_rejected_instead_of_silently_dropping_countries(problem):
    raw, scores = fixture_data()
    if problem == "duplicate":
        raw.loc[1, "country"] = "AAA"
    elif problem == "missing_weight":
        scores = scores.iloc[1:]
    elif problem == "score_mismatch":
        scores.loc[0, "E2ECI"] = 90.
    elif problem == "outside":
        raw.loc[0, "fulfilment_percent"] = 100.1
    elif problem == "nan":
        raw.loc[0, "E2ECI"] = np.nan
    else:
        scores.loc[0, "final_demand"] = 0.
    with pytest.raises(ValueError):
        prepare_data(raw, scores)


def test_weighted_correlation_matches_hand_calculation_and_weight_scale_is_irrelevant():
    x, y, w = np.array([0., 1., 2.]), np.array([0., 1., 0.]), np.array([1., 2., 3.])
    # Weighted covariance = -1/9; variances = 5/9 and 2/9.
    assert weighted_pearson(x, y, w) == pytest.approx(-1 / np.sqrt(10))
    assert weighted_pearson(x, y, 100 * w) == pytest.approx(-1 / np.sqrt(10))
    assert weighted_pearson(x, y, np.ones(3)) == pytest.approx(0., abs=1e-15)


def test_bootstrap_reranks_each_paired_sample_and_is_reproducible():
    x = np.arange(12.)
    y = np.array([0., 1., 1., 3., 2., 5., 8., 7., 6., 9., 9., 9.])
    sampled = np.random.default_rng(17).integers(0, len(x), size=(200, len(x)))
    independently = [stats.spearmanr(x[i], y[i]).statistic for i in sampled]
    expected = np.quantile(independently, [.025, .975])
    actual = bootstrap_ci(x, y, rank=True, n_resamples=200, seed=17)
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    assert actual == bootstrap_ci(x, y, rank=True, n_resamples=200, seed=17)


def test_permutation_p_uses_absolute_two_sided_statistic_and_plus_one_correction():
    x = np.arange(6.)
    y = np.array([1., 0., 2., 5., 4., 3.])
    observed = stats.spearmanr(x, y).statistic
    rng = np.random.default_rng(8)
    extreme = sum(abs(stats.spearmanr(x, rng.permutation(y)).statistic) >= abs(observed) - 1e-12
                  for _ in range(99))
    result = permutation_spearman(x, y, n_resamples=99, seed=8)
    assert result["estimate"] == pytest.approx(observed)
    assert result["p_value"] == (extreme + 1) / 100
    assert 0 < result["p_value"] <= 1


def test_fractional_logit_retains_endpoints_and_matches_independent_score_solution():
    x = np.arange(0., 100., 10.)
    y = np.array([0., .25, .2, .45, .4, .7, .65, .85, .9, 1.])
    design = np.column_stack([np.ones(len(x)), (x - x.mean()) / 10])
    reference = optimize.root(lambda b: design.T @ (y - special.expit(design @ b)), [0., .3])
    assert reference.success
    result = fractional_logit(x, y)
    np.testing.assert_allclose([result["intercept"], result["estimate"]], reference.x, atol=1e-7)
    assert result["n"] == 10
    # Independent HC1 sandwich, using the root solution, not the fitted helper.
    mu = special.expit(design @ reference.x)
    bread = np.linalg.inv(design.T @ ((mu * (1 - mu))[:, None] * design))
    meat = design.T @ (((y - mu) ** 2)[:, None] * design)
    expected_se = np.sqrt((bread @ meat @ bread * 10 / 8)[1, 1])
    assert result["standard_error"] == pytest.approx(expected_se, rel=1e-6)
    assert result["ci_low"] < result["estimate"] < result["ci_high"]
