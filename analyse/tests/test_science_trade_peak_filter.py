from pathlib import Path
import sys

import numpy as np
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'library'))

from steel_supply.mfa.trade_peak_filter import (  # noqa: E402
    correct_single_and_double_peaks,
    double_peak_mask,
    single_peak_mask,
)


def test_valley_is_never_replaced():
    values = np.array([[100.0, 1.0, 100.0]])

    corrected, counts = correct_single_and_double_peaks(values)

    np.testing.assert_array_equal(corrected, values)
    assert counts == {"single_peaks": 0, "double_peaks": 0}


def test_single_peak_keeps_existing_formula():
    values = np.array([[100.0, 5000.0, 300.0]])

    corrected, counts = correct_single_and_double_peaks(values)

    assert corrected[0, 1] == 200.0
    assert counts["single_peaks"] == 1


def test_double_peak_keeps_existing_formula():
    values = np.array([[100.0, 5000.0, 6000.0, 300.0]])

    corrected, counts = correct_single_and_double_peaks(values)

    np.testing.assert_array_equal(corrected, [[100.0, 200.0, 200.0, 300.0]])
    assert counts["double_peaks"] == 1


def test_single_peak_uses_high_threshold_at_one_billion_boundary():
    values = np.array([[200_000_000.0, 1_000_000_000.0, 200_000_000.0]])

    corrected, counts = correct_single_and_double_peaks(values)

    np.testing.assert_array_equal(
        corrected,
        [[200_000_000.0, 200_000_000.0, 200_000_000.0]],
    )
    assert counts == {"single_peaks": 1, "double_peaks": 0}


def test_single_peak_uses_low_threshold_below_one_billion_boundary():
    values = np.array([[200_000_000.0, 999_999_999.0, 200_000_000.0]])

    corrected, counts = correct_single_and_double_peaks(values)

    np.testing.assert_array_equal(corrected, values)
    assert counts == {"single_peaks": 0, "double_peaks": 0}


def test_zero_placeholders_follow_existing_peak_formula():
    values = np.array([[1e-6, 1.0, 1e-6]])

    corrected, counts = correct_single_and_double_peaks(values)

    np.testing.assert_array_equal(corrected, [[1e-6, 1e-6, 1e-6]])
    assert counts == {"single_peaks": 1, "double_peaks": 0}


@pytest.mark.parametrize(
    "values",
    [
        np.array([[5000.0, 100.0, 300.0]]),
        np.array([[100.0, 300.0, 5000.0]]),
    ],
    ids=["first-year", "last-year"],
)
def test_first_and_last_years_are_never_replaced(values):
    corrected, counts = correct_single_and_double_peaks(values)

    np.testing.assert_array_equal(corrected, values)
    assert counts == {"single_peaks": 0, "double_peaks": 0}


@pytest.mark.parametrize(
    ("series_name", "values", "expected"),
    [
        (
            "weight_unified_kg",
            np.array([[10.0, 500.0, 30.0]]),
            np.array([[10.0, 20.0, 30.0]]),
        ),
        (
            "monetary_usd",
            np.array([[1_000.0, 50_000.0, 3_000.0]]),
            np.array([[1_000.0, 2_000.0, 3_000.0]]),
        ),
    ],
)
def test_weight_and_monetary_value_paths_use_peak_only_correction(
    series_name,
    values,
    expected,
):
    corrected, counts = correct_single_and_double_peaks(values)

    np.testing.assert_array_equal(corrected, expected, err_msg=series_name)
    assert counts == {"single_peaks": 1, "double_peaks": 0}


def test_masks_select_only_peaks_and_preserve_expected_shapes():
    single_values = np.array(
        [
            [100.0, 5000.0, 300.0],
            [100.0, 1.0, 100.0],
        ]
    )
    double_values = np.array(
        [
            [100.0, 5000.0, 6000.0, 300.0],
            [100.0, 1.0, 1.0, 100.0],
        ]
    )

    np.testing.assert_array_equal(single_peak_mask(single_values), [[True], [False]])
    np.testing.assert_array_equal(double_peak_mask(double_values), [[True], [False]])


def test_input_array_is_not_mutated():
    values = np.array([[100.0, 5000.0, 300.0]])
    original = values.copy()

    correct_single_and_double_peaks(values)

    np.testing.assert_array_equal(values, original)


def test_integer_input_is_promoted_before_fractional_replacement():
    values = np.array([[1, 50, 4]], dtype=np.int64)

    corrected, counts = correct_single_and_double_peaks(values)

    np.testing.assert_array_equal(corrected, [[1.0, 2.5, 4.0]])
    assert np.issubdtype(corrected.dtype, np.floating)
    assert counts == {"single_peaks": 1, "double_peaks": 0}


@pytest.mark.parametrize(
    "values",
    [
        np.array([1.0, 50.0, 4.0]),
        np.ones((1, 1, 3), dtype=np.float64),
    ],
    ids=["one-dimensional", "three-dimensional"],
)
def test_non_2d_input_is_rejected_with_clear_error(values):
    with pytest.raises(ValueError, match="2-D"):
        correct_single_and_double_peaks(values)


@pytest.mark.parametrize(
    "values",
    [
        np.array([["1", "50", "4"]]),
        np.array([[1, 50, 4]], dtype=object),
    ],
    ids=["string", "object"],
)
def test_non_numeric_input_is_rejected_with_clear_error(values):
    with pytest.raises(TypeError, match="numeric"):
        correct_single_and_double_peaks(values)


def test_float64_production_dtype_and_values_are_preserved():
    values = np.array([[100.0, 5000.0, 300.0]], dtype=np.float64)

    corrected, counts = correct_single_and_double_peaks(values)

    np.testing.assert_array_equal(corrected, [[100.0, 200.0, 300.0]])
    assert corrected.dtype == np.dtype(np.float64)
    assert counts == {"single_peaks": 1, "double_peaks": 0}
