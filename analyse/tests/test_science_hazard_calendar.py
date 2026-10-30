from __future__ import annotations

from datetime import date
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'library'))

from steel_supply.hazard_dynamic.hazard_calendar import (  # noqa: E402
    align_2023_baseline,
    date_from_day_of_year,
    days_in_year,
    iter_dates,
    month_day_key,
)


def baseline_for_one_facility(values: dict[int, float]) -> pd.Series:
    index = pd.MultiIndex.from_tuples(
        [("facility-a", day_of_year) for day_of_year in values],
        names=["ID_internal", "day_of_year"],
    )
    return pd.Series(list(values.values()), index=index, name="loss", dtype="float64")


def target_index_for_one_facility(*day_of_years: int) -> pd.MultiIndex:
    return pd.MultiIndex.from_tuples(
        [("facility-a", day_of_year) for day_of_year in day_of_years],
        names=["ID_internal", "day_of_year"],
    )


@pytest.mark.parametrize(
    ("year", "expected"),
    [(2023, 365), (2024, 366), (1900, 365), (2000, 366)],
)
def test_days_in_year_uses_gregorian_leap_year_rules(year: int, expected: int):
    assert days_in_year(year) == expected


@pytest.mark.parametrize(
    ("year", "day_of_year", "expected"),
    [
        (2023, 59, date(2023, 2, 28)),
        (2023, 60, date(2023, 3, 1)),
        (2023, 365, date(2023, 12, 31)),
        (2024, 59, date(2024, 2, 28)),
        (2024, 60, date(2024, 2, 29)),
        (2024, 61, date(2024, 3, 1)),
        (2024, 366, date(2024, 12, 31)),
    ],
)
def test_date_from_day_of_year_handles_calendar_boundaries(
    year: int, day_of_year: int, expected: date
):
    assert date_from_day_of_year(year, day_of_year) == expected


def test_non_leap_year_rejects_day_366():
    with pytest.raises(ValueError, match=r"day_of_year.*366.*2023"):
        date_from_day_of_year(2023, 366)


def test_leap_year_does_not_shift_after_february():
    assert month_day_key(2024, 61) == (3, 1)
    assert month_day_key(2023, 60) == (3, 1)


def test_iter_dates_is_end_exclusive_across_new_year():
    assert list(iter_dates(date(2023, 12, 30), date(2024, 1, 3))) == [
        date(2023, 12, 30),
        date(2023, 12, 31),
        date(2024, 1, 1),
        date(2024, 1, 2),
    ]


def test_non_leap_baseline_alignment_preserves_month_day_and_target_order():
    baseline = baseline_for_one_facility({59: 0.2, 60: 0.4, 365: 0.8})
    target = target_index_for_one_facility(365, 60, 59)

    aligned = align_2023_baseline(baseline, target, target_year=2023)

    expected = pd.Series([0.8, 0.4, 0.2], index=target, name="loss")
    pd.testing.assert_series_equal(aligned, expected)


def test_leap_baseline_alignment_handles_february_and_year_end():
    baseline = baseline_for_one_facility({59: 0.2, 60: 0.4, 365: 0.8})
    target = target_index_for_one_facility(59, 61, 366)

    aligned = align_2023_baseline(baseline, target, target_year=2024)

    expected = pd.Series([0.2, 0.4, 0.8], index=target, name="loss")
    pd.testing.assert_series_equal(aligned, expected)


def test_february_29_uses_adjacent_baseline_mean():
    baseline = baseline_for_one_facility({59: 0.2, 60: 0.4})
    target = target_index_for_one_facility(60)

    aligned = align_2023_baseline(baseline, target, target_year=2024)

    assert aligned.iloc[0] == pytest.approx(0.3)


def test_duplicate_baseline_keys_are_rejected():
    index = pd.MultiIndex.from_tuples(
        [("facility-a", 59), ("facility-a", 59)],
        names=["ID_internal", "day_of_year"],
    )
    baseline = pd.Series([0.2, 0.3], index=index, dtype="float64")

    with pytest.raises(ValueError, match="duplicate baseline keys"):
        align_2023_baseline(
            baseline,
            target_index_for_one_facility(59),
            target_year=2024,
        )


def test_missing_required_ordinary_baseline_key_is_rejected_without_zero_fill():
    baseline = baseline_for_one_facility({59: 0.2})

    with pytest.raises(ValueError, match="missing required 2023 baseline key"):
        align_2023_baseline(
            baseline,
            target_index_for_one_facility(61),
            target_year=2024,
        )


@pytest.mark.parametrize("missing_day", [59, 60])
def test_february_29_requires_both_adjacent_baseline_keys(missing_day: int):
    values = {59: 0.2, 60: 0.4}
    del values[missing_day]

    with pytest.raises(ValueError, match="February 29 requires both"):
        align_2023_baseline(
            baseline_for_one_facility(values),
            target_index_for_one_facility(60),
            target_year=2024,
        )


def test_alignment_requires_named_day_of_year_level():
    baseline = baseline_for_one_facility({59: 0.2})
    target = target_index_for_one_facility(59).set_names(
        ["ID_internal", "calendar_day"]
    )

    with pytest.raises(ValueError, match="target_index.*day_of_year"):
        align_2023_baseline(baseline, target, target_year=2024)


def test_alignment_requires_matching_entity_levels():
    baseline = baseline_for_one_facility({59: 0.2})
    target = target_index_for_one_facility(59).set_names(
        ["country_a3", "day_of_year"]
    )

    with pytest.raises(ValueError, match="entity levels"):
        align_2023_baseline(baseline, target, target_year=2024)


@pytest.mark.parametrize("dtype", ["float32", "float64", "Float32", "Float64"])
def test_ordinary_alignment_preserves_floating_dtype_name_and_target_index(dtype):
    target = target_index_for_one_facility(60, 59)
    baseline = pd.Series(
        pd.array([0.2, 0.4], dtype=dtype),
        index=target_index_for_one_facility(59, 60),
        name="baseline_loss",
    )

    aligned = align_2023_baseline(baseline, target, target_year=2023)

    assert aligned.dtype == baseline.dtype
    assert aligned.name == "baseline_loss"
    assert aligned.index is target
    assert aligned.tolist() == pytest.approx([0.4, 0.2])


@pytest.mark.parametrize("dtype", ["float32", "float64", "Float32", "Float64"])
def test_february_29_alignment_preserves_floating_dtype_name_and_target_index(
    dtype,
):
    target = target_index_for_one_facility(60)
    baseline = pd.Series(
        pd.array([0.2, 0.4], dtype=dtype),
        index=target_index_for_one_facility(59, 60),
        name="baseline_loss",
    )

    aligned = align_2023_baseline(baseline, target, target_year=2024)

    assert aligned.dtype == baseline.dtype
    assert aligned.name == "baseline_loss"
    assert aligned.index is target
    assert aligned.iloc[0] == pytest.approx(0.3)


@pytest.mark.parametrize("dtype", ["float32", "float64", "Float32", "Float64"])
def test_empty_target_preserves_floating_dtype_name_and_target_index(dtype):
    target = target_index_for_one_facility()
    baseline = pd.Series(
        pd.array([0.2], dtype=dtype),
        index=target_index_for_one_facility(59),
        name="baseline_loss",
    )

    aligned = align_2023_baseline(baseline, target, target_year=2024)

    assert aligned.empty
    assert aligned.dtype == baseline.dtype
    assert aligned.name == "baseline_loss"
    assert aligned.index is target


@pytest.mark.parametrize(
    ("values", "dtype"),
    [
        ([1, 2], "int64"),
        ([True, False], "bool"),
        ([0.2, 0.4], "object"),
    ],
)
def test_alignment_rejects_non_floating_baseline_dtypes(values, dtype):
    baseline = pd.Series(
        values,
        index=target_index_for_one_facility(59, 60),
        dtype=dtype,
    )

    with pytest.raises(TypeError, match="baseline dtype must be floating"):
        align_2023_baseline(
            baseline,
            target_index_for_one_facility(59),
            target_year=2024,
        )


@pytest.mark.parametrize("invalid_year", [True, 10000])
def test_empty_target_still_validates_target_year(invalid_year):
    baseline = baseline_for_one_facility({59: 0.2})

    with pytest.raises((TypeError, ValueError), match="year"):
        align_2023_baseline(
            baseline,
            target_index_for_one_facility(),
            target_year=invalid_year,
        )


def test_day_of_year_level_may_occupy_different_index_positions():
    baseline_index = pd.MultiIndex.from_tuples(
        [("MINE", 59, "AUS"), ("MINE", 60, "AUS")],
        names=["sector", "day_of_year", "country_a3"],
    )
    baseline = pd.Series(
        [0.2, 0.4], index=baseline_index, name="baseline_loss", dtype="float32"
    )
    target = pd.MultiIndex.from_tuples(
        [(60, "MINE", "AUS")],
        names=["day_of_year", "sector", "country_a3"],
    )

    aligned = align_2023_baseline(baseline, target, target_year=2024)

    assert aligned.index is target
    assert aligned.dtype == baseline.dtype
    assert aligned.name == baseline.name
    assert aligned.iloc[0] == pytest.approx(0.3)


def test_baseline_may_be_a_superset_of_required_entities_and_dates():
    baseline_index = pd.MultiIndex.from_tuples(
        [
            ("facility-a", 59),
            ("facility-a", 60),
            ("facility-a", 365),
            ("facility-b", 59),
        ],
        names=["ID_internal", "day_of_year"],
    )
    baseline = pd.Series(
        [0.2, 0.4, 0.8, 0.9],
        index=baseline_index,
        name="baseline_loss",
        dtype="float64",
    )
    target = target_index_for_one_facility(59)

    aligned = align_2023_baseline(baseline, target, target_year=2024)

    assert aligned.tolist() == [0.2]
    assert aligned.index is target


def _three_level_index(values: tuple[object, object, object]) -> pd.MultiIndex:
    return pd.MultiIndex.from_tuples(
        [values], names=["sector", "country_a3", "day_of_year"]
    )


@pytest.mark.parametrize("source", ["baseline.index", "target_index"])
@pytest.mark.parametrize("entity_position", [0, 1], ids=["sector", "country"])
@pytest.mark.parametrize(
    "missing_value", [None, np.nan, pd.NA], ids=["none", "numpy_nan", "pandas_na"]
)
def test_alignment_rejects_every_missing_entity_key(
    source, entity_position, missing_value
):
    complete_key = ["MINE", "AUS", 59]
    invalid_key = complete_key.copy()
    invalid_key[entity_position] = missing_value
    baseline_index = _three_level_index(
        tuple(invalid_key) if source == "baseline.index" else tuple(complete_key)
    )
    target = _three_level_index(
        tuple(invalid_key) if source == "target_index" else tuple(complete_key)
    )
    baseline = pd.Series([0.2], index=baseline_index, dtype="float64")

    with pytest.raises(ValueError, match=rf"{source}.*missing entity key"):
        align_2023_baseline(baseline, target, target_year=2024)


@pytest.mark.parametrize("source", ["baseline.index", "target_index"])
@pytest.mark.parametrize(
    "missing_value", [None, np.nan, pd.NA], ids=["none", "numpy_nan", "pandas_na"]
)
def test_alignment_rejects_every_missing_day_of_year_key(source, missing_value):
    invalid_key = ("MINE", "AUS", missing_value)
    complete_key = ("MINE", "AUS", 59)
    baseline_index = _three_level_index(
        invalid_key if source == "baseline.index" else complete_key
    )
    target = _three_level_index(
        invalid_key if source == "target_index" else complete_key
    )
    baseline = pd.Series([0.2], index=baseline_index, dtype="float64")

    with pytest.raises(ValueError, match=rf"{source}.*missing day_of_year key"):
        align_2023_baseline(baseline, target, target_year=2024)
