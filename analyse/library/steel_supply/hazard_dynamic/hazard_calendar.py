"""Shared Gregorian calendar operations for hazard and ABM data.

The heatwave baseline contract uses a named ``pandas.MultiIndex``. Exactly one
level must be named ``day_of_year`` and all other named levels identify the
facility (or country-sector) series. Baseline and target entity levels must
match. Missing and duplicate baseline keys are errors; they are never treated
as zero-valued losses.
"""

from __future__ import annotations
from steel_supply.pipeline_paths import science_path, bundle_root, data_path

import calendar
from collections.abc import Iterator
from datetime import date, timedelta
from numbers import Integral

import numpy as np
import pandas as pd


__all__ = [
    "align_2023_baseline",
    "date_from_day_of_year",
    "days_in_year",
    "iter_dates",
    "month_day_key",
]


def days_in_year(year: int) -> int:
    """Return the Gregorian length of *year*."""
    normalized_year = _validated_integer(year, "year")
    date(normalized_year, 1, 1)  # Validate the range supported by datetime.
    return 366 if calendar.isleap(normalized_year) else 365


def date_from_day_of_year(year: int, day_of_year: int) -> date:
    """Convert a one-based Gregorian day-of-year to a date.

    Raises
    ------
    ValueError
        If ``day_of_year`` falls outside the selected Gregorian year.
    """
    normalized_year = _validated_integer(year, "year")
    normalized_day = _validated_integer(day_of_year, "day_of_year")
    year_length = days_in_year(normalized_year)
    if not 1 <= normalized_day <= year_length:
        raise ValueError(
            f"day_of_year {normalized_day} is invalid for {normalized_year}; "
            f"expected 1..{year_length}"
        )
    return date(normalized_year, 1, 1) + timedelta(days=normalized_day - 1)


def month_day_key(year: int, day_of_year: int) -> tuple[int, int]:
    """Return ``(month, day)`` for a one-based Gregorian day-of-year."""
    selected_date = date_from_day_of_year(year, day_of_year)
    return selected_date.month, selected_date.day


def align_2023_baseline(
    baseline: pd.Series,
    target_index: pd.MultiIndex,
    target_year: int,
) -> pd.Series:
    """Align a 2023 baseline series to target-year facility-day keys.

    ``baseline.index`` and ``target_index`` must be named ``MultiIndex``
    objects with the same entity levels (in the same relative order) and one
    ``day_of_year`` level. The day level may occupy a different position in
    each index. Baseline values must use NumPy ``float32``/``float64`` or
    pandas nullable ``Float32``/``Float64`` dtype. Target dates map to the same
    month-day in 2023. A target February 29 maps to the arithmetic mean of that
    entity's 2023 February 28 and March 1 values, and both adjacent keys are
    mandatory. Extra baseline entities and dates are allowed.
    """
    if not isinstance(baseline, pd.Series):
        raise TypeError("baseline must be a pandas.Series")
    if not isinstance(baseline.index, pd.MultiIndex):
        raise TypeError("baseline.index must be a pandas.MultiIndex")
    if not isinstance(target_index, pd.MultiIndex):
        raise TypeError("target_index must be a pandas.MultiIndex")

    days_in_year(target_year)
    _validate_index_schema(baseline.index, "baseline.index")
    _validate_index_schema(target_index, "target_index")
    _validate_index_values(baseline.index, "baseline.index")
    _validate_index_values(target_index, "target_index")

    baseline_entities = _entity_level_names(baseline.index)
    target_entities = _entity_level_names(target_index)
    if baseline_entities != target_entities:
        raise ValueError(
            "baseline and target entity levels must match in name and order; "
            f"got {baseline_entities!r} and {target_entities!r}"
        )
    if baseline.index.has_duplicates:
        duplicates = baseline.index[baseline.index.duplicated()].unique().tolist()
        raise ValueError(f"duplicate baseline keys are not allowed: {duplicates!r}")
    if not _is_supported_floating_dtype(baseline.dtype):
        raise TypeError(
            "baseline dtype must be floating: expected float32, float64, "
            f"Float32, or Float64; got {baseline.dtype}"
        )
    if baseline.isna().any():
        missing_values = baseline.index[baseline.isna()].tolist()
        raise ValueError(f"baseline values must not be missing: {missing_values!r}")

    baseline_day_position = baseline.index.names.index("day_of_year")
    target_day_position = target_index.names.index("day_of_year")
    for baseline_key in baseline.index:
        date_from_day_of_year(2023, baseline_key[baseline_day_position])

    baseline_values = baseline.to_dict()
    aligned_values: list[float] = []
    for target_key in target_index:
        target_day = target_key[target_day_position]
        target_date = date_from_day_of_year(target_year, target_day)
        if (target_date.month, target_date.day) == (2, 29):
            adjacent_keys = (
                _baseline_key_from_target(
                    target_key,
                    target_index.names,
                    baseline.index.names,
                    baseline_day=59,
                ),
                _baseline_key_from_target(
                    target_key,
                    target_index.names,
                    baseline.index.names,
                    baseline_day=60,
                ),
            )
            missing = [key for key in adjacent_keys if key not in baseline_values]
            if missing:
                raise ValueError(
                    "February 29 requires both 2023 February 28 and March 1 "
                    f"baseline keys; missing {missing!r}"
                )
            aligned_values.append(
                float(baseline_values[adjacent_keys[0]]) / 2.0
                + float(baseline_values[adjacent_keys[1]]) / 2.0
            )
            continue

        baseline_day = date(2023, target_date.month, target_date.day).timetuple().tm_yday
        baseline_key = _baseline_key_from_target(
            target_key,
            target_index.names,
            baseline.index.names,
            baseline_day,
        )
        if baseline_key not in baseline_values:
            raise ValueError(f"missing required 2023 baseline key: {baseline_key!r}")
        aligned_values.append(baseline_values[baseline_key])

    typed_values = pd.array(aligned_values, dtype=baseline.dtype)
    return pd.Series(typed_values, index=target_index, name=baseline.name, copy=False)


def iter_dates(start: date, end_exclusive: date) -> Iterator[date]:
    """Yield dates from ``start`` up to, but not including, ``end_exclusive``."""
    current = start
    while current < end_exclusive:
        yield current
        current += timedelta(days=1)


def _validated_integer(value: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{label} must be an integer, got {type(value).__name__}")
    return int(value)


def _validate_index_schema(index: pd.MultiIndex, label: str) -> None:
    if index.names.count("day_of_year") != 1:
        raise ValueError(f"{label} must have exactly one level named 'day_of_year'")
    if any(name is None for name in index.names):
        raise ValueError(f"{label} must have names for every level")
    if len(set(index.names)) != len(index.names):
        raise ValueError(f"{label} level names must be unique")


def _validate_index_values(index: pd.MultiIndex, label: str) -> None:
    frame = index.to_frame(index=False)
    if frame["day_of_year"].isna().any():
        raise ValueError(f"{label} contains a missing day_of_year key")

    entity_names = _entity_level_names(index)
    if entity_names and frame[entity_names].isna().to_numpy().any():
        raise ValueError(f"{label} contains a missing entity key")


def _entity_level_names(index: pd.MultiIndex) -> list[object]:
    return [name for name in index.names if name != "day_of_year"]


def _is_supported_floating_dtype(dtype: object) -> bool:
    allowed = (
        np.dtype("float32"),
        np.dtype("float64"),
        pd.Float32Dtype(),
        pd.Float64Dtype(),
    )
    return any(dtype == candidate for candidate in allowed)


def _baseline_key_from_target(
    target_key: tuple[object, ...],
    target_names: list[object],
    baseline_names: list[object],
    baseline_day: int,
) -> tuple[object, ...]:
    target_values = dict(zip(target_names, target_key))
    return tuple(
        baseline_day if name == "day_of_year" else target_values[name]
        for name in baseline_names
    )

