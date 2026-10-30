"""Shared vectorized helpers for peak-only trade-series correction."""

from __future__ import annotations
from steel_supply.pipeline_paths import science_path, bundle_root, data_path

import numpy as np


def single_peak_mask(
    values: np.ndarray,
    low_threshold: float = 10,
    high_threshold: float = 3,
) -> np.ndarray:
    """Return the existing threshold mask for isolated interior peaks."""
    previous = values[:, :-2]
    current = values[:, 1:-1]
    following = values[:, 2:]
    is_peak = (current > previous) & (current > following)
    threshold_factors = np.where(current < 1e9, low_threshold, high_threshold)
    with np.errstate(divide="ignore", invalid="ignore"):
        difference_from_previous = np.abs(current - previous) / previous
        difference_from_following = np.abs(current - following) / following
    return is_peak & (
        (difference_from_previous > threshold_factors)
        & (difference_from_following > threshold_factors)
    )


def double_peak_mask(
    values: np.ndarray,
    low_threshold: float = 10,
    high_threshold: float = 3,
) -> np.ndarray:
    """Return the existing threshold mask for adjacent interior peak pairs."""
    previous = values[:, :-3]
    current = values[:, 1:-2]
    following = values[:, 2:-1]
    following_following = values[:, 3:]
    average = (current + following) / 2
    is_peak = (average > previous) & (average > following_following)
    threshold_factors = np.where(current < 1e9, low_threshold, high_threshold)
    with np.errstate(divide="ignore", invalid="ignore"):
        difference_from_previous = np.abs(average - previous) / previous
        difference_from_following = (
            np.abs(average - following_following) / following_following
        )
    return is_peak & (
        (difference_from_previous > threshold_factors)
        & (difference_from_following > threshold_factors)
    )


def correct_single_and_double_peaks(
    values: np.ndarray,
) -> tuple[np.ndarray, dict[str, int]]:
    """Correct peaks in a real numeric 2-D array without mutating the input."""
    values_array = np.asarray(values)
    if values_array.ndim != 2:
        raise ValueError("values must be a 2-D array")
    if not np.issubdtype(values_array.dtype, np.number) or np.issubdtype(
        values_array.dtype,
        np.complexfloating,
    ):
        raise TypeError("values must have a real numeric dtype")

    floating_dtype = np.result_type(values_array.dtype, np.float64)
    corrected = values_array.astype(floating_dtype, copy=True)

    single_peaks = single_peak_mask(corrected)
    before_single = corrected.copy()
    corrected[:, 1:-1][single_peaks] = (
        before_single[:, :-2][single_peaks] + before_single[:, 2:][single_peaks]
    ) / 2

    double_peaks = double_peak_mask(corrected)
    before_double = corrected.copy()
    double_replacements = (
        before_double[:, :-3][double_peaks] + before_double[:, 3:][double_peaks]
    ) / 2
    corrected[:, 1:-2][double_peaks] = double_replacements
    corrected[:, 2:-1][double_peaks] = double_replacements

    return corrected, {
        "single_peaks": int(np.count_nonzero(single_peaks)),
        "double_peaks": int(np.count_nonzero(double_peaks)),
    }

