"""Shared nearest-valid-grid helpers for monthly wildfire extraction."""

from __future__ import annotations
from steel_supply.pipeline_paths import science_path, bundle_root, data_path

from datetime import datetime
from numbers import Integral
import re

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
import xarray as xr

try:
    import cftime
except ImportError:  # pragma: no cover - only environments without cftime
    cftime = None


EARTH_RADIUS_KM = 6371.0088
TIME_CONTRACT_VERSION = "wildfire-monthly-v1"
WILDFIRE_FILENAME_PATTERN = re.compile(
    r"^.+_global_monthly_(?P<start_year>\d{4})_(?P<end_year>\d{4})\.nc$"
)


class MonthlyTimeAxisError(ValueError):
    """Time-contract failure carrying safe partial QA for FAILED receipts."""

    def __init__(self, message: str, qa: dict[str, object]) -> None:
        super().__init__(message)
        self.qa = qa


def _fill_values(data: xr.DataArray) -> tuple[object, ...]:
    values = []
    for metadata in (data.encoding, data.attrs):
        for key in ("_FillValue", "missing_value"):
            value = metadata.get(key)
            if value is not None:
                values.extend(np.asarray(value).reshape(-1).tolist())
    return tuple(values)


def _decoded_valid_values(data: xr.DataArray, values: np.ndarray) -> np.ndarray:
    mask = np.isfinite(values)
    for fill_value in _fill_values(data):
        mask &= values != fill_value
    return mask


def valid_mask_from_samples(
    data: xr.DataArray,
    sample_indices: tuple[int, int, int],
) -> np.ndarray:
    """Return a decoded finite-value mask from sampled time slices."""
    samples = np.stack(
        [np.asarray(data.isel(time=index).values) for index in sample_indices]
    )
    masks = _decoded_valid_values(data, samples)
    return masks[0]


def all_time_stable_mask(
    data: xr.DataArray,
    chunk_size: int = 120,
) -> tuple[np.ndarray, dict[str, object]]:
    """Return cells valid in every month, scanning time in bounded chunks."""
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, Integral) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    if tuple(data.dims) != ("time", "lat", "lon"):
        raise ValueError(
            "data dimensions must be exactly ('time', 'lat', 'lon')"
        )
    time_count = data.sizes["time"]
    if time_count < 1:
        raise ValueError("source time dimension is empty")

    sample_indices = (0, time_count // 2, time_count - 1)
    sample_values = np.stack(
        [np.asarray(data.isel(time=index).values) for index in sample_indices]
    )
    sample_masks = _decoded_valid_values(data, sample_values)
    spatial_shape = (data.sizes["lat"], data.sizes["lon"])
    ever_valid = np.zeros(spatial_shape, dtype=bool)
    all_valid = np.ones(spatial_shape, dtype=bool)
    for start in range(0, time_count, int(chunk_size)):
        stop = min(start + int(chunk_size), time_count)
        values = np.asarray(data.isel(time=slice(start, stop)).values)
        valid = _decoded_valid_values(data, values)
        ever_valid |= valid.any(axis=0)
        all_valid &= valid.all(axis=0)

    stable_count = int(all_valid.sum())
    if stable_count == 0:
        raise ValueError("source contains no all-time-stable valid grid cells")
    pair_differences = [
        int(np.count_nonzero(sample_masks[0] ^ sample_masks[1])),
        int(np.count_nonzero(sample_masks[0] ^ sample_masks[2])),
        int(np.count_nonzero(sample_masks[1] ^ sample_masks[2])),
    ]
    qa: dict[str, object] = {
        "time_count": time_count,
        "sample_indices": list(sample_indices),
        "sample_valid_counts": [int(mask.sum()) for mask in sample_masks],
        "sample_mask_stable": not any(pair_differences),
        "sample_pair_diff_counts": pair_differences,
        "ever_valid_cell_count": int(ever_valid.sum()),
        "all_time_stable_cell_count": stable_count,
        "time_varying_cell_count": int((ever_valid & ~all_valid).sum()),
        "time_chunk_size": int(chunk_size),
    }
    return all_valid, qa


def validate_monthly_time_axis(
    time: xr.DataArray,
    source_name: str,
) -> dict[str, object]:
    """Validate a decoded, complete monthly axis against source filename bounds."""
    values = np.asarray(time.values).reshape(-1)
    calendar = time.encoding.get("calendar") or time.attrs.get("calendar")
    if calendar is None and values.size:
        calendar = getattr(values[0], "calendar", "proleptic_gregorian")
    qa: dict[str, object] = {
        "start_year": None,
        "end_year": None,
        "expected_start_month": None,
        "expected_end_month": None,
        "actual_start_month": None,
        "actual_end_month": None,
        "expected_month_count": None,
        "actual_month_count": int(values.size),
        "calendar": None if calendar is None else str(calendar),
        "strictly_increasing": False,
        "monthly_continuous": False,
        "filename_bounds_match": False,
        "time_contract_version": TIME_CONTRACT_VERSION,
    }
    filename_match = WILDFIRE_FILENAME_PATTERN.fullmatch(source_name)
    if filename_match is None:
        raise MonthlyTimeAxisError(
            "unsupported wildfire NetCDF filename; expected suffix "
            f"'_global_monthly_YYYY_YYYY.nc': {source_name}",
            qa,
        )
    start_year = int(filename_match.group("start_year"))
    end_year = int(filename_match.group("end_year"))
    qa.update(
        {
            "start_year": start_year,
            "end_year": end_year,
            "expected_start_month": f"{start_year:04d}-01",
            "expected_end_month": f"{end_year:04d}-12",
            "expected_month_count": (end_year - start_year + 1) * 12,
        }
    )
    if end_year < start_year:
        raise MonthlyTimeAxisError(
            f"wildfire filename end year {end_year} precedes start year {start_year}",
            qa,
        )

    if values.size == 0:
        raise MonthlyTimeAxisError("source time coordinate is empty", qa)
    decoded_types = (datetime, np.datetime64)
    if cftime is not None:
        decoded_types = (*decoded_types, cftime.datetime)
    if any(
        not isinstance(value, decoded_types)
        or (isinstance(value, np.datetime64) and np.isnat(value))
        for value in values
    ):
        bad_type = next(
            type(value).__name__
            for value in values
            if not isinstance(value, decoded_types)
            or (isinstance(value, np.datetime64) and np.isnat(value))
        )
        raise MonthlyTimeAxisError(
            "source time coordinate must contain decoded cftime/datetime values; "
            f"found {bad_type}",
            qa,
        )

    month_keys = []
    for value in values:
        if isinstance(value, np.datetime64):
            year_text, month_text = np.datetime_as_string(value, unit="M").split("-")
            month_keys.append((int(year_text), int(month_text)))
        else:
            month_keys.append((int(value.year), int(value.month)))
    qa["actual_start_month"] = f"{month_keys[0][0]:04d}-{month_keys[0][1]:02d}"
    qa["actual_end_month"] = f"{month_keys[-1][0]:04d}-{month_keys[-1][1]:02d}"
    for position, (previous, current) in enumerate(
        zip(month_keys, month_keys[1:], strict=False),
        start=1,
    ):
        if current == previous:
            raise MonthlyTimeAxisError(
                f"duplicate calendar month {current[0]:04d}-{current[1]:02d} "
                f"at time index {position}",
                qa,
            )
        previous_ordinal = previous[0] * 12 + previous[1] - 1
        current_ordinal = current[0] * 12 + current[1] - 1
        if current_ordinal < previous_ordinal:
            raise MonthlyTimeAxisError(
                "source time coordinate must be strictly increasing; "
                f"{current[0]:04d}-{current[1]:02d} follows "
                f"{previous[0]:04d}-{previous[1]:02d} at time index {position}",
                qa,
            )
        if current_ordinal != previous_ordinal + 1:
            missing_ordinal = previous_ordinal + 1
            missing_year, zero_based_month = divmod(missing_ordinal, 12)
            raise MonthlyTimeAxisError(
                f"missing calendar month {missing_year:04d}-{zero_based_month + 1:02d} "
                f"before time index {position}",
                qa,
            )

    expected_month_count = (end_year - start_year + 1) * 12
    actual_month_count = len(month_keys)
    if actual_month_count != expected_month_count:
        raise MonthlyTimeAxisError(
            f"filename bounds expected {expected_month_count} months; actual "
            f"{actual_month_count} months",
            qa,
        )
    expected_start = (start_year, 1)
    expected_end = (end_year, 12)
    if month_keys[0] != expected_start:
        raise MonthlyTimeAxisError(
            f"filename start requires {start_year:04d}-01 but time starts at "
            f"{month_keys[0][0]:04d}-{month_keys[0][1]:02d}",
            qa,
        )
    if month_keys[-1] != expected_end:
        raise MonthlyTimeAxisError(
            f"filename end requires {end_year:04d}-12 but time ends at "
            f"{month_keys[-1][0]:04d}-{month_keys[-1][1]:02d}",
            qa,
        )

    qa.update(
        {
            "strictly_increasing": True,
            "monthly_continuous": True,
            "filename_bounds_match": True,
        }
    )
    return qa


def latlon_to_unit_xyz(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Convert latitude/longitude degrees to Cartesian unit-sphere points."""
    latitude, longitude = np.broadcast_arrays(
        np.asarray(lat, dtype="float64"),
        np.asarray(lon, dtype="float64"),
    )
    if not np.isfinite(latitude).all() or not np.isfinite(longitude).all():
        raise ValueError("latitude and longitude must be finite")
    if np.any((latitude < -90.0) | (latitude > 90.0)):
        raise ValueError("latitude must be within [-90, 90] degrees")

    lat_radians = np.deg2rad(latitude)
    lon_radians = np.deg2rad(longitude)
    cos_lat = np.cos(lat_radians)
    return np.stack(
        (
            cos_lat * np.cos(lon_radians),
            cos_lat * np.sin(lon_radians),
            np.sin(lat_radians),
        ),
        axis=-1,
    )


def great_circle_km(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float,
) -> float:
    """Return the shortest great-circle distance in kilometres."""
    points = latlon_to_unit_xyz(
        np.array([lat1, lat2], dtype="float64"),
        np.array([lon1, lon2], dtype="float64"),
    )
    central_angle = np.arctan2(
        np.linalg.norm(np.cross(points[0], points[1])),
        np.dot(points[0], points[1]),
    )
    return float(EARTH_RADIUS_KM * central_angle)


def nearest_valid_cells(
    grid_lat: np.ndarray,
    grid_lon: np.ndarray,
    valid_mask: np.ndarray,
    facility_lat: np.ndarray,
    facility_lon: np.ndarray,
) -> pd.DataFrame:
    """Map facilities to nearest valid grid centers with stable index ties."""
    latitudes = np.asarray(grid_lat, dtype="float64")
    longitudes = np.asarray(grid_lon, dtype="float64")
    mask = np.asarray(valid_mask, dtype=bool)
    target_lat = np.asarray(facility_lat, dtype="float64")
    target_lon = np.asarray(facility_lon, dtype="float64")

    if latitudes.ndim != 1 or longitudes.ndim != 1:
        raise ValueError("grid_lat and grid_lon must be one-dimensional")
    if mask.shape != (latitudes.size, longitudes.size):
        raise ValueError(
            "valid_mask shape must equal (len(grid_lat), len(grid_lon))"
        )
    if target_lat.ndim != 1 or target_lon.ndim != 1:
        raise ValueError("facility_lat and facility_lon must be one-dimensional")
    if target_lat.shape != target_lon.shape:
        raise ValueError("facility_lat and facility_lon must have matching shapes")
    if not mask.any():
        raise ValueError("valid_mask contains no valid grid cells")

    grid_lat_2d, grid_lon_2d = np.meshgrid(
        latitudes,
        longitudes,
        indexing="ij",
    )
    valid_flat_indices = np.flatnonzero(mask.ravel(order="C"))
    valid_xyz = latlon_to_unit_xyz(
        grid_lat_2d.ravel(order="C")[valid_flat_indices],
        grid_lon_2d.ravel(order="C")[valid_flat_indices],
    )
    facility_xyz = latlon_to_unit_xyz(target_lat, target_lon)

    tree = cKDTree(valid_xyz)
    nearest_chord, nearest_position = tree.query(facility_xyz, k=1)
    selected_positions = np.asarray(nearest_position, dtype=np.intp).reshape(-1)
    nearest_chord = np.asarray(nearest_chord, dtype="float64").reshape(-1)

    for facility_index, (point, chord) in enumerate(
        zip(facility_xyz.reshape(-1, 3), nearest_chord, strict=True)
    ):
        radius = np.nextafter(chord, np.inf)
        candidate_positions = np.asarray(
            tree.query_ball_point(point, radius),
            dtype=np.intp,
        )
        candidate_distances = np.sum(
            (valid_xyz[candidate_positions] - point) ** 2,
            axis=1,
        )
        minimum = candidate_distances.min()
        tolerance = 8.0 * np.finfo("float64").eps * max(1.0, minimum)
        tied_positions = candidate_positions[
            np.abs(candidate_distances - minimum) <= tolerance
        ]
        selected_positions[facility_index] = tied_positions[
            np.argmin(valid_flat_indices[tied_positions])
        ]

    selected_flat = valid_flat_indices[selected_positions]
    selected_lat_index, selected_lon_index = np.unravel_index(
        selected_flat,
        mask.shape,
        order="C",
    )
    selected_lat = latitudes[selected_lat_index]
    selected_lon = longitudes[selected_lon_index]
    dot_products = np.einsum(
        "ij,ij->i",
        facility_xyz.reshape(-1, 3),
        valid_xyz[selected_positions],
    )
    cross_norms = np.linalg.norm(
        np.cross(
            facility_xyz.reshape(-1, 3),
            valid_xyz[selected_positions],
        ),
        axis=1,
    )
    distances = EARTH_RADIUS_KM * np.arctan2(cross_norms, dot_products)

    return pd.DataFrame(
        {
            "grid_lat_index": selected_lat_index,
            "grid_lon_index": selected_lon_index,
            "selected_grid_lat": selected_lat,
            "selected_grid_lon": selected_lon,
            "mapping_distance_km": distances,
        }
    )

