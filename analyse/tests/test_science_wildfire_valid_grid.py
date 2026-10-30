from __future__ import annotations

from pathlib import Path
from datetime import datetime
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import xarray as xr


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'library'))

from steel_supply.hazard_dynamic.wildfire_valid_grid import (  # noqa: E402
    all_time_stable_mask,
    great_circle_km,
    latlon_to_unit_xyz,
    nearest_valid_cells,
    validate_monthly_time_axis,
    valid_mask_from_samples,
)


def test_valid_mask_uses_decoded_finite_values_and_fill_value():
    fill_value = np.float32(1.0e20)
    values = np.array(
        [
            [[0.0, fill_value, np.nan, np.inf], [2.0, 3.0, 4.0, 5.0]],
            [[1.0, fill_value, np.nan, np.inf], [6.0, 7.0, 8.0, 9.0]],
            [[2.0, fill_value, np.nan, np.inf], [10.0, 11.0, 12.0, 13.0]],
        ],
        dtype="float32",
    )
    data = xr.DataArray(values, dims=("time", "lat", "lon"))
    data.encoding["_FillValue"] = fill_value

    mask = valid_mask_from_samples(data, (0, 1, 2))

    np.testing.assert_array_equal(
        mask,
        np.array([[True, False, False, False], [True, True, True, True]]),
    )


def test_all_time_stable_mask_excludes_cell_missing_outside_three_samples():
    values = np.ones((5, 1, 2), dtype="float32")
    values[1, 0, 0] = np.nan
    data = xr.DataArray(values, dims=("time", "lat", "lon"))

    mask, qa = all_time_stable_mask(data, chunk_size=2)

    np.testing.assert_array_equal(mask, np.array([[False, True]]))
    assert qa == {
        "time_count": 5,
        "sample_indices": [0, 2, 4],
        "sample_valid_counts": [2, 2, 2],
        "sample_mask_stable": True,
        "sample_pair_diff_counts": [0, 0, 0],
        "ever_valid_cell_count": 2,
        "all_time_stable_cell_count": 1,
        "time_varying_cell_count": 1,
        "time_chunk_size": 2,
    }


def test_all_time_stable_mask_excludes_fill_value_in_any_month():
    fill_value = np.float32(1.0e20)
    values = np.ones((4, 1, 2), dtype="float32")
    values[1, 0, 0] = fill_value
    data = xr.DataArray(values, dims=("time", "lat", "lon"))
    data.encoding["_FillValue"] = fill_value

    mask, qa = all_time_stable_mask(data, chunk_size=3)

    np.testing.assert_array_equal(mask, np.array([[False, True]]))
    assert qa["time_varying_cell_count"] == 1


def test_all_time_stable_mask_rejects_source_with_no_stable_cells():
    values = np.array(
        [
            [[1.0, np.nan]],
            [[np.nan, 1.0]],
        ],
        dtype="float32",
    )
    data = xr.DataArray(values, dims=("time", "lat", "lon"))

    with pytest.raises(ValueError, match="no all-time-stable valid grid cells"):
        all_time_stable_mask(data, chunk_size=1)


def test_all_time_stable_mask_rejects_nonpositive_chunk_size():
    data = xr.DataArray(
        np.ones((1, 1, 1), dtype="float32"),
        dims=("time", "lat", "lon"),
    )

    with pytest.raises(ValueError, match="chunk_size must be a positive integer"):
        all_time_stable_mask(data, chunk_size=0)


def _time_axis(*values: datetime) -> xr.DataArray:
    return xr.DataArray(np.asarray(values, dtype=object), dims="time")


def test_monthly_time_axis_accepts_complete_single_year():
    values = tuple(datetime(2015, month, 1) for month in range(1, 13))

    qa = validate_monthly_time_axis(
        _time_axis(*values),
        "model_global_monthly_2015_2015.nc",
    )

    assert qa == {
        "start_year": 2015,
        "end_year": 2015,
        "expected_start_month": "2015-01",
        "expected_end_month": "2015-12",
        "actual_start_month": "2015-01",
        "actual_end_month": "2015-12",
        "expected_month_count": 12,
        "actual_month_count": 12,
        "calendar": "proleptic_gregorian",
        "strictly_increasing": True,
        "monthly_continuous": True,
        "filename_bounds_match": True,
        "time_contract_version": "wildfire-monthly-v1",
    }


def test_monthly_time_axis_accepts_december_to_january_transition():
    values = tuple(
        datetime(year, month, 1)
        for year in (2015, 2016)
        for month in range(1, 13)
    )

    qa = validate_monthly_time_axis(
        _time_axis(*values),
        "model_global_monthly_2015_2016.nc",
    )

    assert qa["actual_month_count"] == 24
    assert qa["monthly_continuous"] is True


def test_monthly_time_axis_rejects_gap():
    with pytest.raises(ValueError, match=r"missing calendar month.*2015-02"):
        validate_monthly_time_axis(
            _time_axis(datetime(2015, 1, 1), datetime(2015, 3, 1)),
            "model_global_monthly_2015_2015.nc",
        )


def test_monthly_time_axis_rejects_duplicate():
    with pytest.raises(ValueError, match=r"duplicate calendar month.*2015-01"):
        validate_monthly_time_axis(
            _time_axis(datetime(2015, 1, 1), datetime(2015, 1, 15)),
            "model_global_monthly_2015_2015.nc",
        )


def test_monthly_time_axis_rejects_reversed_values():
    with pytest.raises(ValueError, match="strictly increasing"):
        validate_monthly_time_axis(
            _time_axis(datetime(2015, 2, 1), datetime(2015, 1, 1)),
            "model_global_monthly_2015_2015.nc",
        )


def test_monthly_time_axis_rejects_wrong_filename_bounds():
    values = tuple(datetime(2015, month, 1) for month in range(1, 13))

    with pytest.raises(ValueError, match=r"expected 24 months.*actual 12"):
        validate_monthly_time_axis(
            _time_axis(*values),
            "model_global_monthly_2014_2015.nc",
        )


def test_monthly_time_axis_rejects_unsupported_filename():
    with pytest.raises(ValueError, match="unsupported wildfire NetCDF filename"):
        validate_monthly_time_axis(
            _time_axis(datetime(2015, 1, 1)),
            "model.nc",
        )


def test_latlon_to_unit_xyz_places_equator_and_pole_on_unit_sphere():
    xyz = latlon_to_unit_xyz(
        np.array([0.0, 0.0, 90.0]),
        np.array([0.0, 180.0, 45.0]),
    )

    np.testing.assert_allclose(
        xyz,
        np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
        atol=1e-15,
    )


def test_nearest_valid_cells_skips_nearest_ocean_fill_cell():
    result = nearest_valid_cells(
        grid_lat=np.array([0.0]),
        grid_lon=np.array([0.0, 1.0, 2.0]),
        valid_mask=np.array([[False, False, True]]),
        facility_lat=np.array([0.0]),
        facility_lon=np.array([0.1]),
    )

    assert result.loc[0, "grid_lat_index"] == 0
    assert result.loc[0, "grid_lon_index"] == 2
    assert result.loc[0, "selected_grid_lon"] == 2.0


def test_nearest_valid_cells_wraps_across_dateline():
    result = nearest_valid_cells(
        grid_lat=np.array([0.0]),
        grid_lon=np.array([-179.0, 170.0]),
        valid_mask=np.array([[True, True]]),
        facility_lat=np.array([0.0]),
        facility_lon=np.array([179.8]),
    )

    assert result.loc[0, "selected_grid_lon"] == -179.0


def test_nearest_valid_cells_breaks_exact_distance_ties_by_flat_grid_index():
    result = nearest_valid_cells(
        grid_lat=np.array([0.0]),
        grid_lon=np.array([-1.0, 1.0]),
        valid_mask=np.array([[True, True]]),
        facility_lat=np.array([0.0]),
        facility_lon=np.array([0.0]),
    )

    assert result.loc[0, "grid_lon_index"] == 0
    assert result.loc[0, "selected_grid_lon"] == -1.0


def test_nearest_valid_cells_has_no_distance_cutoff_for_far_facility():
    result = nearest_valid_cells(
        grid_lat=np.array([0.0]),
        grid_lon=np.array([0.0]),
        valid_mask=np.array([[True]]),
        facility_lat=np.array([80.0]),
        facility_lon=np.array([170.0]),
    )

    assert result.loc[0, "grid_lat_index"] == 0
    assert result.loc[0, "grid_lon_index"] == 0
    assert result.loc[0, "mapping_distance_km"] == pytest.approx(
        great_circle_km(80.0, 170.0, 0.0, 0.0)
    )
    assert result.loc[0, "mapping_distance_km"] > 10_000


def _write_facility_file(directory: Path) -> Path:
    path = directory / "facilities.parquet"
    pd.DataFrame(
        {
            "ID_internal": ["facility-a"],
            "Geo_lat": [0.0],
            "Geo_lon": [0.1],
        }
    ).to_parquet(path, index=False)
    return path


def _write_wildfire_file(
    path: Path,
    values: np.ndarray,
    *,
    time: pd.DatetimeIndex | None = None,
) -> Path:
    if time is None:
        time = pd.date_range("2015-01-01", periods=values.shape[0], freq="MS")
    data = xr.DataArray(
        values.astype("float32"),
        dims=("time", "lat", "lon"),
        coords={
            "time": time,
            "lat": [0.0],
            "lon": [0.0, 1.0, 2.0],
        },
        name="burntarea-total",
    )
    data.to_dataset().to_netcdf(
        path,
        encoding={"burntarea-total": {"_FillValue": np.float32(1.0e20)}},
    )
    return path


def _run_extractor(facilities_dir: Path, netcdf_dir: Path, output_dir: Path):
    script = Path(__file__).resolve().parents[1] / "03_climate_extraction/06_extract_wildfire.py"
    return subprocess.run(
        [
            sys.executable,
            str(script),
            "--facilities-dir",
            str(facilities_dir),
            "--netcdf-dir",
            str(netcdf_dir),
            "--output-dir",
            str(output_dir),
            "--include-nonformal-inputs",
        ],
        capture_output=True,
        text=True,
        check=False,
    )


def test_extractor_rejects_missing_calendar_month_with_failed_receipts(tmp_path):
    facilities_dir = tmp_path / "facilities"
    netcdf_dir = tmp_path / "netcdf"
    output_dir = tmp_path / "output"
    facilities_dir.mkdir()
    netcdf_dir.mkdir()
    _write_facility_file(facilities_dir)
    source = _write_wildfire_file(
        netcdf_dir / "model_global_monthly_2015_2015.nc",
        np.ones((2, 1, 3), dtype="float32"),
        time=pd.DatetimeIndex(["2015-01-01", "2015-03-01"]),
    )

    completed = _run_extractor(facilities_dir, netcdf_dir, output_dir)

    assert completed.returncode != 0
    per_file = json.loads(
        (output_dir / "receipts" / f"{source.stem}.json").read_text(
            encoding="utf-8"
        )
    )
    assert per_file["status"] == "FAILED"
    assert "missing calendar month" in per_file["checks"]["error"]
    assert per_file["row_counts"]["expected_time_rows"] == 12
    assert per_file["row_counts"]["actual_time_rows"] == 2
    assert per_file["coverage"]["expected_start_month"] == "2015-01"
    assert per_file["coverage"]["actual_end_month"] == "2015-03"
    assert per_file["coverage"]["calendar"] == "proleptic_gregorian"
    assert per_file["checks"]["time_monthly_continuous"] is False
    assert per_file["runtime"]["python_executable"]
    assert per_file["runtime"]["command"]
    stage = json.loads(
        (output_dir / "wildfire_extraction_receipt.json").read_text(
            encoding="utf-8"
        )
    )
    assert stage["status"] == "FAILED"
    assert stage["coverage"]["expected_month_count_across_files"] == 12
    assert stage["coverage"]["actual_month_count_across_files"] == 2
    assert stage["coverage"]["calendar_counts"] == {"proleptic_gregorian": 1}
    assert stage["checks"]["all_time_axes_monthly_continuous"] is False
    assert stage["runtime"] == per_file["runtime"]


def test_extractor_uses_vectorized_valid_indices_and_writes_mapping_receipt(tmp_path):
    facilities_dir = tmp_path / "facilities"
    netcdf_dir = tmp_path / "netcdf"
    output_dir = tmp_path / "output"
    facilities_dir.mkdir()
    netcdf_dir.mkdir()
    _write_facility_file(facilities_dir)
    fill = np.float32(1.0e20)
    values = np.full((12, 1, 3), fill, dtype="float32")
    values[:, 0, 2] = np.arange(3.0, 15.0, dtype="float32")
    source = _write_wildfire_file(
        netcdf_dir / "stable_global_monthly_2015_2015.nc",
        values,
    )

    completed = _run_extractor(facilities_dir, netcdf_dir, output_dir)

    assert completed.returncode == 0, completed.stderr
    result_path = output_dir / f"result_facilities_{source.stem}.parquet"
    result = pd.read_parquet(result_path)
    assert list(result.columns) == [
        "id_internal",
        "source_nc",
        "time",
        "burntarea-total",
        "facility_lat",
        "facility_lon",
        "selected_grid_lat",
        "selected_grid_lon",
        "mapping_distance_km",
    ]
    assert result["burntarea-total"].tolist() == list(
        np.arange(3.0, 15.0, dtype="float32")
    )
    assert result["selected_grid_lon"].tolist() == [2.0] * 12
    assert np.isfinite(result["mapping_distance_km"]).all()
    assert not result.duplicated(["id_internal", "source_nc", "time"]).any()

    mapping_path = output_dir / "receipts" / f"{source.stem}.mapping.parquet"
    mapping = pd.read_parquet(mapping_path)
    assert list(mapping.columns) == [
        "id_internal",
        "source_nc",
        "facility_lat",
        "facility_lon",
        "selected_grid_lat",
        "selected_grid_lon",
        "mapping_distance_km",
        "valid_mask_stable",
    ]
    assert mapping.to_dict("records") == [
        {
            "id_internal": "facility-a",
            "source_nc": source.name,
            "facility_lat": 0.0,
            "facility_lon": 0.1,
            "selected_grid_lat": 0.0,
            "selected_grid_lon": 2.0,
            "mapping_distance_km": pytest.approx(
                great_circle_km(0.0, 0.1, 0.0, 2.0)
            ),
            "valid_mask_stable": True,
        }
    ]
    per_file_receipt = json.loads(
        (output_dir / "receipts" / f"{source.stem}.json").read_text(
            encoding="utf-8"
        )
    )
    assert per_file_receipt["status"] == "PASS"
    assert per_file_receipt["checks"]["complete_finite_unique_keys"] is True
    assert per_file_receipt["checks"]["time_monthly_continuous"] is True
    assert per_file_receipt["coverage"]["expected_start_month"] == "2015-01"
    assert per_file_receipt["coverage"]["actual_end_month"] == "2015-12"
    assert per_file_receipt["row_counts"]["expected_time_rows"] == 12
    assert per_file_receipt["row_counts"]["actual_time_rows"] == 12
    assert per_file_receipt["runtime"]["python_executable"]
    assert per_file_receipt["runtime"]["python_version"]
    assert per_file_receipt["runtime"]["argv"]
    assert per_file_receipt["runtime"]["command"]
    assert set(per_file_receipt["runtime"]["dependency_versions"]) == {
        "numpy",
        "pandas",
        "xarray",
        "scipy",
        "cftime",
        "pyarrow",
    }
    stage_receipt = json.loads(
        (output_dir / "wildfire_extraction_receipt.json").read_text(encoding="utf-8")
    )
    assert stage_receipt["status"] == "PASS"
    assert stage_receipt["coverage"]["expected_month_count_across_files"] == 12
    assert stage_receipt["coverage"]["actual_month_count_across_files"] == 12
    assert stage_receipt["checks"]["all_time_axes_monthly_continuous"] is True
    assert stage_receipt["runtime"] == per_file_receipt["runtime"]


def test_extractor_maps_away_from_time_varying_cell_and_records_mask_qa(
    tmp_path,
):
    facilities_dir = tmp_path / "facilities"
    netcdf_dir = tmp_path / "netcdf"
    output_dir = tmp_path / "output"
    facilities_dir.mkdir()
    netcdf_dir.mkdir()
    _write_facility_file(facilities_dir)
    fill = np.float32(1.0e20)
    dynamic_values = np.full((12, 1, 3), fill, dtype="float32")
    dynamic_values[:, 0, 0] = 1.0
    dynamic_values[1, 0, 0] = fill
    dynamic_values[1:, 0, 1] = 4.0
    dynamic_values[:, 0, 2] = np.arange(3.0, 15.0, dtype="float32")
    dynamic_source = _write_wildfire_file(
        netcdf_dir / "a_unstable_global_monthly_2015_2015.nc",
        dynamic_values,
    )
    stable_values = np.full((12, 1, 3), fill, dtype="float32")
    stable_values[:, 0, 2] = np.arange(3.0, 15.0, dtype="float32")
    unprocessed_source = _write_wildfire_file(
        netcdf_dir / "b_stable_global_monthly_2015_2015.nc",
        stable_values,
    )

    completed = _run_extractor(facilities_dir, netcdf_dir, output_dir)

    assert completed.returncode == 0, completed.stderr
    result = pd.read_parquet(
        output_dir / f"result_facilities_{dynamic_source.stem}.parquet"
    )
    assert result["selected_grid_lon"].tolist() == [2.0] * 12
    assert result["burntarea-total"].tolist() == list(
        np.arange(3.0, 15.0, dtype="float32")
    )
    assert (
        output_dir / f"result_facilities_{unprocessed_source.stem}.parquet"
    ).exists()
    per_file_receipt = json.loads(
        (output_dir / "receipts" / f"{dynamic_source.stem}.json").read_text(
            encoding="utf-8"
        )
    )
    assert per_file_receipt["status"] == "PASS"
    assert per_file_receipt["coverage"]["time_varying_grid_cells"] == 2
    assert per_file_receipt["coverage"]["all_time_stable_grid_cells"] == 1
    assert per_file_receipt["checks"]["valid_cell_rule"] == (
        "decoded_finite_non_fill_for_every_source_month"
    )
    stage_receipt = json.loads(
        (output_dir / "wildfire_extraction_receipt.json").read_text(encoding="utf-8")
    )
    assert stage_receipt["status"] == "PASS"
    assert stage_receipt["coverage"]["processed_netcdf_files"] == 2
    assert str(dynamic_source.resolve()) in stage_receipt["input_hashes"]
    assert str(unprocessed_source.resolve()) in stage_receipt["input_hashes"]


def test_extractor_fails_closed_when_no_all_time_stable_cell_exists(tmp_path):
    facilities_dir = tmp_path / "facilities"
    netcdf_dir = tmp_path / "netcdf"
    output_dir = tmp_path / "output"
    facilities_dir.mkdir()
    netcdf_dir.mkdir()
    _write_facility_file(facilities_dir)
    fill = np.float32(1.0e20)
    values = np.full((12, 1, 3), fill, dtype="float32")
    for time_index in range(12):
        values[time_index, 0, time_index % 3] = float(time_index + 1)
    source = _write_wildfire_file(
        netcdf_dir / "no_stable_global_monthly_2015_2015.nc",
        values,
    )

    completed = _run_extractor(facilities_dir, netcdf_dir, output_dir)

    assert completed.returncode != 0
    per_file_receipt = json.loads(
        (output_dir / "receipts" / f"{source.stem}.json").read_text(
            encoding="utf-8"
        )
    )
    assert per_file_receipt["status"] == "FAILED"
    assert "no all-time-stable valid grid cells" in per_file_receipt["checks"]["error"]
    stage_receipt = json.loads(
        (output_dir / "wildfire_extraction_receipt.json").read_text(encoding="utf-8")
    )
    assert stage_receipt["status"] == "FAILED"
    assert str(source.resolve()) in stage_receipt["input_hashes"]
