#!/usr/bin/env python
# coding: utf-8

"""Extract monthly wildfire values from each model's nearest valid land cell."""

from __future__ import annotations

from pathlib import Path as _DataPath
import sys as _data_sys
_ANALYSE_ROOT = next(p for p in _DataPath(__file__).resolve().parents if (p / 'run_analyse.py').is_file())
if str(_ANALYSE_ROOT / 'library') not in _data_sys.path:
    _data_sys.path.insert(0, str(_ANALYSE_ROOT / 'library'))
from steel_supply.pipeline_paths import data_path, science_path, bundle_root


import argparse
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import os
from pathlib import Path
import platform
import subprocess
import sys
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr


ROOT = bundle_root()
sys.path.insert(0, str(ROOT))

from steel_supply.pipeline_receipts import sha256_file, write_receipt  # noqa: E402
from steel_supply.hazard_dynamic.wildfire_valid_grid import (  # noqa: E402
    MonthlyTimeAxisError,
    TIME_CONTRACT_VERSION,
    all_time_stable_mask,
    nearest_valid_cells,
    validate_monthly_time_axis,
)
from steel_supply.hazard_dynamic.wildfire_coverage import (  # noqa: E402
    FORMAL_CLIMATE_SCENARIOS,
    FORMAL_END_YEAR,
    FORMAL_START_YEAR,
    select_formal_wildfire_sources,
)


TARGET_VARIABLE = "burntarea-total"
MAPPING_COLUMNS = [
    "id_internal",
    "source_nc",
    "facility_lat",
    "facility_lon",
    "selected_grid_lat",
    "selected_grid_lon",
    "mapping_distance_km",
    "valid_mask_stable",
]
RESULT_COLUMNS = [
    "id_internal",
    "source_nc",
    "time",
    TARGET_VARIABLE,
    "facility_lat",
    "facility_lon",
    "selected_grid_lat",
    "selected_grid_lon",
    "mapping_distance_km",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--facilities-dir",
        type=Path,
        default=data_path('input/climate_facilities'),
    )
    parser.add_argument(
        "--netcdf-dir",
        type=Path,
        default=data_path('input/isimip/wildfire'),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=data_path('work/climate_at_facilities/wildfire'),
    )
    parser.add_argument("--netcdf-pattern", default="*.nc")
    parser.add_argument("--max-netcdf-files", type=int)
    parser.add_argument("--max-facilities", type=int)
    parser.add_argument("--time-chunk-size", type=int, default=120)
    parser.add_argument(
        "--include-nonformal-inputs",
        action="store_true",
        help="Also extract source files that the formal step-09 ensemble never uses.",
    )
    parser.add_argument(
        "--selection-only",
        action="store_true",
        help="Report the selected NetCDF input count and exit before extraction.",
    )
    return parser.parse_args(argv)


def _created_at_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _code_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "UNKNOWN"


def _runtime_evidence() -> dict[str, object]:
    dependency_versions: dict[str, str] = {}
    for package in ("numpy", "pandas", "xarray", "scipy", "cftime", "pyarrow"):
        try:
            dependency_versions[package] = version(package)
        except PackageNotFoundError:
            dependency_versions[package] = "NOT_INSTALLED"
    argv = [str(value) for value in sys.argv]
    return {
        "python_executable": sys.executable,
        "python_version": sys.version,
        "platform": platform.platform(),
        "os_name": os.name,
        "working_directory": str(Path.cwd()),
        "argv": argv,
        "command": subprocess.list2cmdline(argv),
        "dependency_versions": dependency_versions,
    }


def _base_receipt(
    *,
    stage: str,
    status: str,
    input_paths: list[str],
    input_hashes: dict[str, str],
    parameters: dict[str, object],
    row_counts: dict[str, int],
    coverage: dict[str, object],
    output_paths: list[str],
    output_hashes: dict[str, str],
    checks: dict[str, object],
    runtime: dict[str, object],
) -> dict[str, object]:
    return {
        "stage": stage,
        "status": status,
        "code_commit": _code_commit(),
        "input_paths": input_paths,
        "input_hashes": input_hashes,
        "parameters": parameters,
        "row_counts": row_counts,
        "coverage": coverage,
        "output_paths": output_paths,
        "output_hashes": output_hashes,
        "checks": checks,
        "runtime": runtime,
        "created_at_utc": _created_at_utc(),
    }


def _load_facilities(
    paths: list[Path],
    max_facilities: int | None,
) -> pd.DataFrame:
    if not paths:
        raise ValueError("no facility parquet files found")
    frames = []
    required = ["ID_internal", "Geo_lat", "Geo_lon"]
    for path in paths:
        frame = pd.read_parquet(path, columns=required)
        frame = frame.rename(
            columns={
                "ID_internal": "id_internal",
                "Geo_lat": "facility_lat",
                "Geo_lon": "facility_lon",
            }
        )
        frame["facility_stem"] = path.stem
        frames.append(frame)
    facilities = pd.concat(frames, ignore_index=True)
    if max_facilities is not None:
        if max_facilities <= 0:
            raise ValueError("max_facilities must be a positive integer")
        facilities = facilities.iloc[:max_facilities].copy()
    if facilities.empty:
        raise ValueError("facility selection is empty")
    if facilities["id_internal"].isna().any():
        raise ValueError("facility IDs must not be missing")
    if facilities["id_internal"].duplicated().any():
        raise ValueError("facility IDs must be unique across facility files")
    coordinates = facilities[["facility_lat", "facility_lon"]].to_numpy(
        dtype="float64"
    )
    if not np.isfinite(coordinates).all():
        raise ValueError("facility coordinates must be finite")
    if np.any((coordinates[:, 0] < -90.0) | (coordinates[:, 0] > 90.0)):
        raise ValueError("facility latitude must be within [-90, 90] degrees")
    return facilities


def _time_strings(data: xr.DataArray) -> np.ndarray:
    values = np.asarray(data["time"].values)
    strings = np.asarray([str(value) for value in values], dtype=object)
    if pd.Index(strings).has_duplicates:
        raise ValueError("source time coordinate contains duplicate months")
    return strings


def _write_outputs_for_source(
    *,
    source_path: Path,
    data: xr.DataArray,
    facilities: pd.DataFrame,
    mapping: pd.DataFrame,
    output_dir: Path,
    time_chunk_size: int,
) -> tuple[list[Path], int]:
    lat_indices = xr.DataArray(
        mapping["grid_lat_index"].to_numpy(dtype=np.intp),
        dims="pt",
    )
    lon_indices = xr.DataArray(
        mapping["grid_lon_index"].to_numpy(dtype=np.intp),
        dims="pt",
    )
    expected_shape = (data.sizes["time"], len(facilities))
    values = np.empty(expected_shape, dtype=data.dtype)
    for start in range(0, data.sizes["time"], time_chunk_size):
        stop = min(start + time_chunk_size, data.sizes["time"])
        loaded_chunk = data.isel(time=slice(start, stop)).load()
        selected_chunk = loaded_chunk.isel(
            lat=lat_indices,
            lon=lon_indices,
        ).transpose("time", "pt")
        values[start:stop] = np.asarray(selected_chunk.values)
    if values.shape != expected_shape:
        raise ValueError(
            f"vectorized extraction shape {values.shape} != expected {expected_shape}"
        )
    if not np.isfinite(values).all():
        missing_count = int(values.size - np.isfinite(values).sum())
        raise ValueError(
            f"selected valid-grid series contains {missing_count} non-finite values"
        )

    time = _time_strings(data)
    output_paths = []
    total_rows = 0
    for facility_stem, positions in facilities.groupby(
        "facility_stem", sort=False
    ).indices.items():
        positions = np.asarray(positions, dtype=np.intp)
        group_facilities = facilities.iloc[positions]
        group_mapping = mapping.iloc[positions]
        group_values = values[:, positions]
        n_facilities = len(positions)
        n_times = len(time)
        frame = pd.DataFrame(
            {
                "id_internal": np.repeat(
                    group_facilities["id_internal"].to_numpy(), n_times
                ),
                "source_nc": source_path.name,
                "time": np.tile(time, n_facilities),
                TARGET_VARIABLE: group_values.T.reshape(-1),
                "facility_lat": np.repeat(
                    group_facilities["facility_lat"].to_numpy(), n_times
                ),
                "facility_lon": np.repeat(
                    group_facilities["facility_lon"].to_numpy(), n_times
                ),
                "selected_grid_lat": np.repeat(
                    group_mapping["selected_grid_lat"].to_numpy(), n_times
                ),
                "selected_grid_lon": np.repeat(
                    group_mapping["selected_grid_lon"].to_numpy(), n_times
                ),
                "mapping_distance_km": np.repeat(
                    group_mapping["mapping_distance_km"].to_numpy(), n_times
                ),
            },
            columns=RESULT_COLUMNS,
        )
        if len(frame) != n_facilities * n_times:
            raise ValueError("extracted row count does not match facility-month product")
        if frame.duplicated(["id_internal", "source_nc", "time"]).any():
            raise ValueError("duplicate facility-model-month key in extracted output")
        numeric_columns = [
            TARGET_VARIABLE,
            "facility_lat",
            "facility_lon",
            "selected_grid_lat",
            "selected_grid_lon",
            "mapping_distance_km",
        ]
        if not np.isfinite(frame[numeric_columns].to_numpy()).all():
            raise ValueError("non-finite value in extracted output")

        output_path = (
            output_dir / f"result_{facility_stem}_{source_path.stem}.parquet"
        )
        frame.to_parquet(output_path, index=False)
        output_paths.append(output_path)
        total_rows += len(frame)
    return output_paths, total_rows


def process_netcdf_file(
    *,
    source_path: Path,
    facilities: pd.DataFrame,
    facility_paths: list[Path],
    facility_hashes: dict[str, str],
    source_hash: str,
    output_dir: Path,
    parameters: dict[str, object],
    runtime: dict[str, object],
) -> dict[str, Any]:
    receipts_dir = output_dir / "receipts"
    receipts_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = receipts_dir / f"{source_path.stem}.json"
    mapping_path = receipts_dir / f"{source_path.stem}.mapping.parquet"
    input_hashes = {**facility_hashes, str(source_path): source_hash}
    output_paths: list[Path] = []
    output_hashes: dict[str, str] = {}
    time_qa: dict[str, object] | None = None
    try:
        time_decoder = xr.coders.CFDatetimeCoder(use_cftime=True)
        with xr.open_dataset(source_path, decode_times=time_decoder) as dataset:
            if TARGET_VARIABLE not in dataset:
                raise ValueError(f"source variable {TARGET_VARIABLE!r} not found")
            data = dataset[TARGET_VARIABLE]
            if tuple(data.dims) != ("time", "lat", "lon"):
                raise ValueError(
                    "burntarea-total dimensions must be exactly "
                    f"('time', 'lat', 'lon'), got {tuple(data.dims)!r}"
                )
            if data.sizes["time"] < 1:
                raise ValueError("source time dimension is empty")
            try:
                time_qa = validate_monthly_time_axis(data["time"], source_path.name)
            except MonthlyTimeAxisError as error:
                time_qa = error.qa
                raise
            valid_mask, mask_qa = all_time_stable_mask(
                data,
                chunk_size=int(parameters["time_chunk_size"]),
            )
            mapping = nearest_valid_cells(
                np.asarray(data["lat"].values),
                np.asarray(data["lon"].values),
                valid_mask,
                facilities["facility_lat"].to_numpy(),
                facilities["facility_lon"].to_numpy(),
            )
            mapping_table = pd.DataFrame(
                {
                    "id_internal": facilities["id_internal"].to_numpy(),
                    "source_nc": source_path.name,
                    "facility_lat": facilities["facility_lat"].to_numpy(),
                    "facility_lon": facilities["facility_lon"].to_numpy(),
                    "selected_grid_lat": mapping["selected_grid_lat"].to_numpy(),
                    "selected_grid_lon": mapping["selected_grid_lon"].to_numpy(),
                    "mapping_distance_km": mapping[
                        "mapping_distance_km"
                    ].to_numpy(),
                    "valid_mask_stable": True,
                },
                columns=MAPPING_COLUMNS,
            )
            mapping_table.to_parquet(mapping_path, index=False)
            output_paths.append(mapping_path)
            output_hashes[str(mapping_path)] = sha256_file(mapping_path)
            result_paths, extracted_rows = _write_outputs_for_source(
                source_path=source_path,
                data=data,
                facilities=facilities,
                mapping=mapping,
                output_dir=output_dir,
                time_chunk_size=int(parameters["time_chunk_size"]),
            )
            output_paths.extend(result_paths)
            for path in result_paths:
                output_hashes[str(path)] = sha256_file(path)

        max_position = mapping_table["mapping_distance_km"].idxmax()
        max_row = mapping_table.loc[max_position]
        receipt = _base_receipt(
            stage="wildfire_extraction_file",
            status="PASS",
            input_paths=[str(path) for path in [*facility_paths, source_path]],
            input_hashes=input_hashes,
            parameters={
                **parameters,
                "sample_indices": mask_qa["sample_indices"],
                "valid_cell_rule": "decoded_finite_non_fill_for_every_source_month",
                "time_contract_version": TIME_CONTRACT_VERSION,
                "source_start_year": time_qa["start_year"],
                "source_end_year": time_qa["end_year"],
                "source_calendar": time_qa["calendar"],
            },
            row_counts={
                "facility_rows": len(facilities),
                "mapping_rows": len(mapping_table),
                "time_rows": data.sizes["time"],
                "expected_time_rows": int(time_qa["expected_month_count"]),
                "actual_time_rows": int(time_qa["actual_month_count"]),
                "extracted_rows": extracted_rows,
            },
            coverage={
                "expected_facility_model_months": len(facilities)
                * data.sizes["time"],
                "finite_facility_model_months": extracted_rows,
                "duplicate_facility_model_months": 0,
                "expected_start_month": time_qa["expected_start_month"],
                "expected_end_month": time_qa["expected_end_month"],
                "actual_start_month": time_qa["actual_start_month"],
                "actual_end_month": time_qa["actual_end_month"],
                "expected_month_count": int(time_qa["expected_month_count"]),
                "actual_month_count": int(time_qa["actual_month_count"]),
                "calendar": time_qa["calendar"],
                "all_time_stable_grid_cells": int(
                    mask_qa["all_time_stable_cell_count"]
                ),
                "ever_valid_grid_cells": int(mask_qa["ever_valid_cell_count"]),
                "time_varying_grid_cells": int(
                    mask_qa["time_varying_cell_count"]
                ),
                "stable_fraction_of_ever_valid": float(
                    int(mask_qa["all_time_stable_cell_count"])
                    / int(mask_qa["ever_valid_cell_count"])
                ),
                "sample_valid_counts": mask_qa["sample_valid_counts"],
                "sample_pair_diff_counts": mask_qa["sample_pair_diff_counts"],
                "maximum_distance_km": float(max_row["mapping_distance_km"]),
                "maximum_distance_id_internal": str(max_row["id_internal"]),
            },
            output_paths=[str(path) for path in output_paths],
            output_hashes=output_hashes,
            checks={
                "valid_cell_rule": "decoded_finite_non_fill_for_every_source_month",
                "sample_mask_stable": mask_qa["sample_mask_stable"],
                "all_time_mask_scanned": True,
                "complete_finite_unique_keys": True,
                "maximum_distance_cutoff_applied": False,
                "time_contract_version": TIME_CONTRACT_VERSION,
                "time_strictly_increasing": time_qa["strictly_increasing"],
                "time_monthly_continuous": time_qa["monthly_continuous"],
                "filename_time_bounds_match": time_qa["filename_bounds_match"],
            },
            runtime=runtime,
        )
        write_receipt(receipt_path, receipt)
        return {
            "receipt_path": receipt_path,
            "input_hashes": input_hashes,
            "output_paths": [*output_paths, receipt_path],
            "output_hashes": {
                **output_hashes,
                str(receipt_path): sha256_file(receipt_path),
            },
            "extracted_rows": extracted_rows,
            "facility_months": len(facilities) * data.sizes["time"],
            "maximum_distance_km": float(max_row["mapping_distance_km"]),
            "maximum_distance_id_internal": str(max_row["id_internal"]),
            "all_time_stable_grid_cells": int(
                mask_qa["all_time_stable_cell_count"]
            ),
            "time_varying_grid_cells": int(mask_qa["time_varying_cell_count"]),
            "sample_mask_stable": bool(mask_qa["sample_mask_stable"]),
            "time_qa": time_qa,
        }
    except Exception as error:
        failed_receipt = _base_receipt(
            stage="wildfire_extraction_file",
            status="FAILED",
            input_paths=[str(path) for path in [*facility_paths, source_path]],
            input_hashes=input_hashes,
            parameters=parameters,
            row_counts={
                "facility_rows": len(facilities),
                "expected_time_rows": (
                    None if time_qa is None else time_qa["expected_month_count"]
                ),
                "actual_time_rows": (
                    None if time_qa is None else time_qa["actual_month_count"]
                ),
            },
            coverage={
                "source_start_year": (
                    None if time_qa is None else time_qa["start_year"]
                ),
                "source_end_year": None if time_qa is None else time_qa["end_year"],
                "expected_start_month": (
                    None if time_qa is None else time_qa["expected_start_month"]
                ),
                "expected_end_month": (
                    None if time_qa is None else time_qa["expected_end_month"]
                ),
                "actual_start_month": (
                    None if time_qa is None else time_qa["actual_start_month"]
                ),
                "actual_end_month": (
                    None if time_qa is None else time_qa["actual_end_month"]
                ),
                "expected_month_count": (
                    None if time_qa is None else time_qa["expected_month_count"]
                ),
                "actual_month_count": (
                    None if time_qa is None else time_qa["actual_month_count"]
                ),
                "calendar": None if time_qa is None else time_qa["calendar"],
            },
            output_paths=[str(path) for path in output_paths],
            output_hashes=output_hashes,
            checks={
                "error": str(error),
                "time_contract_version": TIME_CONTRACT_VERSION,
                "time_strictly_increasing": (
                    False if time_qa is None else time_qa["strictly_increasing"]
                ),
                "time_monthly_continuous": (
                    False if time_qa is None else time_qa["monthly_continuous"]
                ),
                "filename_time_bounds_match": (
                    False if time_qa is None else time_qa["filename_bounds_match"]
                ),
            },
            runtime=runtime,
        )
        write_receipt(receipt_path, failed_receipt)
        raise


def run(args: argparse.Namespace) -> int:
    runtime = _runtime_evidence()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stage_receipt_path = output_dir / "wildfire_extraction_receipt.json"
    facility_paths = sorted(args.facilities_dir.resolve().glob("*.parquet"))
    all_netcdf_paths = sorted(args.netcdf_dir.resolve().glob(args.netcdf_pattern))
    netcdf_paths = (
        all_netcdf_paths
        if args.include_nonformal_inputs
        else select_formal_wildfire_sources(all_netcdf_paths)
    )
    if args.max_netcdf_files is not None:
        if args.max_netcdf_files <= 0:
            raise ValueError("max_netcdf_files must be a positive integer")
        netcdf_paths = netcdf_paths[: args.max_netcdf_files]
    print(
        f"Selected {len(netcdf_paths)} of {len(all_netcdf_paths)} wildfire NetCDF files.",
        flush=True,
    )
    if args.selection_only:
        return 0
    intended_inputs = [str(path) for path in [*facility_paths, *netcdf_paths]]
    parameters: dict[str, object] = {
        "target_variable": TARGET_VARIABLE,
        "netcdf_pattern": args.netcdf_pattern,
        "max_netcdf_files": args.max_netcdf_files,
        "max_facilities": args.max_facilities,
        "distance_cutoff_km": None,
        "time_chunk_size": args.time_chunk_size,
        "valid_cell_rule": "decoded_finite_non_fill_for_every_source_month",
        "time_contract_version": TIME_CONTRACT_VERSION,
        "formal_scenarios": list(FORMAL_CLIMATE_SCENARIOS),
        "formal_start_year": FORMAL_START_YEAR,
        "formal_end_year": FORMAL_END_YEAR,
        "include_nonformal_inputs": args.include_nonformal_inputs,
        "available_netcdf_files": len(all_netcdf_paths),
        "selected_netcdf_files": len(netcdf_paths),
    }
    input_hashes: dict[str, str] = {}
    output_paths: list[str] = []
    output_hashes: dict[str, str] = {}
    processed_files = 0
    extracted_rows = 0
    expected_rows = 0
    maximum_distance_km = -1.0
    maximum_distance_id_internal: str | None = None
    all_time_stable_grid_cells = 0
    time_varying_grid_cells = 0
    sample_unstable_files = 0
    expected_months = 0
    actual_months = 0
    source_start_years: list[int] = []
    source_end_years: list[int] = []
    calendar_counts: dict[str, int] = {}
    try:
        if not netcdf_paths:
            raise ValueError("no NetCDF files found")
        facility_hashes = {
            str(path): sha256_file(path) for path in facility_paths
        }
        input_hashes.update(facility_hashes)
        source_hashes = {str(path): sha256_file(path) for path in netcdf_paths}
        input_hashes.update(source_hashes)
        facilities = _load_facilities(facility_paths, args.max_facilities)
        for source_path in netcdf_paths:
            source_hash = source_hashes[str(source_path)]
            result = process_netcdf_file(
                source_path=source_path,
                facilities=facilities,
                facility_paths=facility_paths,
                facility_hashes=facility_hashes,
                source_hash=source_hash,
                output_dir=output_dir,
                parameters=parameters,
                runtime=runtime,
            )
            processed_files += 1
            input_hashes.update(result["input_hashes"])
            output_paths.extend(str(path) for path in result["output_paths"])
            output_hashes.update(result["output_hashes"])
            extracted_rows += int(result["extracted_rows"])
            expected_rows += int(result["facility_months"])
            all_time_stable_grid_cells += int(
                result["all_time_stable_grid_cells"]
            )
            time_varying_grid_cells += int(result["time_varying_grid_cells"])
            sample_unstable_files += int(not result["sample_mask_stable"])
            time_qa = result["time_qa"]
            expected_months += int(time_qa["expected_month_count"])
            actual_months += int(time_qa["actual_month_count"])
            source_start_years.append(int(time_qa["start_year"]))
            source_end_years.append(int(time_qa["end_year"]))
            calendar = str(time_qa["calendar"])
            calendar_counts[calendar] = calendar_counts.get(calendar, 0) + 1
            if result["maximum_distance_km"] > maximum_distance_km:
                maximum_distance_km = float(result["maximum_distance_km"])
                maximum_distance_id_internal = str(
                    result["maximum_distance_id_internal"]
                )
            print(
                f"Processed {processed_files}/{len(netcdf_paths)}: {source_path.name}",
                flush=True,
            )

        stage_receipt = _base_receipt(
            stage="wildfire_extraction",
            status="PASS",
            input_paths=intended_inputs,
            input_hashes=input_hashes,
            parameters=parameters,
            row_counts={
                "facility_rows": len(facilities),
                "netcdf_files": len(netcdf_paths),
                "extracted_rows": extracted_rows,
            },
            coverage={
                "processed_netcdf_files": processed_files,
                "expected_netcdf_files": len(netcdf_paths),
                "expected_facility_model_months": expected_rows,
                "finite_facility_model_months": extracted_rows,
                "duplicate_facility_model_months": 0,
                "all_time_stable_grid_cells_across_files": all_time_stable_grid_cells,
                "time_varying_grid_cells_across_files": time_varying_grid_cells,
                "sample_unstable_netcdf_files": sample_unstable_files,
                "maximum_distance_km": maximum_distance_km,
                "maximum_distance_id_internal": maximum_distance_id_internal,
                "source_start_year_min": min(source_start_years),
                "source_end_year_max": max(source_end_years),
                "expected_month_count_across_files": expected_months,
                "actual_month_count_across_files": actual_months,
                "calendar_counts": calendar_counts,
            },
            output_paths=output_paths,
            output_hashes=output_hashes,
            checks={
                "all_files_processed": processed_files == len(netcdf_paths),
                "valid_cell_rule": "decoded_finite_non_fill_for_every_source_month",
                "all_time_masks_scanned": True,
                "complete_finite_unique_keys": extracted_rows == expected_rows,
                "maximum_distance_cutoff_applied": False,
                "time_contract_version": TIME_CONTRACT_VERSION,
                "all_time_axes_strictly_increasing": True,
                "all_time_axes_monthly_continuous": True,
                "all_filename_time_bounds_match": True,
                "expected_actual_month_counts_match": expected_months
                == actual_months,
            },
            runtime=runtime,
        )
        write_receipt(stage_receipt_path, stage_receipt)
        return 0
    except Exception as error:
        failed_time_qa = (
            error.qa if isinstance(error, MonthlyTimeAxisError) else None
        )
        failed_calendar_counts = dict(calendar_counts)
        if failed_time_qa is not None and failed_time_qa["calendar"] is not None:
            failed_calendar = str(failed_time_qa["calendar"])
            failed_calendar_counts[failed_calendar] = (
                failed_calendar_counts.get(failed_calendar, 0) + 1
            )
        failed_receipt = _base_receipt(
            stage="wildfire_extraction",
            status="FAILED",
            input_paths=intended_inputs,
            input_hashes=input_hashes,
            parameters=parameters,
            row_counts={
                "netcdf_files": len(netcdf_paths),
                "extracted_rows": extracted_rows,
            },
            coverage={
                "processed_netcdf_files": processed_files,
                "expected_netcdf_files": len(netcdf_paths),
                "source_start_year_min": (
                    min(source_start_years) if source_start_years else None
                ),
                "source_end_year_max": (
                    max(source_end_years) if source_end_years else None
                ),
                "expected_month_count_across_files": expected_months
                + (
                    0
                    if failed_time_qa is None
                    or failed_time_qa["expected_month_count"] is None
                    else int(failed_time_qa["expected_month_count"])
                ),
                "actual_month_count_across_files": actual_months
                + (
                    0
                    if failed_time_qa is None
                    else int(failed_time_qa["actual_month_count"])
                ),
                "calendar_counts": failed_calendar_counts,
            },
            output_paths=output_paths,
            output_hashes=output_hashes,
            checks={
                "error": str(error),
                "time_contract_version": TIME_CONTRACT_VERSION,
                "all_time_axes_strictly_increasing": False,
                "all_time_axes_monthly_continuous": False,
                "all_filename_time_bounds_match": False,
            },
            runtime=runtime,
        )
        write_receipt(stage_receipt_path, failed_receipt)
        print(f"Wildfire extraction failed: {error}", file=sys.stderr)
        return 1


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        return run(args)
    except Exception as error:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        receipt = _base_receipt(
            stage="wildfire_extraction",
            status="FAILED",
            input_paths=[],
            input_hashes={},
            parameters={},
            row_counts={},
            coverage={},
            output_paths=[],
            output_hashes={},
            checks={
                "error": str(error),
                "time_contract_version": TIME_CONTRACT_VERSION,
            },
            runtime=_runtime_evidence(),
        )
        write_receipt(args.output_dir / "wildfire_extraction_receipt.json", receipt)
        print(f"Wildfire extraction failed: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
