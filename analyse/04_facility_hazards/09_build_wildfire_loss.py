#!/usr/bin/env python
"""Build sparse daily wildfire losses from complete fixed ensembles."""

from __future__ import annotations

from pathlib import Path as _DataPath
import sys as _data_sys
_ANALYSE_ROOT = next(p for p in _DataPath(__file__).resolve().parents if (p / 'run_analyse.py').is_file())
if str(_ANALYSE_ROOT / 'library') not in _data_sys.path:
    _data_sys.path.insert(0, str(_ANALYSE_ROOT / 'library'))
from steel_supply.pipeline_paths import data_path, science_path, bundle_root


import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np
import pandas as pd


ROOT = bundle_root()
sys.path.insert(0, str(ROOT))

from steel_supply.pipeline_receipts import require_pass, sha256_file, write_receipt  # noqa: E402
from steel_supply.hazard_dynamic.wildfire_coverage import (  # noqa: E402
    FORMAL_CLIMATE_SCENARIOS,
    WILDFIRE_FILENAME_PATTERN,
    selected_ensemble_keys,
    validate_model_coverage,
)


CLIMATE_SCENARIOS = FORMAL_CLIMATE_SCENARIOS
FACILITY_PREFIXES = (
    "result_global_energy_monitor_iron_steel_basic_info_",
    "result_spglobal_iron_mine_basic_info_",
)
FILENAME_PATTERN = WILDFIRE_FILENAME_PATTERN


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=data_path('work/climate_at_facilities/wildfire'),
    )
    parser.add_argument(
        "--facilities-dir",
        type=Path,
        default=data_path('input/climate_facilities'),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=data_path('work/hazard_facility_loss/wildfire_loss'),
    )
    parser.add_argument("--extraction-receipt", type=Path)
    parser.add_argument("--receipt-path", type=Path)
    parser.add_argument("--sources", nargs="+")
    parser.add_argument("--scenarios", nargs="+", choices=CLIMATE_SCENARIOS)
    parser.add_argument("--start-year", type=int, default=2015)
    parser.add_argument("--end-year", type=int, default=2100)
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


def _runtime(argv: list[str]) -> dict[str, object]:
    return {
        "argv": argv,
        "command": subprocess.list2cmdline(argv),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
    }


def parse_wildfire_metadata(input_dir: Path) -> pd.DataFrame:
    if not input_dir.is_dir():
        raise ValueError(f"wildfire input directory not found: {input_dir}")
    rows: list[dict[str, object]] = []
    malformed: list[str] = []
    for path in sorted(input_dir.glob("*.parquet")):
        matched_prefix = next(
            (prefix for prefix in FACILITY_PREFIXES if path.stem.startswith(prefix)),
            None,
        )
        if matched_prefix is None:
            malformed.append(path.name)
            continue
        remainder = path.stem[len(matched_prefix) :]
        match = FILENAME_PATTERN.fullmatch(remainder)
        if match is None:
            malformed.append(path.name)
            continue
        values = match.groupdict()
        rows.append(
            {
                "raw_filename": path.name,
                "filepath": str(path.resolve()),
                "dataset_source": matched_prefix.rstrip("_"),
                "source_nc": f"{remainder}.nc",
                **values,
            }
        )
    if malformed:
        raise ValueError(f"malformed wildfire Parquet filenames: {malformed[:3]}")
    if not rows:
        raise ValueError("no wildfire Parquet inputs found")
    return pd.DataFrame(rows)


def _facility_path(facilities_dir: Path, source: str) -> Path:
    source_stem = source.removeprefix("result_")
    return facilities_dir / f"{source_stem}.parquet"


def _facility_ids(path: Path) -> pd.Index:
    if not path.is_file():
        raise ValueError(f"facility source file not found: {path}")
    frame = pd.read_parquet(path, columns=["ID_internal"])
    ids = pd.Index(frame["ID_internal"], name="id_internal")
    if ids.empty or ids.hasnans or not ids.is_unique:
        raise ValueError(f"facility IDs must be nonempty, non-missing, and unique: {path}")
    return ids.sort_values()


def _selection_record(source: str, scenario: str, selected: pd.DataFrame) -> dict[str, object]:
    return {
        "source": source,
        "scenario": scenario,
        "selected_model_count": len(selected),
        "selected_model_keys": selected[
            [
                "model",
                "forcing_member",
                "source_nc",
                "soc_scenario",
                "raw_filename",
                "filepath",
                "start_year",
                "end_year",
            ]
        ].to_dict("records"),
    }


def _base_receipt(
    *,
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
        "stage": "wildfire_fixed_ensemble_loss",
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


def _dimension_digest(ids: pd.Index, months: pd.DatetimeIndex) -> str:
    digest = hashlib.sha256()
    for value in ids:
        digest.update(str(value).encode("utf-8"))
        digest.update(b"\0")
    digest.update(b"|")
    for value in months:
        digest.update(value.isoformat().encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _full_key_digest(
    models: pd.DataFrame,
    base_digests: list[str],
) -> str:
    digest = hashlib.sha256()
    for row, base_digest in zip(models.itertuples(index=False), base_digests, strict=True):
        digest.update(str(row.model).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(row.forcing_member).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(row.source_nc).encode("utf-8"))
        digest.update(b"\0")
        digest.update(base_digest.encode("ascii"))
    return digest.hexdigest()


def _load_one_model(
    metadata_row: Any,
    expected_ids: pd.Index,
    expected_months: pd.DatetimeIndex,
) -> tuple[pd.Series, int, str]:
    path = Path(metadata_row.filepath)
    frame = pd.read_parquet(
        path,
        columns=["id_internal", "source_nc", "time", "burntarea-total"],
    )
    frame["model"] = metadata_row.model
    frame["forcing_member"] = metadata_row.forcing_member
    if not frame["source_nc"].eq(metadata_row.source_nc).all():
        raise ValueError(f"source-file key mismatch in {path}")
    parsed_time = pd.to_datetime(frame["time"], errors="raise")
    frame["month_start"] = parsed_time.dt.to_period("M").dt.to_timestamp()
    expected_model = pd.DataFrame(
        {
            "model": [metadata_row.model],
            "forcing_member": [metadata_row.forcing_member],
            "source_nc": [metadata_row.source_nc],
        }
    )
    validate_model_coverage(frame, expected_model, expected_ids, expected_months)
    values = pd.to_numeric(frame["burntarea-total"], errors="coerce")
    if (values < 0).any():
        raise ValueError(f"negative wildfire value in {path}")
    series = frame.set_index(["id_internal", "month_start"])["burntarea-total"]
    series = series.sort_index()
    actual_ids = pd.Index(series.index.get_level_values("id_internal").unique()).sort_values()
    actual_months = pd.DatetimeIndex(
        series.index.get_level_values("month_start").unique()
    ).sort_values()
    return series, int(values.eq(0).sum()), _dimension_digest(actual_ids, actual_months)


def _ensemble_mean(
    selected: pd.DataFrame,
    expected_ids: pd.Index,
    expected_months: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, dict[str, object]]:
    models = selected[["model", "forcing_member", "source_nc", "filepath"]].copy()
    expected_index = pd.MultiIndex.from_product(
        [expected_ids, expected_months], names=["id_internal", "month_start"]
    ).sort_values()
    total = np.zeros(len(expected_index), dtype=np.float64)
    observed_count = np.zeros(len(expected_index), dtype=np.uint16)
    source_zero_count = 0
    actual_base_digests: list[str] = []
    for row in models.itertuples(index=False):
        series, zero_count, actual_digest = _load_one_model(
            row, expected_ids, expected_months
        )
        if not series.index.equals(expected_index):
            raise ValueError(f"facility-month ordering mismatch in {row.filepath}")
        total += series.to_numpy(dtype=np.float64)
        observed_count += 1
        source_zero_count += zero_count
        actual_base_digests.append(actual_digest)

    expected_model_count = len(models)
    if not np.all(observed_count == expected_model_count):
        raise ValueError(
            "observed model count does not equal selected ensemble count for every "
            "facility-month"
        )
    mean = total / expected_model_count
    if not np.isfinite(mean).all():
        raise ValueError("fixed wildfire ensemble mean contains non-finite values")
    if (mean < 0).any():
        raise ValueError("fixed wildfire ensemble mean contains negative values")

    result = expected_index.to_frame(index=False)
    result["burntarea-total"] = mean
    result["observed_model_count"] = observed_count
    expected_base_digest = _dimension_digest(expected_ids, expected_months)
    expected_key_digest = _full_key_digest(
        models, [expected_base_digest] * expected_model_count
    )
    observed_key_digest = _full_key_digest(models, actual_base_digests)
    if observed_key_digest != expected_key_digest:
        raise ValueError("observed wildfire key digest does not match expected digest")
    expected_key_count = len(expected_index) * expected_model_count
    return result, {
        "expected_facility_model_month_keys": expected_key_count,
        "observed_facility_model_month_keys": expected_key_count,
        "expected_facility_month_groups": len(expected_index),
        "observed_facility_month_groups": len(result),
        "expected_model_count": expected_model_count,
        "observed_model_count_min": int(observed_count.min()),
        "observed_model_count_max": int(observed_count.max()),
        "missing_key_count": 0,
        "duplicate_key_count": 0,
        "nan_value_count": 0,
        "source_value_zero_count": source_zero_count,
        "expected_key_sha256": expected_key_digest,
        "observed_key_sha256": observed_key_digest,
    }


def _write_sparse_annual_outputs(
    monthly: pd.DataFrame,
    source: str,
    scenario: str,
    output_dir: Path,
    start_year: int,
    end_year: int,
) -> tuple[list[Path], int, int, int]:
    zero_mask = monthly["burntarea-total"].eq(0)
    if not (zero_mask | monthly["burntarea-total"].gt(0)).all():
        raise ValueError("omitted monthly values must be exact physical zeros")
    zero_count = int(zero_mask.sum())
    positive_count = int((~zero_mask).sum())
    output_paths: list[Path] = []
    daily_rows = 0
    for year in range(start_year, end_year + 1):
        year_months = monthly.loc[
            monthly["month_start"].dt.year.eq(year)
            & monthly["burntarea-total"].gt(0)
        ].copy()
        if year_months.empty:
            continue
        year_months["days_in_month"] = year_months["month_start"].dt.days_in_month
        expanded = year_months.loc[
            year_months.index.repeat(year_months["days_in_month"])
        ].copy()
        expanded["day_offset"] = expanded.groupby(level=0).cumcount()
        dates = expanded["month_start"] + pd.to_timedelta(expanded["day_offset"], unit="D")
        output = pd.DataFrame(
            {
                "id_internal": expanded["id_internal"].to_numpy(),
                "day_of_year": dates.dt.dayofyear.to_numpy(),
                "production_loss": (
                    expanded["burntarea-total"].to_numpy(dtype=np.float64) / 100.0
                ),
            }
        )
        path = output_dir / (
            f"production_loss_{source.replace('_', '-')}_{year}_{scenario}.parquet"
        )
        output.to_parquet(path, index=False)
        output_paths.append(path)
        daily_rows += len(output)
    return output_paths, zero_count, positive_count, daily_rows


def run(args: argparse.Namespace, argv: list[str]) -> dict[str, object]:
    started = datetime.now(timezone.utc)
    input_dir = args.input_dir.resolve()
    facilities_dir = args.facilities_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = (
        args.receipt_path.resolve()
        if args.receipt_path
        else output_dir / "wildfire_coverage_receipt.json"
    )
    extraction_receipt_path = (
        args.extraction_receipt.resolve()
        if args.extraction_receipt
        else input_dir / "wildfire_extraction_receipt.json"
    )
    runtime = _runtime(argv)
    parameters: dict[str, object] = {
        "input_dir": str(input_dir),
        "facilities_dir": str(facilities_dir),
        "output_dir": str(output_dir),
        "extraction_receipt": str(extraction_receipt_path),
        "start_year": args.start_year,
        "end_year": args.end_year,
        "requested_sources": args.sources,
        "requested_scenarios": args.scenarios or list(CLIMATE_SCENARIOS),
        "soc_preference": ["2015soc-from-histsoc", "2015soc"],
        "sparse_output_rule": "write positive daily rows only after complete monthly coverage PASS",
    }
    input_paths: list[str] = [str(extraction_receipt_path)]
    input_hashes: dict[str, str] = {}
    selection_records: list[dict[str, object]] = []
    ensemble_receipts: list[dict[str, object]] = []
    output_paths: list[Path] = []
    output_hashes: dict[str, str] = {}
    coverage: dict[str, object] = {
        "selected_ensemble_keys": selection_records,
        "ensembles": ensemble_receipts,
    }
    row_counts: dict[str, int] = {}
    checks: dict[str, object] = {
        "selection_written_before_value_loading": False,
        "selected_extraction_output_hashes_match": False,
        "facility_inputs_hashed": False,
        "all_monthly_matrices_complete": False,
        "all_omitted_monthly_values_exact_zeros": False,
        "sparse_outputs_written_after_coverage_pass": False,
    }

    try:
        if args.start_year > args.end_year:
            raise ValueError("start-year must not exceed end-year")
        stale_outputs = sorted(output_dir.glob("production_loss_*.parquet"))
        if stale_outputs:
            raise ValueError(
                f"output directory already contains {len(stale_outputs)} wildfire loss files"
            )
        upstream = require_pass(extraction_receipt_path)
        if not upstream.get("checks", {}).get("complete_finite_unique_keys", False):
            raise ValueError("extraction receipt does not prove complete finite unique keys")
        input_hashes[str(extraction_receipt_path)] = sha256_file(extraction_receipt_path)
        metadata = parse_wildfire_metadata(input_dir)
        sources = args.sources or sorted(metadata["dataset_source"].unique().tolist())
        scenarios = args.scenarios or list(CLIMATE_SCENARIOS)
        selections: list[tuple[str, str, pd.DataFrame]] = []
        for source in sources:
            for scenario in scenarios:
                selected = selected_ensemble_keys(metadata, source, scenario)
                if not selected["start_year"].astype(int).eq(args.start_year).all() or not selected[
                    "end_year"
                ].astype(int).eq(args.end_year).all():
                    raise ValueError(
                        f"selected wildfire source bounds do not match requested interval for "
                        f"{source}/{scenario}"
                    )
                selection_records.append(_selection_record(source, scenario, selected))
                selections.append((source, scenario, selected))

        selected_paths = [
            Path(path)
            for _, _, selected in selections
            for path in selected["filepath"].tolist()
        ]
        upstream_hashes = upstream.get("output_hashes", {})
        for path in selected_paths:
            path_string = str(path.resolve())
            expected_hash = upstream_hashes.get(path_string)
            if expected_hash is None:
                raise ValueError(f"selected input is absent from extraction receipt: {path}")
            actual_hash = sha256_file(path)
            if actual_hash != expected_hash:
                raise ValueError(f"selected input hash mismatch: {path}")
            input_paths.append(path_string)
            input_hashes[path_string] = actual_hash

        for source in sources:
            facility_path = _facility_path(facilities_dir, source)
            if not facility_path.is_file():
                raise ValueError(f"facility source file not found: {facility_path}")
            facility_path_string = str(facility_path.resolve())
            input_paths.append(facility_path_string)
            input_hashes[facility_path_string] = sha256_file(facility_path)

        checks["selected_extraction_output_hashes_match"] = True
        checks["facility_inputs_hashed"] = True
        checks["selection_written_before_value_loading"] = True
        write_receipt(
            receipt_path,
            _base_receipt(
                status="RUNNING",
                input_paths=input_paths,
                input_hashes=input_hashes,
                parameters=parameters,
                row_counts=row_counts,
                coverage=coverage,
                output_paths=[],
                output_hashes={},
                checks=checks,
                runtime=runtime,
            ),
        )

        expected_months = pd.date_range(
            f"{args.start_year}-01-01", f"{args.end_year}-12-01", freq="MS"
        )
        total_input_rows = 0
        total_monthly_groups = 0
        total_daily_rows = 0
        total_zero_months = 0
        total_positive_months = 0
        with tempfile.TemporaryDirectory(
            prefix="wildfire-monthly-coverage-", dir=output_dir
        ) as monthly_directory:
            monthly_root = Path(monthly_directory)
            monthly_results: list[tuple[str, str, Path]] = []
            for source, scenario, selected in selections:
                facility_path = _facility_path(facilities_dir, source)
                expected_ids = _facility_ids(facility_path)
                monthly, ensemble_coverage = _ensemble_mean(
                    selected, expected_ids, expected_months
                )
                zero_mask = monthly["burntarea-total"].eq(0)
                if not (zero_mask | monthly["burntarea-total"].gt(0)).all():
                    raise ValueError("omitted monthly values are not all exact physical zeros")
                zero_count = int(zero_mask.sum())
                positive_count = int((~zero_mask).sum())
                daily_rows = int(
                    monthly.loc[~zero_mask, "month_start"].dt.days_in_month.sum()
                )
                ensemble_coverage.update(
                    {
                        "source": source,
                        "scenario": scenario,
                        "model_set": sorted(selected["model"].unique().tolist()),
                        "forcing_member_set": sorted(
                            selected["forcing_member"].unique().tolist()
                        ),
                        "selected_model_keys": selected[
                            ["model", "forcing_member", "source_nc", "soc_scenario"]
                        ].to_dict("records"),
                        "ensemble_zero_month_count": zero_count,
                        "ensemble_positive_month_count": positive_count,
                        "daily_output_rows": daily_rows,
                    }
                )
                ensemble_receipts.append(ensemble_coverage)
                total_input_rows += int(
                    ensemble_coverage["observed_facility_model_month_keys"]
                )
                total_monthly_groups += len(monthly)
                total_zero_months += zero_count
                total_positive_months += positive_count
                total_daily_rows += daily_rows
                monthly_path = monthly_root / f"{source}__{scenario}.parquet"
                monthly.to_parquet(monthly_path, index=False)
                monthly_results.append((source, scenario, monthly_path))

            checks["all_monthly_matrices_complete"] = True
            checks["all_omitted_monthly_values_exact_zeros"] = True
            row_counts.update(
                {
                    "selected_input_rows": total_input_rows,
                    "monthly_facility_groups": total_monthly_groups,
                    "ensemble_zero_months": total_zero_months,
                    "ensemble_positive_months": total_positive_months,
                    "daily_output_rows": total_daily_rows,
                }
            )

            # This PASS receipt, including zero/positive counts, is the write gate.
            write_receipt(
                receipt_path,
                _base_receipt(
                    status="PASS",
                    input_paths=input_paths,
                    input_hashes=input_hashes,
                    parameters=parameters,
                    row_counts=row_counts,
                    coverage=coverage,
                    output_paths=[],
                    output_hashes={},
                    checks=checks,
                    runtime=runtime,
                ),
            )
            require_pass(receipt_path)

            for source, scenario, monthly_path in monthly_results:
                monthly = pd.read_parquet(monthly_path)
                paths, zero_count, positive_count, daily_rows = _write_sparse_annual_outputs(
                    monthly,
                    source,
                    scenario,
                    output_dir,
                    args.start_year,
                    args.end_year,
                )
                output_paths.extend(paths)
                matching = next(
                    item
                    for item in ensemble_receipts
                    if item["source"] == source and item["scenario"] == scenario
                )
                if (
                    zero_count != matching["ensemble_zero_month_count"]
                    or positive_count != matching["ensemble_positive_month_count"]
                    or daily_rows != matching["daily_output_rows"]
                ):
                    raise ValueError("sparse output counts differ from coverage-gate counts")
                matching["annual_output_file_count"] = len(paths)
        output_hashes = {str(path): sha256_file(path) for path in output_paths}
        row_counts["annual_output_files"] = len(output_paths)
        checks["sparse_outputs_written_after_coverage_pass"] = True
        runtime["elapsed_seconds"] = (
            datetime.now(timezone.utc) - started
        ).total_seconds()
        final = _base_receipt(
            status="PASS",
            input_paths=input_paths,
            input_hashes=input_hashes,
            parameters=parameters,
            row_counts=row_counts,
            coverage=coverage,
            output_paths=[str(path) for path in output_paths],
            output_hashes=output_hashes,
            checks=checks,
            runtime=runtime,
        )
        write_receipt(receipt_path, final)
        return final
    except Exception as error:
        checks["error"] = f"{type(error).__name__}: {error}"
        runtime["elapsed_seconds"] = (
            datetime.now(timezone.utc) - started
        ).total_seconds()
        failed = _base_receipt(
            status="FAILED",
            input_paths=input_paths,
            input_hashes=input_hashes,
            parameters=parameters,
            row_counts=row_counts,
            coverage=coverage,
            output_paths=[str(path) for path in output_paths],
            output_hashes=output_hashes,
            checks=checks,
            runtime=runtime,
        )
        write_receipt(receipt_path, failed)
        raise


def main(argv: list[str] | None = None) -> int:
    parsed_argv = sys.argv[1:] if argv is None else argv
    args = parse_args(parsed_argv)
    command_argv = [sys.executable, str(Path(__file__).resolve()), *parsed_argv]
    try:
        run(args, command_argv)
    except Exception as error:
        print(f"Wildfire loss build failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
