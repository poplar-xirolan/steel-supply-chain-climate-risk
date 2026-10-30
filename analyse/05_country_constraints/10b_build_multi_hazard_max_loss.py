#!/usr/bin/env python
# coding: utf-8

"""Build facility- and country-level multi-hazard max loss files.

Coupling rule:
- heatwave is treated as an incremental loss relative to the same scenario's
  baseline year;
- tropical-cyclone and wildfire are treated as absolute losses; and
- the coupled loss is the maximum of the three facility-day loss fractions.

The output schema intentionally matches ``10_aggregate_country_loss.py`` so the
ABM engine can consume the coupled files via ``--hazard-type multi-hazard-max``.
"""

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
import logging
import re
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from tqdm import tqdm

ROOT = bundle_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from steel_supply.hazard_dynamic.hazard_calendar import align_2023_baseline, days_in_year

SOURCE_HAZARDS = ("heatwave", "tropical-cyclone", "wildfire")
DEFAULT_OUTPUT_HAZARD = "multi-hazard-max"
COVERAGE_POLICY_INDUSTRIES = ("iron-factory", "iron-mine", "steel-factory")
COVERAGE_POLICY_SCENARIOS = ("picontrol", "ssp126", "ssp370", "ssp585")
COVERAGE_POLICY_NAME = "ssp126_cyclone_2023_2060_declared_unavailable"
COVERAGE_POLICY_VERSION = "1.0.0"
COVERAGE_POLICY_SOURCE_IDENTITY = "authoritative_ssp126_cyclone_inventory_manifest"

FACILITY_COLUMNS = [
    "ID_internal",
    "country_a3",
    "daily_fe_eq_weight_kg_scaled",
    "day_of_year",
]
INDEX_COLUMNS = ["ID_internal", "day_of_year"]
LOSS_COLUMN = "production_loss_fraction"
OUTPUT_COLUMNS = FACILITY_COLUMNS + [LOSS_COLUMN, "production_loss_kg_scaled"]

FILENAME_RE = re.compile(
    r"^(?P<industry>.+?)_"
    r"(?P<hazard>heatwave|tropical-cyclone|wildfire)_"
    r"(?P<year>\d{4})_"
    r"(?P<scenario>[^_]+)\.parquet$"
)

FileKey = tuple[str, str, int, str]  # industry, hazard, year, scenario
ComboKey = tuple[str, int, str]  # industry, year, scenario

FACILITY_INVENTORY_FILES = {
    "iron-mine": "mine_scaled.parquet",
    "iron-factory": "iron_factory_scaled.parquet",
    "steel-factory": "steel_factory_scaled.parquet",
}

SPARSE_WILDFIRE_RE = re.compile(
    r"^production_loss_(?P<source>.+)_(?P<year>\d{4})_(?P<scenario>[^_]+)\.parquet$"
)
REQUIRED_WILDFIRE_CHECKS = (
    "all_monthly_matrices_complete",
    "all_omitted_monthly_values_exact_zeros",
    "sparse_outputs_written_after_coverage_pass",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine heatwave, tropical-cyclone, and wildfire loss fractions by facility-day maximum."
    )
    parser.add_argument(
        "--facility-input-dir",
        type=Path,
        default=data_path('results/hazard_facility_loss'),
        help="Directory containing single-hazard facility-level parquet files.",
    )
    parser.add_argument(
        "--facility-output-dir",
        type=Path,
        default=data_path('results/hazard_facility_loss'),
        help="Directory where coupled facility-level parquet files will be written.",
    )
    parser.add_argument(
        "--country-output-dir",
        type=Path,
        default=data_path('results/hazard_country_level_loss'),
        help="Directory where coupled country-level parquet files will be written.",
    )
    parser.add_argument(
        "--facilities-dir",
        type=Path,
        default=data_path('results/facilities'),
        help="Directory containing the corrected facility inventories.",
    )
    parser.add_argument(
        "--wildfire-coverage-receipt",
        type=Path,
        default=data_path('work/hazard_facility_loss/wildfire_loss/wildfire_coverage_receipt.json'),
        help="Task-7 PASS receipt required only when sparse wildfire rows are zero-filled.",
    )
    parser.add_argument(
        "--coverage-policy",
        type=Path,
        default=data_path('results/hazard_coverage/ssp126_cyclone_coverage_policy.json'),
        help=(
            "Validated declared-availability policy. The only supported exception is "
            "SSP1-2.6 tropical cyclone for 2023-2060."
        ),
    )
    parser.add_argument(
        "--receipt-path",
        type=Path,
        default=None,
        help="Optional machine-readable stage receipt written after all combinations pass.",
    )
    parser.add_argument(
        "--code-commit",
        default="WORKTREE",
        help="Commit identity recorded in --receipt-path.",
    )
    parser.add_argument("--output-hazard", default=DEFAULT_OUTPUT_HAZARD)
    parser.add_argument("--heatwave-baseline-year", type=int, default=2023)
    parser.add_argument("--start-year", type=int, default=2023)
    parser.add_argument("--end-year", type=int, default=2100)
    parser.add_argument("--industries", nargs="*", default=None)
    parser.add_argument("--scenarios", nargs="*", default=None)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned combinations without writing parquet files.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip only structurally complete facility/country output pairs.",
    )
    return parser.parse_args()


def discover_source_files(facility_input_dir: Path) -> dict[FileKey, Path]:
    files: dict[FileKey, Path] = {}
    for file_path in facility_input_dir.glob("*.parquet"):
        match = FILENAME_RE.match(file_path.name)
        if not match:
            continue

        key: FileKey = (
            match.group("industry"),
            match.group("hazard"),
            int(match.group("year")),
            match.group("scenario"),
        )
        files[key] = file_path
    return files


def expected_declared_unavailable_keys() -> tuple[FileKey, ...]:
    """Return the one exact policy exception, never inferred from missing files."""
    return tuple(
        (industry, "tropical-cyclone", year, "ssp126")
        for industry in COVERAGE_POLICY_INDUSTRIES
        for year in range(2023, 2061)
    )


def load_coverage_policy(policy_path: Path) -> dict[str, object]:
    """Load and validate the exact SSP1-2.6 cyclone declaration and provenance."""
    try:
        payload = json.loads(policy_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read hazard coverage policy: {policy_path}") from exc
    if payload.get("stage") != "hazard_coverage_policy" or payload.get("status") != "PASS":
        raise ValueError("Coverage policy must be a hazard_coverage_policy PASS artifact")
    if payload.get("policy") != COVERAGE_POLICY_NAME:
        raise ValueError(f"Unsupported coverage policy: {payload.get('policy')!r}")
    if payload.get("version") != COVERAGE_POLICY_VERSION:
        raise ValueError(f"Unsupported coverage policy version: {payload.get('version')!r}")
    created_at = payload.get("created_at_utc")
    try:
        parsed_created_at = datetime.fromisoformat(str(created_at).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("Coverage policy has an invalid creation time") from exc
    if parsed_created_at.tzinfo is None:
        raise ValueError("Coverage policy creation time must include a timezone")

    components = payload.get("unavailable_components")
    if not isinstance(components, list):
        raise ValueError("Coverage policy unavailable_components must be a list")
    unavailable_keys: list[FileKey] = []
    for item in components:
        if not isinstance(item, dict) or item.get("availability") != "unavailable":
            raise ValueError("Every declared component must be labelled unavailable")
        try:
            key: FileKey = (
                str(item["industry"]),
                str(item["hazard"]),
                int(item["year"]),
                str(item["scenario"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Coverage policy contains a malformed unavailable component") from exc
        unavailable_keys.append(key)
    expected = expected_declared_unavailable_keys()
    if len(unavailable_keys) != 114 or len(set(unavailable_keys)) != 114 or set(unavailable_keys) != set(expected):
        raise ValueError("Coverage policy must enumerate the exact 114 declared-unavailable components")

    provenance = payload.get("source_provenance")
    if not isinstance(provenance, list) or len(provenance) != 1:
        raise ValueError("Coverage policy must contain one authoritative source provenance record")
    source = provenance[0]
    if source.get("source_identity") != COVERAGE_POLICY_SOURCE_IDENTITY:
        raise ValueError("Coverage policy source identity does not match the authoritative inventory")
    expected_source_fields = {
        "scenario": "ssp126",
        "hazard": "tropical-cyclone",
        "available_start_year": 2061,
        "available_end_year": 2100,
    }
    if any(source.get(key) != value for key, value in expected_source_fields.items()):
        raise ValueError("Coverage policy provenance does not establish SSP126 cyclone availability from 2061")
    source_path_value = source.get("path")
    source_hash = source.get("sha256")
    if not isinstance(source_path_value, str) or not isinstance(source_hash, str):
        raise ValueError("Coverage policy source provenance requires path and SHA-256")
    source_path = Path(source_path_value).resolve()
    if not source_path.is_file():
        raise ValueError(f"Coverage policy provenance file does not exist: {source_path}")
    actual_source_hash = _sha256_file(source_path)
    input_hashes = payload.get("input_hashes")
    expected_input_hashes = {str(source_path): source_hash}
    if input_hashes != expected_input_hashes:
        raise ValueError("Coverage policy input hashes do not match source provenance identity")
    if source_hash != actual_source_hash:
        raise ValueError(
            f"Coverage policy source hash mismatch: {actual_source_hash} != {source_hash}"
        )

    validated = dict(payload)
    validated["unavailable_keys"] = frozenset(unavailable_keys)
    validated["policy_path"] = policy_path.resolve()
    validated["policy_hash"] = _sha256_file(policy_path.resolve())
    validated["validated"] = True
    return validated


def _is_declared_unavailable(key: FileKey, coverage_policy: dict[str, object] | None) -> bool:
    return bool(
        coverage_policy is not None
        and coverage_policy.get("validated") is True
        and key in coverage_policy.get("unavailable_keys", frozenset())
    )


def describe_combo_coverage(
    files: dict[FileKey, Path],
    industry: str,
    year: int,
    scenario: str,
    coverage_policy: dict[str, object] | None,
) -> dict[str, list[str]]:
    observed: list[str] = []
    unavailable: list[str] = []
    for hazard in SOURCE_HAZARDS:
        key = (industry, hazard, year, scenario)
        if key in files:
            if _is_declared_unavailable(key, coverage_policy):
                raise ValueError(f"Declared-unavailable component is unexpectedly observed: {key!r}")
            observed.append(hazard)
        elif _is_declared_unavailable(key, coverage_policy):
            unavailable.append(hazard)
        else:
            raise FileNotFoundError(
                f"Unexpected missing hazard component: {industry}, {hazard}, {year}, {scenario}"
            )
    return {"observed_hazards": observed, "unavailable_hazards": unavailable}


def build_combos(
    files: dict[FileKey, Path],
    start_year: int,
    end_year: int,
    industries: set[str] | None,
    scenarios: set[str] | None,
    heatwave_baseline_year: int = 2023,
    coverage_policy: dict[str, object] | None = None,
) -> list[ComboKey]:
    if coverage_policy is not None:
        requested_industries = industries or set(COVERAGE_POLICY_INDUSTRIES)
        requested_scenarios = scenarios or set(COVERAGE_POLICY_SCENARIOS)
        combos = [
            (industry, year, scenario)
            for industry in sorted(requested_industries)
            for scenario in sorted(requested_scenarios)
            for year in range(start_year, end_year + 1)
        ]
        for industry, year, scenario in combos:
            baseline_key = (industry, "heatwave", heatwave_baseline_year, scenario)
            if baseline_key not in files:
                raise FileNotFoundError(
                    f"Unexpected missing heatwave baseline: {industry}, {heatwave_baseline_year}, {scenario}"
                )
            coverage = describe_combo_coverage(files, industry, year, scenario, coverage_policy)
        return combos

    candidates = {
        (industry, year, scenario)
        for industry, _hazard, year, scenario in files
        if start_year <= year <= end_year
    }
    combos = {
        combo
        for combo in candidates
        if all((combo[0], hazard, combo[1], combo[2]) in files for hazard in SOURCE_HAZARDS)
        and (combo[0], "heatwave", heatwave_baseline_year, combo[2]) in files
    }

    if industries is not None:
        combos = {combo for combo in combos if combo[0] in industries}
    if scenarios is not None:
        combos = {combo for combo in combos if combo[2] in scenarios}

    return sorted(combos, key=lambda x: (x[0], x[2], x[1]))


def load_base_frame(
    facility_inventory: pd.DataFrame,
    industry: str,
    year: int,
) -> pd.DataFrame:
    """Expand the corrected industry inventory over the full Gregorian year."""
    required = {"ID_internal", "country_a3", "fe_eq_weight_kg_scaled"}
    missing = required.difference(facility_inventory.columns)
    if missing:
        raise ValueError(f"{industry} facility inventory missing columns: {sorted(missing)!r}")
    inventory = facility_inventory[list(required)].copy()
    if inventory.empty:
        raise ValueError(f"{industry} facility inventory must not be empty")
    if inventory[["ID_internal", "country_a3"]].isna().any().any():
        raise ValueError(f"{industry} facility inventory contains missing identifiers")
    if inventory["ID_internal"].duplicated().any():
        raise ValueError(f"{industry} facility inventory contains duplicate facility IDs")
    weights = pd.to_numeric(inventory["fe_eq_weight_kg_scaled"], errors="coerce")
    if not np.isfinite(weights.to_numpy(dtype="float64")).all() or (weights < 0).any():
        raise ValueError(f"{industry} facility weights must be finite and non-negative")

    year_days = days_in_year(year)
    inventory["daily_fe_eq_weight_kg_scaled"] = weights.astype("float64") / year_days
    inventory = inventory.drop(columns="fe_eq_weight_kg_scaled")
    calendar = pd.DataFrame({"day_of_year": np.arange(1, year_days + 1, dtype="int64")})
    base = inventory.merge(calendar, how="cross")
    return base[FACILITY_COLUMNS].sort_values(INDEX_COLUMNS).reset_index(drop=True)


def _read_raw_loss_series(file_path: Path) -> pd.Series:
    if not file_path.is_file():
        raise FileNotFoundError(f"Required hazard file does not exist: {file_path}")
    frame = pd.read_parquet(file_path)
    aliases = {
        "id_internal": "ID_internal",
        "gross_production_loss_fraction": LOSS_COLUMN,
        "production_loss": LOSS_COLUMN,
        "avg_production_loss": LOSS_COLUMN,
    }
    frame = frame.rename(columns={key: value for key, value in aliases.items() if key in frame.columns})
    required = set(INDEX_COLUMNS + [LOSS_COLUMN])
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Hazard file {file_path} missing columns: {sorted(missing)!r}")
    selected = frame[INDEX_COLUMNS + [LOSS_COLUMN]].copy()
    if selected[INDEX_COLUMNS].isna().any().any():
        raise ValueError(f"Hazard file {file_path} contains missing facility-day keys")
    if selected.duplicated(INDEX_COLUMNS).any():
        raise ValueError(f"Hazard file {file_path} contains duplicate facility-day keys")
    selected["day_of_year"] = pd.to_numeric(selected["day_of_year"], errors="raise").astype("int64")
    values = pd.to_numeric(selected[LOSS_COLUMN], errors="coerce").astype("float64")
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError(f"Hazard file {file_path} contains non-finite loss values")
    selected[LOSS_COLUMN] = values
    return selected.set_index(INDEX_COLUMNS)[LOSS_COLUMN].sort_index()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_sparse_wildfire_receipt(file_path: Path, receipt_path: Path | None) -> None:
    if receipt_path is None:
        raise ValueError("Sparse wildfire zero fill requires a coverage receipt")
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read wildfire coverage receipt: {receipt_path}") from exc
    if receipt.get("stage") != "wildfire_fixed_ensemble_loss" or receipt.get("status") != "PASS":
        raise ValueError("Wildfire coverage receipt must be a Task-7 PASS receipt")
    checks = receipt.get("checks", {})
    if not all(checks.get(name) is True for name in REQUIRED_WILDFIRE_CHECKS):
        raise ValueError("Wildfire coverage receipt does not prove sparse physical zeros")

    match = SPARSE_WILDFIRE_RE.match(file_path.name)
    if not match:
        raise ValueError(f"Cannot identify wildfire source/scenario from {file_path.name}")
    source = match.group("source").replace("-", "_")
    scenario = match.group("scenario")
    ensembles = receipt.get("coverage", {}).get("ensembles", [])
    if not any(
        item.get("source") == source
        and item.get("scenario") == scenario
        and item.get("missing_key_count") == 0
        and item.get("duplicate_key_count") == 0
        and item.get("nan_value_count") == 0
        for item in ensembles
    ):
        raise ValueError(
            f"Wildfire coverage receipt source/scenario does not match {source}/{scenario}"
        )

    resolved = file_path.resolve()
    recorded_hash = None
    for recorded_path, digest in receipt.get("output_hashes", {}).items():
        if Path(recorded_path).resolve() == resolved:
            recorded_hash = digest
            break
    if recorded_hash is None:
        raise ValueError(f"Wildfire coverage receipt has no hash for {resolved}")
    actual_hash = _sha256_file(resolved)
    if recorded_hash != actual_hash:
        raise ValueError(
            f"Wildfire input hash does not match coverage receipt: {actual_hash} != {recorded_hash}"
        )


def read_loss_series(
    file_path: Path,
    target_index: pd.MultiIndex,
    *,
    allow_sparse_zero: bool,
    coverage_receipt: Path | None = None,
) -> pd.Series:
    """Read one hazard series, failing closed unless sparse zeros are proven."""
    if not isinstance(target_index, pd.MultiIndex) or target_index.names != INDEX_COLUMNS:
        raise ValueError(f"target_index must be a MultiIndex named {INDEX_COLUMNS!r}")
    if target_index.has_duplicates:
        raise ValueError("target_index contains duplicate facility-day keys")
    series = _read_raw_loss_series(file_path)
    unexpected = series.index.difference(target_index)
    if len(unexpected):
        raise ValueError(f"Hazard file contains unexpected facility-day keys: {unexpected[:5].tolist()!r}")
    missing = target_index.difference(series.index)
    if len(missing):
        if not allow_sparse_zero:
            raise ValueError(
                f"Hazard file is missing required facility-day keys: {missing[:5].tolist()!r}"
            )
        _validate_sparse_wildfire_receipt(file_path, coverage_receipt)
    return series.reindex(target_index, fill_value=0.0).astype("float64")


def load_hazard_component(
    files: dict[FileKey, Path],
    baseline_cache: dict[tuple[str, str], pd.Series],
    target_index: pd.MultiIndex,
    industry: str,
    hazard: str,
    year: int,
    scenario: str,
    heatwave_baseline_year: int,
    wildfire_coverage_receipt: Path | None,
    coverage_policy: dict[str, object] | None = None,
) -> pd.Series:
    component_key = (industry, hazard, year, scenario)
    file_path = files.get(component_key)
    if file_path is None:
        if _is_declared_unavailable(component_key, coverage_policy):
            return pd.Series(0.0, index=target_index, dtype="float64")
        raise FileNotFoundError(
            f"Missing required {hazard} file for {industry}, {year}, {scenario}"
        )
    if _is_declared_unavailable(component_key, coverage_policy):
        raise ValueError(
            f"Declared-unavailable component is unexpectedly observed: {component_key!r}"
        )

    component = read_loss_series(
        file_path,
        target_index,
        allow_sparse_zero=hazard == "wildfire",
        coverage_receipt=wildfire_coverage_receipt if hazard == "wildfire" else None,
    )
    if hazard != "heatwave":
        return component.clip(lower=0.0, upper=1.0)
    if heatwave_baseline_year != 2023:
        raise ValueError(
            f"Heatwave baseline year must be 2023; got {heatwave_baseline_year}"
        )

    baseline_key = (industry, scenario)
    if baseline_key not in baseline_cache:
        baseline_path = files.get((industry, "heatwave", heatwave_baseline_year, scenario))
        if baseline_path is None:
            raise FileNotFoundError(
                f"Missing required heatwave baseline file for {industry}, "
                f"{heatwave_baseline_year}, {scenario}"
            )
        baseline_cache[baseline_key] = _read_raw_loss_series(baseline_path)

    baseline = baseline_cache[baseline_key]
    baseline_aligned = align_2023_baseline(baseline, target_index, target_year=year)

    # Match the original ABM heatwave treatment: future years only inherit the
    # positive increment over the baseline; lower heat stress never creates a benefit.
    return (component - baseline_aligned).clip(lower=0.0, upper=1.0)


def build_one_combo(
    files: dict[FileKey, Path],
    baseline_cache: dict[tuple[str, str], pd.Series],
    facility_inventory: pd.DataFrame,
    industry: str,
    year: int,
    scenario: str,
    heatwave_baseline_year: int,
    wildfire_coverage_receipt: Path | None,
    coverage_policy: dict[str, object] | None = None,
) -> pd.DataFrame:
    facility_df = load_base_frame(facility_inventory, industry, year)
    target_index = pd.MultiIndex.from_frame(facility_df[INDEX_COLUMNS])

    components = [
        load_hazard_component(
            files=files,
            baseline_cache=baseline_cache,
            target_index=target_index,
            industry=industry,
            hazard=hazard,
            year=year,
            scenario=scenario,
            heatwave_baseline_year=heatwave_baseline_year,
            wildfire_coverage_receipt=wildfire_coverage_receipt,
            coverage_policy=coverage_policy,
        )
        for hazard in SOURCE_HAZARDS
    ]

    combined_loss = pd.concat(components, axis=1).max(axis=1).to_numpy()
    facility_df[LOSS_COLUMN] = combined_loss
    facility_df[LOSS_COLUMN] = facility_df[LOSS_COLUMN].clip(lower=0.0, upper=1.0)
    facility_df["production_loss_kg_scaled"] = (
        facility_df["daily_fe_eq_weight_kg_scaled"] * facility_df[LOSS_COLUMN]
    )
    return facility_df[OUTPUT_COLUMNS]


def aggregate_country_loss(facility_df: pd.DataFrame) -> pd.DataFrame:
    country_df = facility_df.groupby(["country_a3", "day_of_year"], as_index=False).agg(
        {
            "daily_fe_eq_weight_kg_scaled": "sum",
            "production_loss_kg_scaled": "sum",
        }
    )
    country_df[LOSS_COLUMN] = (
        country_df["production_loss_kg_scaled"] / country_df["daily_fe_eq_weight_kg_scaled"]
    ).fillna(0.0)
    return country_df[
        [
            "country_a3",
            "day_of_year",
            "daily_fe_eq_weight_kg_scaled",
            "production_loss_kg_scaled",
            LOSS_COLUMN,
        ]
    ]


def write_outputs(
    facility_df: pd.DataFrame,
    industry: str,
    output_hazard: str,
    year: int,
    scenario: str,
    facility_output_dir: Path,
    country_output_dir: Path,
) -> None:
    facility_output_dir.mkdir(parents=True, exist_ok=True)
    country_output_dir.mkdir(parents=True, exist_ok=True)

    facility_path = facility_output_dir / f"{industry}_{output_hazard}_{year}_{scenario}.parquet"
    country_path = country_output_dir / f"{industry}_{output_hazard}_{year}_{scenario}.parquet"

    facility_df.to_parquet(facility_path, index=False, compression="zstd")
    country_df = aggregate_country_loss(facility_df)
    country_df.to_parquet(country_path, index=False, compression="zstd")


def output_paths(
    industry: str,
    output_hazard: str,
    year: int,
    scenario: str,
    facility_output_dir: Path,
    country_output_dir: Path,
) -> tuple[Path, Path]:
    return (
        facility_output_dir / f"{industry}_{output_hazard}_{year}_{scenario}.parquet",
        country_output_dir / f"{industry}_{output_hazard}_{year}_{scenario}.parquet",
    )


def build_stage_receipt_payload(
    *,
    files: dict[FileKey, Path],
    combos: list[ComboKey],
    coverage_policy: dict[str, object],
    coverage_policy_path: Path,
    output_pairs: list[tuple[Path, Path]],
    validated_output_combos: list[ComboKey] | None,
    code_commit: str,
    created_at_utc: str,
) -> dict[str, object]:
    if validated_output_combos is None:
        raise ValueError("Stage receipt requires validated output pairs")
    expected_combos = [
        (industry, year, scenario)
        for industry in sorted(COVERAGE_POLICY_INDUSTRIES)
        for scenario in sorted(COVERAGE_POLICY_SCENARIOS)
        for year in range(2023, 2101)
    ]
    if len(combos) != 936 or len(set(combos)) != 936 or set(combos) != set(expected_combos):
        raise ValueError("Stage receipt requires the exact 936 requested combinations")
    if (
        len(validated_output_combos) != 936
        or len(set(validated_output_combos)) != 936
        or set(validated_output_combos) != set(expected_combos)
    ):
        raise ValueError("Stage receipt requires the exact 936 validated output pairs")
    if (
        coverage_policy.get("validated") is not True
        or coverage_policy.get("unavailable_keys")
        != frozenset(expected_declared_unavailable_keys())
    ):
        raise ValueError("Stage receipt requires the exact validated 114-component policy")
    if len(output_pairs) != 936 or len(set(output_pairs)) != 936:
        raise ValueError("Stage receipt requires exactly 936 unique output pairs")
    for combo, pair in zip(combos, output_pairs, strict=True):
        if len(pair) != 2:
            raise ValueError(f"Invalid output pair for {combo!r}")
        expected_name = f"{combo[0]}_{DEFAULT_OUTPUT_HAZARD}_{combo[1]}_{combo[2]}.parquet"
        facility_path, country_path = pair
        if facility_path.name != expected_name or country_path.name != expected_name:
            raise ValueError(f"output pair identity does not match combination {combo!r}")
        if not facility_path.is_file() or not country_path.is_file():
            raise ValueError(f"Required output pair does not exist for {combo!r}")

    combination_coverage = []
    unavailable_count = 0
    three_hazard_count = 0
    declared_two_hazard_count = 0
    for industry, year, scenario in combos:
        coverage = describe_combo_coverage(files, industry, year, scenario, coverage_policy)
        unavailable_count += len(coverage["unavailable_hazards"])
        if coverage["observed_hazards"] == list(SOURCE_HAZARDS) and not coverage["unavailable_hazards"]:
            three_hazard_count += 1
        elif coverage == {
            "observed_hazards": ["heatwave", "wildfire"],
            "unavailable_hazards": ["tropical-cyclone"],
        }:
            declared_two_hazard_count += 1
        else:
            raise ValueError(
                f"Unexpected observed/unavailable hazard identity for "
                f"{industry}, {year}, {scenario}: {coverage!r}"
            )
        combination_coverage.append(
            {
                "industry": industry,
                "year": year,
                "scenario": scenario,
                **coverage,
            }
        )
    output_paths = [path.resolve() for pair in output_pairs for path in pair]
    output_hashes = {str(path): _sha256_file(path) for path in output_paths}
    if (
        unavailable_count != 114
        or three_hazard_count != 822
        or declared_two_hazard_count != 114
        or len(output_paths) != 1872
        or len(output_hashes) != 1872
    ):
        raise ValueError("Stage receipt coverage/output validation is incomplete")
    resolved_policy_path = coverage_policy_path.resolve()
    return {
        "stage": "multi_hazard_max_loss",
        "status": "PASS",
        "code_commit": code_commit,
        "input_paths": [str(resolved_policy_path)],
        "input_hashes": {str(resolved_policy_path): _sha256_file(resolved_policy_path)},
        "parameters": {
            "formula": (
                "max(incremental heatwave, absolute tropical-cyclone when observed, "
                "absolute wildfire) clipped [0,1]"
            ),
            "coverage_policy": coverage_policy["policy"],
            "coverage_policy_version": coverage_policy["version"],
        },
        "row_counts": {
            "combination_count": len(combos),
            "validated_output_pairs": len(validated_output_combos),
            "facility_output_files": sum(pair[0].is_file() for pair in output_pairs),
            "country_output_files": sum(pair[1].is_file() for pair in output_pairs),
        },
        "coverage": {
            "complete_under_declared_availability_policy": True,
            "complete_three_hazard_source_coverage": unavailable_count == 0,
            "declared_unavailable_component_count": unavailable_count,
            "three_hazard_combination_count": three_hazard_count,
            "declared_two_hazard_combination_count": declared_two_hazard_count,
            "unexpected_missing_component_count": 0,
            "combinations": combination_coverage,
        },
        "output_paths": [str(path) for path in output_paths],
        "output_hashes": output_hashes,
        "checks": {
            "coverage_policy_validated": coverage_policy.get("validated") is True,
            "observed_and_unavailable_hazards_reported_separately": True,
            "missing_components_not_labelled_observed_zero": True,
            "all_output_pairs_semantically_validated": True,
        },
        "created_at_utc": created_at_utc,
    }


def validate_existing_output_pair(
    facility_path: Path,
    country_path: Path,
    facility_inventory: pd.DataFrame,
    year: int,
) -> bool:
    """Return true only for a semantically complete resumable output pair.

    The corrected inventory and Gregorian calendar define the expected keys,
    attributes, and daily weights.  Facility losses must be finite and bounded,
    loss mass must equal daily weight times loss fraction bit-for-bit, and the
    country output must equal a fresh aggregation of the validated facility
    output.  Any existing but invalid pair fails closed instead of being
    silently rebuilt or skipped.
    """
    facility_exists = facility_path.is_file()
    country_exists = country_path.is_file()
    if not facility_exists and not country_exists:
        return False
    if facility_exists != country_exists:
        raise ValueError(
            f"partial existing output pair: {facility_path}, {country_path}"
        )
    facility_parquet = pq.ParquetFile(facility_path)
    country_parquet = pq.ParquetFile(country_path)
    facility_metadata = facility_parquet.metadata
    country_metadata = country_parquet.metadata
    facility_schema = facility_parquet.schema_arrow.names
    country_schema = country_parquet.schema_arrow.names
    if facility_schema != OUTPUT_COLUMNS:
        raise ValueError(
            f"existing facility schema does not match: {facility_schema!r}"
        )
    expected_country_columns = [
        "country_a3",
        "day_of_year",
        "daily_fe_eq_weight_kg_scaled",
        "production_loss_kg_scaled",
        LOSS_COLUMN,
    ]
    if country_schema != expected_country_columns:
        raise ValueError(
            f"existing country schema does not match: {country_schema!r}"
        )
    expected_facility_rows = len(facility_inventory) * days_in_year(year)
    if facility_metadata.num_rows != expected_facility_rows:
        raise ValueError(
            f"existing facility row count {facility_metadata.num_rows} does not match "
            f"{expected_facility_rows}"
        )
    expected_country_rows = (
        facility_inventory["country_a3"].nunique(dropna=False) * days_in_year(year)
    )
    if country_metadata.num_rows != expected_country_rows:
        raise ValueError(
            f"existing country row count {country_metadata.num_rows} does not match "
            f"{expected_country_rows}"
        )

    expected_base = load_base_frame(facility_inventory, "resume-validation", year)
    facility = pd.read_parquet(facility_path, columns=OUTPUT_COLUMNS)
    if facility.duplicated(INDEX_COLUMNS).any():
        raise ValueError("existing facility-day keys contain duplicates")
    facility = facility.sort_values(INDEX_COLUMNS).reset_index(drop=True)
    if not facility[INDEX_COLUMNS].equals(expected_base[INDEX_COLUMNS]):
        raise ValueError("existing facility-day keys do not match inventory x Gregorian calendar")
    if not facility["country_a3"].equals(expected_base["country_a3"]):
        raise ValueError("existing facility countries do not match corrected inventory")
    if not np.array_equal(
        facility["daily_fe_eq_weight_kg_scaled"].to_numpy(),
        expected_base["daily_fe_eq_weight_kg_scaled"].to_numpy(),
    ):
        raise ValueError("existing facility daily weights do not match corrected inventory")

    facility_numeric = facility[
        ["daily_fe_eq_weight_kg_scaled", "production_loss_kg_scaled", LOSS_COLUMN]
    ].to_numpy(dtype="float64")
    if not np.isfinite(facility_numeric).all():
        raise ValueError("existing facility numeric values must be finite")
    if not facility[LOSS_COLUMN].between(0.0, 1.0).all():
        raise ValueError("existing facility loss fractions must be bounded in [0, 1]")
    expected_loss_mass = (
        facility["daily_fe_eq_weight_kg_scaled"].to_numpy()
        * facility[LOSS_COLUMN].to_numpy()
    )
    if not np.array_equal(
        facility["production_loss_kg_scaled"].to_numpy(),
        expected_loss_mass,
    ):
        raise ValueError("existing facility production loss mass does not match weight x fraction")

    country = pd.read_parquet(country_path, columns=expected_country_columns)
    country_keys = ["country_a3", "day_of_year"]
    if country.duplicated(country_keys).any():
        raise ValueError("existing country-day keys contain duplicates")
    country = country.sort_values(country_keys).reset_index(drop=True)
    expected_country = aggregate_country_loss(facility).sort_values(
        country_keys
    ).reset_index(drop=True)
    if not country[country_keys].equals(expected_country[country_keys]):
        raise ValueError("existing country-day keys do not match facility aggregation")
    country_numeric_columns = [
        "daily_fe_eq_weight_kg_scaled",
        "production_loss_kg_scaled",
        LOSS_COLUMN,
    ]
    country_numeric = country[country_numeric_columns].to_numpy(dtype="float64")
    if not np.isfinite(country_numeric).all():
        raise ValueError("existing country numeric values must be finite")
    if not country[LOSS_COLUMN].between(0.0, 1.0).all():
        raise ValueError("existing country loss fractions must be bounded in [0, 1]")
    if not np.array_equal(
        country[country_numeric_columns].to_numpy(),
        expected_country[country_numeric_columns].to_numpy(),
    ):
        raise ValueError("existing country aggregation does not match validated facility output")
    return True


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = parse_args()

    if args.start_year > args.end_year:
        raise ValueError("--start-year must not exceed --end-year")
    if "_" in args.output_hazard:
        raise ValueError("--output-hazard must not contain underscores because ABM file parsing uses underscores.")

    files = discover_source_files(args.facility_input_dir)
    coverage_policy = (
        load_coverage_policy(args.coverage_policy)
        if args.coverage_policy is not None
        else None
    )
    industries = set(args.industries) if args.industries else None
    scenarios = set(args.scenarios) if args.scenarios else None
    combos = build_combos(
        files=files,
        start_year=args.start_year,
        end_year=args.end_year,
        industries=industries,
        scenarios=scenarios,
        heatwave_baseline_year=args.heatwave_baseline_year,
        coverage_policy=coverage_policy,
    )

    logging.info("Discovered %s source files.", len(files))
    logging.info("Prepared %s multi-hazard combinations.", len(combos))

    if args.dry_run:
        for industry, year, scenario in combos:
            print(f"{industry}_{args.output_hazard}_{year}_{scenario}.parquet")
        return

    baseline_cache: dict[tuple[str, str], pd.Series] = {}
    facility_inventories: dict[str, pd.DataFrame] = {}
    skipped_existing = 0
    output_pairs: list[tuple[Path, Path]] = []
    validated_output_combos: list[ComboKey] = []
    for industry, year, scenario in tqdm(combos, desc="Building multi-hazard max loss"):
        facility_path, country_path = output_paths(
            industry,
            args.output_hazard,
            year,
            scenario,
            args.facility_output_dir,
            args.country_output_dir,
        )
        output_pairs.append((facility_path, country_path))
        if industry not in FACILITY_INVENTORY_FILES:
            raise ValueError(f"No corrected facility inventory is declared for industry {industry!r}")
        if industry not in facility_inventories:
            inventory_path = args.facilities_dir / FACILITY_INVENTORY_FILES[industry]
            if not inventory_path.is_file():
                raise FileNotFoundError(f"Missing corrected facility inventory: {inventory_path}")
            facility_inventories[industry] = pd.read_parquet(inventory_path)
        if args.skip_existing and validate_existing_output_pair(
            facility_path,
            country_path,
            facility_inventories[industry],
            year,
        ):
            skipped_existing += 1
            validated_output_combos.append((industry, year, scenario))
            continue
        facility_df = build_one_combo(
            files=files,
            baseline_cache=baseline_cache,
            facility_inventory=facility_inventories[industry],
            industry=industry,
            year=year,
            scenario=scenario,
            heatwave_baseline_year=args.heatwave_baseline_year,
            wildfire_coverage_receipt=args.wildfire_coverage_receipt,
            coverage_policy=coverage_policy,
        )
        write_outputs(
            facility_df=facility_df,
            industry=industry,
            output_hazard=args.output_hazard,
            year=year,
            scenario=scenario,
            facility_output_dir=args.facility_output_dir,
            country_output_dir=args.country_output_dir,
        )
        if args.receipt_path is not None:
            if not validate_existing_output_pair(
                facility_path,
                country_path,
                facility_inventories[industry],
                year,
            ):
                raise ValueError(
                    f"New output pair was not semantically validated for "
                    f"{industry}, {year}, {scenario}"
                )
            validated_output_combos.append((industry, year, scenario))

    if args.receipt_path is not None:
        if coverage_policy is None or args.coverage_policy is None:
            raise ValueError("--receipt-path requires a validated --coverage-policy")
        receipt = build_stage_receipt_payload(
            files=files,
            combos=combos,
            coverage_policy=coverage_policy,
            coverage_policy_path=args.coverage_policy,
            output_pairs=output_pairs,
            validated_output_combos=validated_output_combos,
            code_commit=args.code_commit,
            created_at_utc=datetime.now(timezone.utc).isoformat(),
        )
        args.receipt_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = args.receipt_path.with_suffix(args.receipt_path.suffix + ".tmp")
        temporary_path.write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary_path.replace(args.receipt_path)

    logging.info("Done. Skipped %s complete existing output pairs.", skipped_existing)


if __name__ == "__main__":
    main()
