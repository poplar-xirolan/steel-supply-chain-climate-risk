#!/usr/bin/env python
"""Build the explicit SSP1-2.6 cyclone availability policy used by step 10b."""

from __future__ import annotations

from pathlib import Path as _DataPath
import sys as _data_sys
_ANALYSE_ROOT = next(p for p in _DataPath(__file__).resolve().parents if (p / 'run_analyse.py').is_file())
if str(_ANALYSE_ROOT / 'library') not in _data_sys.path:
    _data_sys.path.insert(0, str(_ANALYSE_ROOT / 'library'))
from steel_supply.pipeline_paths import data_path, science_path, bundle_root


import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys


ROOT = bundle_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from steel_supply.pipeline_receipts import sha256_file  # noqa: E402


AVAILABLE_YEARS = tuple(range(2061, 2101))
UNAVAILABLE_YEARS = tuple(range(2023, 2061))
INDUSTRIES = ("iron-factory", "iron-mine", "steel-factory")
POLICY_NAME = "ssp126_cyclone_2023_2060_declared_unavailable"
POLICY_VERSION = "1.0.0"
SOURCE_IDENTITY = "authoritative_ssp126_cyclone_inventory_manifest"

RAW_RE = re.compile(
    r"^(?P<event_set>[^_]+)_(?P<producer>[^_]+)_(?P<model>[^_]+)_"
    r"(?P<scenario>[^_]+)_windlifetimemax_(?P<year>\d{4})\.nc$"
)
EXTRACT_RE = re.compile(
    r"^result_(?P<facility>global_energy_monitor_iron_steel_basic_info|"
    r"spglobal_iron_mine_basic_info)_"
    r"(?P<event_set>[^_]+)_(?P<producer>[^_]+)_(?P<model>[^_]+)_"
    r"(?P<scenario>[^_]+)_windlifetimemax_(?P<year>\d{4})\.parquet$"
)
LOSS_RE = re.compile(
    r"^production_loss_(?P<facility>result-global-energy-monitor-iron-steel-basic-info|"
    r"result-spglobal-iron-mine-basic-info)_(?P<year>\d{4})_"
    r"(?P<scenario>[^_]+)\.parquet$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wind-input-dir",
        type=Path,
        default=data_path('input/isimip/wind'),
    )
    parser.add_argument(
        "--wind-extract-dir",
        type=Path,
        default=data_path('work/climate_at_facilities/wind'),
    )
    parser.add_argument(
        "--cyclone-loss-dir",
        type=Path,
        default=data_path('work/hazard_facility_loss/tropical_cyclone_loss'),
    )
    parser.add_argument(
        "--inventory-path",
        type=Path,
        default=data_path('results/hazard_coverage/ssp126_cyclone_source_inventory.csv'),
    )
    parser.add_argument(
        "--policy-path",
        type=Path,
        default=data_path('results/hazard_coverage/ssp126_cyclone_coverage_policy.json'),
    )
    return parser.parse_args()


def _inventory_rows(directory: Path, pattern: re.Pattern[str], layer: str) -> list[dict[str, object]]:
    if not directory.is_dir():
        raise FileNotFoundError(f"Required directory does not exist: {directory}")
    rows: list[dict[str, object]] = []
    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue
        match = pattern.fullmatch(path.name)
        if match is None or match.group("scenario") != "ssp126":
            continue
        stat = path.stat()
        rows.append(
            {
                "layer": layer,
                "year": int(match.group("year")),
                "path": str(path.resolve()),
                "size_bytes": stat.st_size,
                "last_write_time_utc": datetime.fromtimestamp(
                    stat.st_mtime, tz=timezone.utc
                ).isoformat(),
            }
        )
    return rows


def build_inventory(
    wind_input_dir: Path,
    wind_extract_dir: Path,
    cyclone_loss_dir: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    layers = [
        ("isimip_wind_nc", _inventory_rows(wind_input_dir, RAW_RE, "isimip_wind_nc"), 320),
        (
            "facility_wind_extract",
            _inventory_rows(wind_extract_dir, EXTRACT_RE, "facility_wind_extract"),
            640,
        ),
        (
            "cyclone_facility_loss",
            _inventory_rows(cyclone_loss_dir, LOSS_RE, "cyclone_facility_loss"),
            80,
        ),
    ]
    evidence: list[dict[str, object]] = []
    rows: list[dict[str, object]] = []
    expected_years = set(AVAILABLE_YEARS)
    for layer, layer_rows, expected_count in layers:
        years = {int(row["year"]) for row in layer_rows}
        if len(layer_rows) != expected_count or years != expected_years:
            raise ValueError(
                f"{layer} does not prove exact SSP126 cyclone availability: "
                f"files={len(layer_rows)} expected={expected_count}, "
                f"years={min(years) if years else None}-{max(years) if years else None}"
            )
        rows.extend(layer_rows)
        evidence.append(
            {
                "layer": layer,
                "file_count": len(layer_rows),
                "first_year": min(years),
                "last_year": max(years),
                "distinct_year_count": len(years),
            }
        )
    return sorted(rows, key=lambda row: (str(row["layer"]), int(row["year"]), str(row["path"]))), evidence


def _write_inventory(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["layer", "year", "path", "size_bytes", "last_write_time_utc"],
            quoting=csv.QUOTE_ALL,
        )
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    rows, evidence = build_inventory(
        args.wind_input_dir.resolve(),
        args.wind_extract_dir.resolve(),
        args.cyclone_loss_dir.resolve(),
    )
    inventory_path = args.inventory_path.resolve()
    policy_path = args.policy_path.resolve()
    _write_inventory(inventory_path, rows)
    inventory_hash = sha256_file(inventory_path)

    unavailable_components = [
        {
            "industry": industry,
            "hazard": "tropical-cyclone",
            "year": year,
            "scenario": "ssp126",
            "availability": "unavailable",
            "reason": "authoritative source inventory begins in 2061",
        }
        for industry in INDUSTRIES
        for year in UNAVAILABLE_YEARS
    ]
    payload = {
        "stage": "hazard_coverage_policy",
        "status": "PASS",
        "policy": POLICY_NAME,
        "version": POLICY_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "unavailable_components": unavailable_components,
        "source_provenance": [
            {
                "source_identity": SOURCE_IDENTITY,
                "path": str(inventory_path),
                "sha256": inventory_hash,
                "scenario": "ssp126",
                "hazard": "tropical-cyclone",
                "available_start_year": min(AVAILABLE_YEARS),
                "available_end_year": max(AVAILABLE_YEARS),
                "evidence_layers": evidence,
            }
        ],
        "input_hashes": {str(inventory_path): inventory_hash},
    }
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = policy_path.with_suffix(policy_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(policy_path)
    print(
        json.dumps(
            {
                "status": "PASS",
                "inventory_rows": len(rows),
                "unavailable_components": len(unavailable_components),
                "inventory_path": str(inventory_path),
                "policy_path": str(policy_path),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
