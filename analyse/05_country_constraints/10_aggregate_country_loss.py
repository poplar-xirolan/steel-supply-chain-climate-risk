#!/usr/bin/env python
# coding: utf-8

"""Convert facility hazard fractions to daily facility and country losses."""

from __future__ import annotations

from pathlib import Path as _DataPath
import sys as _data_sys
_ANALYSE_ROOT = next(p for p in _DataPath(__file__).resolve().parents if (p / 'run_analyse.py').is_file())
if str(_ANALYSE_ROOT / 'library') not in _data_sys.path:
    _data_sys.path.insert(0, str(_ANALYSE_ROOT / 'library'))
from steel_supply.pipeline_paths import data_path, science_path, bundle_root


import argparse
import os
from pathlib import Path
import sys

import pandas as pd
from tqdm import tqdm

ROOT = bundle_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from steel_supply.hazard_dynamic.hazard_calendar import days_in_year


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hazard-dir",
        type=Path,
        default=data_path('work/hazard_facility_loss'),
    )
    parser.add_argument(
        "--facilities-dir",
        type=Path,
        default=data_path('results/facilities'),
    )
    parser.add_argument(
        "--facility-output-dir",
        type=Path,
        default=data_path('results/hazard_facility_loss'),
    )
    parser.add_argument(
        "--country-output-dir",
        type=Path,
        default=data_path('results/hazard_country_level_loss'),
    )
    return parser


def hazard_datasets(hazard_dir: Path) -> list[dict[str, str]]:
    return [
        {
            "name": "heatwave",
            "path": str(hazard_dir / "heatwave_gross_loss"),
            "day_of_year_column_name": "day_of_year",
            "id_internal_column_name": "id_internal",
            "production_loss_percent_column_name": "gross_production_loss_fraction",
        },
        {
            "name": "tropical-cyclone",
            "path": str(hazard_dir / "tropical_cyclone_loss"),
            "day_of_year_column_name": "day_of_year",
            "id_internal_column_name": "id_internal",
            "production_loss_percent_column_name": "production_loss_fraction",
        },
        {
            "name": "wildfire",
            "path": str(hazard_dir / "wildfire_loss"),
            "day_of_year_column_name": "day_of_year",
            "id_internal_column_name": "id_internal",
            "production_loss_percent_column_name": "production_loss",
        },
    ]


def production_datasets(facilities_dir: Path) -> list[dict[str, str]]:
    return [
        {
            "name": "iron-mine",
            "data_source": "result-spglobal-iron-mine-basic-info",
            "file_path": str(facilities_dir / "mine_scaled.parquet"),
        },
        {
            "name": "iron-factory",
            "data_source": "result-global-energy-monitor-iron-steel-basic-info",
            "file_path": str(facilities_dir / "iron_factory_scaled.parquet"),
        },
        {
            "name": "steel-factory",
            "data_source": "result-global-energy-monitor-iron-steel-basic-info",
            "file_path": str(facilities_dir / "steel_factory_scaled.parquet"),
        },
    ]


def hazard_filename_interpret(file_path: str | Path) -> dict[str, object]:
    file_name = os.path.basename(file_path)
    remain = file_name.split("production_loss_")[1].split(".parquet")[0]
    metadata_parts = remain.split("_")
    return {
        "data_source": metadata_parts[0],
        "year_int": int(metadata_parts[-2]),
        "scenario": metadata_parts[-1],
    }


def get_parquet_files_in_directory(directory: str | Path) -> list[str]:
    parquet_files = []
    for root, _dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".parquet"):
                parquet_files.append(os.path.join(root, file))
    return parquet_files


def aggregate_facility_losses(
    production_df: pd.DataFrame,
    hazard_df: pd.DataFrame,
    metadata: dict[str, object],
    *,
    hazard_id_column: str,
    hazard_day_column: str,
    hazard_loss_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Expand facilities over a complete year and aggregate their losses.

    Hazard inputs are sparse event records. Missing facility-day keys acquire
    the physical zero loss only after source keys, days, and fractions have
    passed validation. Source event fractions are copied without arithmetic.
    """
    production_columns = {
        "ID_internal",
        "country_a3",
        "fe_eq_weight_kg_scaled",
    }
    missing_production_columns = production_columns.difference(production_df.columns)
    if missing_production_columns:
        raise ValueError(
            f"production data missing required columns: {sorted(missing_production_columns)!r}"
        )
    hazard_columns = {hazard_id_column, hazard_day_column, hazard_loss_column}
    missing_hazard_columns = hazard_columns.difference(hazard_df.columns)
    if missing_hazard_columns:
        raise ValueError(
            f"hazard data missing required columns: {sorted(missing_hazard_columns)!r}"
        )
    if production_df.empty:
        raise ValueError("production data must not be empty")
    if production_df["ID_internal"].isna().any():
        raise ValueError("production data contains a missing facility ID")
    if production_df["ID_internal"].duplicated().any():
        raise ValueError("duplicate production facility IDs are not allowed")

    year = metadata["year_int"]
    year_days = days_in_year(year)
    events = hazard_df[
        [hazard_id_column, hazard_day_column, hazard_loss_column]
    ].rename(
        columns={
            hazard_id_column: "ID_internal",
            hazard_day_column: "day_of_year",
            hazard_loss_column: "production_loss_fraction",
        }
    )
    if events["ID_internal"].isna().any():
        raise ValueError("source event data contains a missing facility ID")
    if events["day_of_year"].isna().any():
        raise ValueError("source event data contains a missing day_of_year")
    if not pd.api.types.is_integer_dtype(events["day_of_year"].dtype):
        raise ValueError("source event day_of_year must use an integer dtype")
    invalid_days = ~events["day_of_year"].between(1, year_days)
    if invalid_days.any():
        values = events.loc[invalid_days, "day_of_year"].unique().tolist()
        raise ValueError(
            f"source event day_of_year outside {year} calendar year: {values!r}"
        )
    if events.duplicated(["ID_internal", "day_of_year"]).any():
        raise ValueError("duplicate source event facility-day keys are not allowed")
    if events["production_loss_fraction"].isna().any():
        raise ValueError("source event data contains missing production_loss_fraction")
    if not pd.api.types.is_numeric_dtype(events["production_loss_fraction"].dtype):
        raise TypeError("source event production_loss_fraction must be numeric")
    finite_loss = events["production_loss_fraction"].map(
        lambda value: bool(pd.notna(value)) and float("-inf") < float(value) < float("inf")
    )
    if not finite_loss.all():
        raise ValueError("source event production_loss_fraction must be finite")

    production = production_df[
        ["ID_internal", "country_a3", "fe_eq_weight_kg_scaled"]
    ].copy()
    production["daily_fe_eq_weight_kg_scaled"] = (
        production["fe_eq_weight_kg_scaled"] / year_days
    )
    production = production.drop(columns=["fe_eq_weight_kg_scaled"])
    calendar = pd.DataFrame({"day_of_year": range(1, year_days + 1)})
    expanded = production.merge(calendar, how="cross")

    merged = expanded.merge(
        events,
        on=["ID_internal", "day_of_year"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    absent_event = merged["_merge"].eq("left_only")
    merged.loc[absent_event, "production_loss_fraction"] = 0.0
    merged = merged.drop(columns=["_merge"])
    merged["production_loss_kg_scaled"] = (
        merged["daily_fe_eq_weight_kg_scaled"]
        * merged["production_loss_fraction"]
    )

    country = merged.drop(columns=["production_loss_fraction"]).groupby(
        ["country_a3", "day_of_year"], as_index=False
    ).agg(
        {
            "daily_fe_eq_weight_kg_scaled": "sum",
            "production_loss_kg_scaled": "sum",
        }
    )
    country["production_loss_fraction"] = (
        country["production_loss_kg_scaled"]
        / country["daily_fe_eq_weight_kg_scaled"]
    )
    return merged, country


def _aggregate_event_span_losses(
    production_df: pd.DataFrame,
    hazard_df: pd.DataFrame,
    *,
    hazard_id_column: str,
    hazard_day_column: str,
    hazard_loss_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Preserve the pre-Task-8 aggregation path for non-cyclone hazards."""
    max_day = hazard_df[hazard_day_column].max()
    min_day = hazard_df[hazard_day_column].min()
    event_span_days = max_day - min_day + 1
    production = production_df.copy()
    production["daily_fe_eq_weight_kg_scaled"] = (
        production["fe_eq_weight_kg_scaled"] / event_span_days
    )
    production = production.drop(columns=["fe_eq_weight_kg_scaled"])
    expanded_parts = []
    for day in range(min_day, max_day + 1):
        day_frame = production.copy()
        day_frame["day_of_year"] = day
        expanded_parts.append(day_frame)
    expanded = pd.concat(expanded_parts, ignore_index=True)
    merged = expanded.merge(
        hazard_df,
        left_on=["ID_internal", "day_of_year"],
        right_on=[hazard_id_column, hazard_day_column],
        how="left",
    )
    merged[hazard_loss_column] = merged[hazard_loss_column].fillna(0)
    if hazard_id_column != "ID_internal":
        merged = merged.drop(columns=[hazard_id_column])
    if hazard_day_column != "day_of_year":
        merged = merged.drop(columns=[hazard_day_column])
    merged = merged.rename(
        columns={hazard_loss_column: "production_loss_fraction"}
    )
    merged["production_loss_kg_scaled"] = (
        merged["daily_fe_eq_weight_kg_scaled"]
        * merged["production_loss_fraction"]
    )
    country = merged.drop(columns=["production_loss_fraction"]).groupby(
        ["country_a3", "day_of_year"], as_index=False
    ).agg(
        {
            "daily_fe_eq_weight_kg_scaled": "sum",
            "production_loss_kg_scaled": "sum",
        }
    )
    country["production_loss_fraction"] = (
        country["production_loss_kg_scaled"]
        / country["daily_fe_eq_weight_kg_scaled"]
    )
    return merged, country


def aggregate_hazard_losses(
    production_df: pd.DataFrame,
    hazard_df: pd.DataFrame,
    metadata: dict[str, object],
    *,
    hazard_name: str,
    hazard_id_column: str,
    hazard_day_column: str,
    hazard_loss_column: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Route Task 8's calendar correction only to tropical-cyclone data."""
    if hazard_name == "tropical-cyclone":
        return aggregate_facility_losses(
            production_df,
            hazard_df,
            metadata,
            hazard_id_column=hazard_id_column,
            hazard_day_column=hazard_day_column,
            hazard_loss_column=hazard_loss_column,
        )
    return _aggregate_event_span_losses(
        production_df,
        hazard_df,
        hazard_id_column=hazard_id_column,
        hazard_day_column=hazard_day_column,
        hazard_loss_column=hazard_loss_column,
    )


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    mine_level_output_dir = str(args.facility_output_dir)
    country_level_output_dir = str(args.country_output_dir)
    # Never delete existing scientific results, including when inputs are missing.
    for destination in (args.facility_output_dir, args.country_output_dir):
        if destination.exists() and any(destination.iterdir()):
            raise FileExistsError(f"Use an empty output directory: {destination}")
    for dataset in production_datasets(args.facilities_dir):
        if not Path(dataset["file_path"]).is_file():
            raise FileNotFoundError(dataset["file_path"])
    for dataset in hazard_datasets(args.hazard_dir):
        if not get_parquet_files_in_directory(dataset["path"]):
            raise FileNotFoundError(f"No hazard inputs: {dataset['path']}")
    os.makedirs(mine_level_output_dir, exist_ok=True)
    os.makedirs(country_level_output_dir, exist_ok=True)

    for production_dataset in production_datasets(args.facilities_dir):
        production_df_outer = pd.read_parquet(
            production_dataset["file_path"],
            columns=["ID_internal", "country_a3", "fe_eq_weight_kg_scaled"],
        )
        for hazard_dataset in hazard_datasets(args.hazard_dir):
            hazard_files = get_parquet_files_in_directory(hazard_dataset["path"])
            for hazard_file in tqdm(
                hazard_files,
                desc=(
                    f"Processing {hazard_dataset['name']} for "
                    f"{production_dataset['name']}"
                ),
            ):
                production_df = production_df_outer.copy()
                metadata = hazard_filename_interpret(hazard_file)
                if metadata["data_source"] != production_dataset["data_source"]:
                    continue

                hazard_df = pd.read_parquet(
                    hazard_file,
                    columns=[
                        hazard_dataset["id_internal_column_name"],
                        hazard_dataset["day_of_year_column_name"],
                        hazard_dataset["production_loss_percent_column_name"],
                    ],
                )
                merged_df, country_level_df = aggregate_hazard_losses(
                    production_df,
                    hazard_df,
                    metadata,
                    hazard_name=hazard_dataset["name"],
                    hazard_id_column=hazard_dataset["id_internal_column_name"],
                    hazard_day_column=hazard_dataset["day_of_year_column_name"],
                    hazard_loss_column=hazard_dataset[
                        "production_loss_percent_column_name"
                    ],
                )

                merged_df.to_parquet(
                    os.path.join(
                        mine_level_output_dir,
                        f"{production_dataset['name']}_{hazard_dataset['name']}_"
                        f"{metadata['year_int']}_{metadata['scenario']}.parquet",
                    ),
                    index=False,
                    compression="zstd",
                )

                country_level_df.to_parquet(
                    os.path.join(
                        country_level_output_dir,
                        f"{production_dataset['name']}_{hazard_dataset['name']}_"
                        f"{metadata['year_int']}_{metadata['scenario']}.parquet",
                    ),
                    index=False,
                    compression="zstd",
                )


if __name__ == "__main__":
    main()
