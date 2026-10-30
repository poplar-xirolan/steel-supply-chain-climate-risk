"""Shared ore-grade parsing, resolution, and trade-adjustment logic."""

from __future__ import annotations
from steel_supply.pipeline_paths import science_path, bundle_root, data_path

import re

import numpy as np
import pandas as pd


COUNTRY_COLUMN = "Country_code_iso3166-2_alpha_3"
GRADE_COLUMNS = ["country", "year", "grade"]
GRADE_QA_COLUMNS = ["country", "year", "raw_value", "reason"]
RECEIPT_COLUMNS = [
    "trade_row_key",
    "importer_iso_a3",
    "exporter_iso_a3",
    "trade_year",
    "grade",
    "fallback_level",
    "matched_country",
    "matched_year",
    "raw_mass_kg",
    "corrected_fe_equivalent_mass_kg",
]
IRON_ORE_COMMODITY_CODE = 281


def parse_grade_table(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert the wide USGS workbook table to country/year/grade rows."""
    usable, _ = parse_grade_table_with_qa(frame)
    return usable


def parse_grade_table_with_qa(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return usable grades and every rejected workbook grade cell."""
    if COUNTRY_COLUMN not in frame.columns:
        raise ValueError(f"Grade table is missing required column: {COUNTRY_COLUMN}")

    grade_columns = []
    for column in frame.columns:
        match = re.match(r"^\s*(\d{4})\s+Fe eq", str(column))
        if match:
            grade_columns.append((column, int(match.group(1))))
    if not grade_columns:
        raise ValueError("Grade table has no '<year> Fe eq' columns")

    usable_pieces = []
    rejected_pieces = []
    for column, year in grade_columns:
        piece = frame[[COUNTRY_COLUMN, column]].copy()
        piece.columns = ["country", "raw_value"]
        piece["year"] = year
        country = piece["country"].astype("string").str.strip()
        numeric = pd.to_numeric(piece["raw_value"], errors="coerce")
        reason = pd.Series(pd.NA, index=piece.index, dtype="string")
        reason.loc[country.isna() | country.eq("")] = "missing_country"
        reason.loc[reason.isna() & piece["raw_value"].isna()] = "missing"
        reason.loc[reason.isna() & numeric.isna()] = "nonnumeric"
        reason.loc[reason.isna() & ~np.isfinite(numeric)] = "nonfinite"
        reason.loc[reason.isna() & numeric.le(0)] = "nonpositive"
        reason.loc[reason.isna() & numeric.gt(1)] = "greater_than_one"

        valid = reason.isna()
        usable_pieces.append(
            pd.DataFrame(
                {
                    "country": country.loc[valid],
                    "year": year,
                    "grade": numeric.loc[valid],
                }
            )
        )
        rejected_pieces.append(
            pd.DataFrame(
                {
                    "country": country.loc[~valid],
                    "year": year,
                    "raw_value": piece.loc[~valid, "raw_value"],
                    "reason": reason.loc[~valid],
                }
            )
        )

    parsed = pd.concat(usable_pieces, ignore_index=True)
    parsed["country"] = parsed["country"].astype(object)
    parsed["year"] = parsed["year"].astype(int)
    parsed["grade"] = parsed["grade"].astype(float)
    parsed = parsed[GRADE_COLUMNS].sort_values(
        ["country", "year"], kind="stable", ignore_index=True
    )
    rejected = pd.concat(rejected_pieces, ignore_index=True)[GRADE_QA_COLUMNS]
    rejected["country"] = rejected["country"].astype(object)
    rejected["year"] = rejected["year"].astype(int)
    rejected["reason"] = rejected["reason"].astype(object)
    rejected = rejected.sort_values(
        ["country", "year"], kind="stable", na_position="last", ignore_index=True
    )
    return _validated_grade_table(parsed), rejected


def _validated_grade_table(grade_table: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in GRADE_COLUMNS if column not in grade_table.columns]
    if missing:
        raise ValueError(f"Normalized grade table is missing columns: {missing}")

    validated = grade_table[GRADE_COLUMNS].copy()
    validated["year"] = pd.to_numeric(validated["year"], errors="raise").astype(int)
    validated["grade"] = pd.to_numeric(
        validated["grade"], errors="raise"
    ).astype(float)
    if validated[GRADE_COLUMNS].isna().any().any():
        raise ValueError("Normalized grade table contains missing values")
    physical = (
        np.isfinite(validated["grade"])
        & validated["grade"].gt(0)
        & validated["grade"].le(1)
    )
    if not physical.all():
        invalid = validated.loc[~physical].to_dict(orient="records")
        raise ValueError(
            "Normalized grade table contains values outside 0 < grade <= 1: "
            f"{invalid}"
        )
    duplicate = validated.duplicated(["country", "year"], keep=False)
    if duplicate.any():
        pairs = validated.loc[duplicate, ["country", "year"]].drop_duplicates()
        raise ValueError(
            "Normalized grade table has duplicate country/year rows: "
            f"{pairs.to_dict(orient='records')}"
        )
    return validated.reset_index(drop=True)


def _nearest_row(rows: pd.DataFrame, year: int) -> pd.Series | None:
    if rows.empty:
        return None
    ranked = rows.assign(
        _distance=(rows["year"] - year).abs()
    ).sort_values(["_distance", "year"], kind="stable")
    return ranked.iloc[0]


def _resolve_grade_from_validated(
    country: str, year: int, grade_table: pd.DataFrame
) -> tuple[float, str, str, int]:
    exporter_rows = grade_table.loc[grade_table["country"].eq(country)]
    exporter_exact = exporter_rows.loc[exporter_rows["year"].eq(year)]
    if not exporter_exact.empty:
        row = exporter_exact.iloc[0]
        return float(row["grade"]), "exporter_exact_year", str(country), int(year)

    exporter_nearest = _nearest_row(exporter_rows, year)
    if exporter_nearest is not None:
        return (
            float(exporter_nearest["grade"]),
            "exporter_nearest_year",
            str(country),
            int(exporter_nearest["year"]),
        )

    world_rows = grade_table.loc[grade_table["country"].eq("Total")]
    world_exact = world_rows.loc[world_rows["year"].eq(year)]
    if not world_exact.empty:
        row = world_exact.iloc[0]
        return float(row["grade"]), "world_exact_year", "Total", int(year)

    world_nearest = _nearest_row(world_rows, year)
    if world_nearest is not None:
        return (
            float(world_nearest["grade"]),
            "world_nearest_year",
            "Total",
            int(world_nearest["year"]),
        )

    raise ValueError(
        f"No ore grade fallback for exporter={country!r}, trade_year={year}"
    )


def resolve_grade(
    country: str, year: int, grade_table: pd.DataFrame
) -> tuple[float, str, str, int]:
    """Resolve one exporter/year using the explicit four-level fallback order."""
    validated = _validated_grade_table(grade_table)
    return _resolve_grade_from_validated(country, int(year), validated)


def apply_exporter_ore_grades(
    trade: pd.DataFrame, grade_table: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Apply exporter-keyed grades to SITC 281 and return a row-level receipt."""
    required = [
        "commodity_code",
        "trade_year",
        "reporter_iso_a3",
        "partner_iso_a3",
        "weight_unified_kg",
        "iron_eq_ratio_unit_1",
    ]
    missing = [column for column in required if column not in trade.columns]
    if missing:
        raise ValueError(f"Trade table is missing required columns: {missing}")

    validated_grades = _validated_grade_table(grade_table)
    corrected = trade.copy()
    mass_column = "iron_eq_weight_unified_kg"
    if mass_column not in corrected.columns:
        corrected[mass_column] = (
            corrected["weight_unified_kg"] * corrected["iron_eq_ratio_unit_1"]
        )

    iron_mask = corrected["commodity_code"].eq(IRON_ORE_COMMODITY_CODE).to_numpy()
    positions = np.flatnonzero(iron_mask)
    if positions.size == 0:
        return corrected, pd.DataFrame(columns=RECEIPT_COLUMNS)

    selected = corrected.iloc[positions][required].copy()
    selected.insert(0, "trade_row_key", positions.astype(np.int64))
    lookup = selected[["trade_year", "partner_iso_a3"]].drop_duplicates()
    resolutions = [
        _resolve_grade_from_validated(country, int(year), validated_grades)
        for year, country in lookup[["trade_year", "partner_iso_a3"]].itertuples(
            index=False, name=None
        )
    ]
    lookup[["grade", "fallback_level", "matched_country", "matched_year"]] = (
        pd.DataFrame(resolutions, index=lookup.index)
    )

    selected = selected.merge(
        lookup,
        on=["trade_year", "partner_iso_a3"],
        how="left",
        sort=False,
        validate="many_to_one",
    )
    if len(selected) != len(positions) or selected["fallback_level"].isna().any():
        raise RuntimeError("Every SITC 281 record must have exactly one grade resolution")

    selected["corrected_fe_equivalent_mass_kg"] = (
        selected["weight_unified_kg"] * selected["grade"]
    )
    ratio_location = corrected.columns.get_loc("iron_eq_ratio_unit_1")
    mass_location = corrected.columns.get_loc(mass_column)
    corrected.iloc[positions, ratio_location] = selected["grade"].to_numpy()
    corrected.iloc[positions, mass_location] = selected[
        "corrected_fe_equivalent_mass_kg"
    ].to_numpy()

    receipt = selected.rename(
        columns={
            "reporter_iso_a3": "importer_iso_a3",
            "partner_iso_a3": "exporter_iso_a3",
            "weight_unified_kg": "raw_mass_kg",
        }
    )[RECEIPT_COLUMNS]
    return corrected, receipt.reset_index(drop=True)

