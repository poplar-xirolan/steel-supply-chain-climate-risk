"""Pre-scaling facility coverage against the canonical 2023 MFA system."""

from __future__ import annotations
from steel_supply.pipeline_paths import science_path, bundle_root, data_path

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


OPERATING_STATUSES = {"operating", "operating pre-retirement"}


@dataclass(frozen=True)
class StageSpec:
    source: str
    stage: str
    source_measure: str
    process: str
    market: str
    result_path: Path
    unscaled_mass_column: str
    scaled_mass_column: str


def _require_columns(data: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = sorted(columns - set(data.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def _require_unique_ids(data: pd.DataFrame, label: str) -> None:
    if data["ID_internal"].isna().any():
        raise ValueError(f"{label} contains missing ID_internal values")
    duplicate_count = int(data["ID_internal"].duplicated().sum())
    if duplicate_count:
        raise ValueError(f"{label} contains {duplicate_count} duplicate facility IDs")


def select_spglobal_2023_facilities(
    basic_info: pd.DataFrame,
    production: pd.DataFrame,
    *,
    year: int = 2023,
) -> pd.DataFrame:
    """Return S&P mine IDs with one positive production observation in ``year``."""
    _require_columns(basic_info, {"ID_internal"}, "S&P basic information")
    _require_columns(
        production,
        {"ID_internal", "Prod_year", "Production"},
        "S&P production",
    )
    _require_unique_ids(basic_info, "S&P basic information")
    selected = production.loc[
        production["Prod_year"].eq(year), ["ID_internal", "Production"]
    ].copy()
    _require_unique_ids(selected, f"S&P production for {year}")
    if (~np.isfinite(selected["Production"])).any() or (selected["Production"] <= 0).any():
        raise ValueError(f"S&P production for {year} must be finite and positive")
    missing_basic = sorted(set(selected["ID_internal"]) - set(basic_info["ID_internal"]))
    if missing_basic:
        raise ValueError(f"S&P production has {len(missing_basic)} IDs without basic information")
    return selected.sort_values("ID_internal", kind="stable").reset_index(drop=True)


def select_gem_operating_facilities(
    basic_info: pd.DataFrame,
    capacity: pd.DataFrame,
    *,
    facility_type: str,
) -> pd.DataFrame:
    """Apply the pipeline's GEM status, type and Total-first capacity selection."""
    _require_columns(basic_info, {"ID_internal"}, "GEM basic information")
    _require_columns(
        capacity,
        {"ID_internal", "Status", "Type", "Sub_type", "Capacity"},
        "GEM capacity",
    )
    _require_unique_ids(basic_info, "GEM basic information")
    selected = capacity.loc[
        capacity["Status"].isin(OPERATING_STATUSES)
        & capacity["Type"].eq(facility_type)
    ].copy()
    totals = (
        selected.loc[selected["Sub_type"].eq("Total")]
        .groupby("ID_internal", as_index=False)["Capacity"]
        .sum()
    )
    ids_with_total = set(totals["ID_internal"])
    components = (
        selected.loc[~selected["ID_internal"].isin(ids_with_total)]
        .groupby("ID_internal", as_index=False)["Capacity"]
        .sum()
    )
    aggregated = pd.concat([totals, components], ignore_index=True)
    _require_unique_ids(aggregated, f"GEM {facility_type} capacity after aggregation")
    if (~np.isfinite(aggregated["Capacity"])).any() or (aggregated["Capacity"] <= 0).any():
        raise ValueError(f"GEM {facility_type} capacity must be finite and positive")
    missing_basic = sorted(set(aggregated["ID_internal"]) - set(basic_info["ID_internal"]))
    if missing_basic:
        raise ValueError(f"GEM {facility_type} capacity has {len(missing_basic)} IDs without basic information")
    return aggregated.sort_values("ID_internal", kind="stable").reset_index(drop=True)


def mfa_stage_production(mfa: pd.DataFrame, process: str, market: str) -> pd.DataFrame:
    """Return positive domestic process-to-market production by economy."""
    _require_columns(
        mfa,
        {"entity_from", "entity_to", "sector_from", "sector_to", "mass", "unit"},
        "MFA",
    )
    selected = mfa.loc[
        mfa["entity_from"].eq(mfa["entity_to"])
        & mfa["sector_from"].eq(process)
        & mfa["sector_to"].eq(market)
    ].copy()
    if selected.empty:
        raise ValueError(f"MFA contains no {process}->{market} domestic production")
    if set(selected["unit"].dropna()) != {"kg"}:
        raise ValueError(f"MFA {process}->{market} production must use kg")
    grouped = selected.groupby("entity_from", as_index=False)["mass"].sum()
    if (~np.isfinite(grouped["mass"])).any() or (grouped["mass"] < 0).any():
        raise ValueError(f"MFA {process}->{market} production must be finite and non-negative")
    grouped = grouped.loc[grouped["mass"] > 0].rename(
        columns={"entity_from": "country", "mass": "mfa_production_kg"}
    )
    return grouped.sort_values("country", kind="stable").reset_index(drop=True)


def _correlation(x: pd.Series, y: pd.Series, method: str) -> tuple[float, float]:
    if len(x) < 3 or x.nunique() < 2 or y.nunique() < 2:
        raise ValueError("At least three non-constant country pairs are required")
    result = stats.pearsonr(x, y) if method == "pearson" else stats.spearmanr(x, y)
    return float(result.statistic), float(result.pvalue)


def summarize_stage(
    *,
    mfa: pd.DataFrame,
    facilities: pd.DataFrame,
    eligible_source_ids: pd.DataFrame,
    spec: StageSpec,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Quantify coverage and pre-scaling country-level correspondence."""
    _require_columns(
        facilities,
        {
            "ID_internal",
            "country_a3",
            spec.unscaled_mass_column,
            spec.scaled_mass_column,
        },
        f"{spec.stage} facility results",
    )
    _require_unique_ids(facilities, f"{spec.stage} facility results")
    if facilities["country_a3"].isna().any():
        raise ValueError(f"{spec.stage} facility results contain missing country codes")
    for column, label in (
        (spec.unscaled_mass_column, "unscaled"),
        (spec.scaled_mass_column, "scaled"),
    ):
        if (~np.isfinite(facilities[column])).any() or (facilities[column] <= 0).any():
            raise ValueError(f"{spec.stage} {label} facility masses must be finite and positive")
    unknown_ids = sorted(set(facilities["ID_internal"]) - set(eligible_source_ids["ID_internal"]))
    if unknown_ids:
        raise ValueError(f"{spec.stage} results contain {len(unknown_ids)} ineligible facility IDs")

    production = mfa_stage_production(mfa, spec.process, spec.market)
    facility_country = (
        facilities.groupby("country_a3", as_index=False)
        .agg(
            facility_count=("ID_internal", "nunique"),
            unscaled_source_fe_eq_kg=(spec.unscaled_mass_column, "sum"),
            scaled_facility_fe_eq_kg=(spec.scaled_mass_column, "sum"),
        )
        .rename(columns={"country_a3": "country"})
    )
    comparison = production.merge(facility_country, on="country", how="left")
    comparison["facility_count"] = comparison["facility_count"].fillna(0).astype(int)
    comparison["unscaled_source_fe_eq_kg"] = comparison[
        "unscaled_source_fe_eq_kg"
    ].fillna(0.0)
    comparison["scaled_facility_fe_eq_kg"] = comparison[
        "scaled_facility_fe_eq_kg"
    ].fillna(0.0)
    comparison["represented"] = comparison["facility_count"] > 0

    matched = comparison.loc[comparison["represented"]].copy()
    if matched.empty:
        raise ValueError(f"{spec.stage} has no represented MFA-producing economies")
    if not np.allclose(
        matched["scaled_facility_fe_eq_kg"],
        matched["mfa_production_kg"],
        rtol=1e-10,
        atol=1e-3,
    ):
        raise ValueError(
            f"{spec.stage} scaled facility mass does not reconcile by country to MFA production"
        )

    global_mfa = float(comparison["mfa_production_kg"].sum())
    represented_mfa = float(matched["mfa_production_kg"].sum())
    unscaled_source_total = float(matched["unscaled_source_fe_eq_kg"].sum())
    pearson_r, pearson_p = _correlation(
        matched["unscaled_source_fe_eq_kg"], matched["mfa_production_kg"], "pearson"
    )
    spearman_rho, spearman_p = _correlation(
        matched["unscaled_source_fe_eq_kg"], matched["mfa_production_kg"], "spearman"
    )

    comparison["share_of_global_stage_pct"] = (
        comparison["mfa_production_kg"] / global_mfa * 100
    )
    comparison["source_to_mfa_ratio"] = np.where(
        comparison["represented"],
        comparison["unscaled_source_fe_eq_kg"] / comparison["mfa_production_kg"],
        np.nan,
    )
    comparison["mfa_to_source_scaling_factor"] = np.where(
        comparison["represented"],
        comparison["mfa_production_kg"] / comparison["unscaled_source_fe_eq_kg"],
        np.nan,
    )
    scaling = comparison.loc[
        comparison["represented"], "mfa_to_source_scaling_factor"
    ]

    comparison.insert(0, "source", spec.source)
    comparison.insert(1, "stage", spec.stage)
    comparison.insert(2, "source_measure", spec.source_measure)
    summary = {
        "source": spec.source,
        "stage": spec.stage,
        "source_measure": spec.source_measure,
        "source_facilities_eligible": int(eligible_source_ids["ID_internal"].nunique()),
        "facilities_retained": int(facilities["ID_internal"].nunique()),
        "producing_economies": int(len(comparison)),
        "economies_represented": int(comparison["represented"].sum()),
        "economy_coverage_pct": float(comparison["represented"].mean() * 100),
        "global_mfa_production_kg": global_mfa,
        "represented_mfa_production_kg": represented_mfa,
        "production_coverage_pct": represented_mfa / global_mfa * 100,
        "unscaled_source_total_fe_eq_kg": unscaled_source_total,
        "source_to_represented_mfa_ratio": unscaled_source_total / represented_mfa,
        "mfa_to_source_global_ratio": represented_mfa / unscaled_source_total,
        "countries_compared": int(len(matched)),
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_rho": spearman_rho,
        "spearman_p": spearman_p,
        "mfa_to_source_scaling_q25": float(scaling.quantile(0.25)),
        "mfa_to_source_scaling_median": float(scaling.median()),
        "mfa_to_source_scaling_q75": float(scaling.quantile(0.75)),
    }
    return summary, comparison.sort_values("country", kind="stable").reset_index(drop=True)

