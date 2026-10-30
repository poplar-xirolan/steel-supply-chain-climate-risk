"""Rebuild Fig. 3 inset values from current scientific hazard outputs."""
from __future__ import annotations
from steel_supply.pipeline_paths import science_path, bundle_root, data_path

import calendar
from pathlib import Path

import numpy as np
import pandas as pd

FACILITIES = {"iron-mine": "mine_scaled.parquet", "iron-factory": "iron_factory_scaled.parquet", "steel-factory": "steel_factory_scaled.parquet"}
CASES = [("iron-mine", "heatwave"), ("iron-mine", "tropical-cyclone"),
         ("iron-factory", "heatwave"), ("steel-factory", "wildfire")]
YEARS = tuple(range(2055, 2066))


def relative_metrics(annual: pd.DataFrame, years) -> tuple[pd.DataFrame, float]:
    years = set(years)
    if annual.duplicated(["ID_internal", "year"]).any():
        raise ValueError("Duplicate facility-year")
    if any(set(g.year) != years for _, g in annual.groupby("ID_internal")):
        raise ValueError("Incomplete facility year coverage")
    if not np.isfinite(annual[["production", "loss"]]).all().all():
        raise ValueError("Nonfinite loss/production")
    if (annual[["production", "loss"]] < 0).any().any() or (annual.loss > annual.production + 1e-10 * annual.production.clip(lower=1)).any():
        raise ValueError("Loss outside physical bounds")
    global_annual = annual.groupby("year")[["production", "loss"]].sum()
    if global_annual.production.le(0).any():
        raise ValueError("No positive global production")
    denominator = float((global_annual.loss / global_annual.production).mean())
    if denominator <= 0:
        raise ValueError("Relative risk undefined for zero global loss")
    annual = annual.assign(loss_fraction=annual.loss / annual.production.where(annual.production.gt(0)))
    result = annual.groupby("ID_internal").agg(mean_loss_fraction=("loss_fraction", "mean"), available_years=("loss_fraction", "count")).reset_index()
    if not result.available_years.isin([0, len(years)]).all():
        raise ValueError("Partial positive-production year coverage")
    result["global_mean_loss_fraction"] = denominator
    result["facility_to_global_ratio"] = result.mean_loss_fraction / denominator
    return result, denominator


def build(root: Path) -> pd.DataFrame:
    result = []
    for stage, hazard in CASES:
        sites = pd.read_parquet(science_path(root, 'results/facilities') / FACILITIES[stage], columns=["ID_internal", "country_a3", "Geo_lat", "Geo_lon"])
        if not sites.ID_internal.is_unique or not sites.Geo_lat.between(-90, 90).all() or not sites.Geo_lon.between(-180, 180).all():
            raise ValueError(f"Invalid facility identity/coordinates: {stage}")
        annual = []
        for year in YEARS:
            path = science_path(root, 'results/hazard_facility_loss') / f"{stage}_{hazard}_{year}_ssp370.parquet"
            frame = pd.read_parquet(path)
            if frame.duplicated(["ID_internal", "day_of_year"]).any() or set(frame.ID_internal) != set(sites.ID_internal):
                raise ValueError(f"Invalid facility-day identity: {path}")
            days = 366 if calendar.isleap(year) else 365
            if not frame.day_of_year.between(1, days).all() or not frame.groupby("ID_internal").day_of_year.nunique().eq(days).all():
                raise ValueError(f"Incomplete daily coverage: {path}")
            numeric = frame[["daily_fe_eq_weight_kg_scaled", "production_loss_kg_scaled"]]
            if not np.isfinite(numeric).all().all() or (numeric < 0).any().any():
                raise ValueError(f"Invalid daily values: {path}")
            if not frame.groupby("ID_internal").country_a3.nunique().eq(1).all():
                raise ValueError(f"Facility has multiple country codes: {path}")
            grouped = frame.groupby("ID_internal").agg(production=("daily_fe_eq_weight_kg_scaled", "sum"), loss=("production_loss_kg_scaled", "sum"), country_a3=("country_a3", "first")).reset_index()
            expected = sites.set_index("ID_internal").country_a3.reindex(grouped.ID_internal).to_numpy()
            if not np.array_equal(expected, grouped.country_a3):
                raise ValueError(f"Country identities differ: {path}")
            # Independent country aggregation is an upstream scientific check, not
            # a reference to a dated audit CSV.
            country = pd.read_parquet(science_path(root, 'results/hazard_country_level_loss') / path.name)
            left = frame.groupby("country_a3")[["daily_fe_eq_weight_kg_scaled", "production_loss_kg_scaled"]].sum()
            right = country.groupby("country_a3")[["daily_fe_eq_weight_kg_scaled", "production_loss_kg_scaled"]].sum()
            if set(left.index) != set(right.index):
                raise ValueError(f"Country coverage differs: {path}")
            np.testing.assert_allclose(left.sort_index(), right.sort_index(), rtol=1e-10, atol=1e-4)
            annual.append(grouped.assign(year=year))
        metrics, _ = relative_metrics(pd.concat(annual, ignore_index=True), YEARS)
        result.append(sites.merge(metrics, on="ID_internal", validate="one_to_one").assign(industry=stage, hazard=hazard))
    return pd.concat(result, ignore_index=True)

