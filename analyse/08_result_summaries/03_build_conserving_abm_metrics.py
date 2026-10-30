"""Step 3: extend the canonical mass-conserving ABM manuscript metrics."""

from __future__ import annotations

from pathlib import Path as _DataPath
import sys as _data_sys
_ANALYSE_ROOT = next(p for p in _DataPath(__file__).resolve().parents if (p / 'run_analyse.py').is_file())
if str(_ANALYSE_ROOT / 'library') not in _data_sys.path:
    _data_sys.path.insert(0, str(_ANALYSE_ROOT / 'library'))
from steel_supply.pipeline_paths import data_path, science_path, bundle_root


import json
from datetime import date
from pathlib import Path
import sys

import pandas as pd
import pyarrow.dataset as ds


REPO = bundle_root()
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from steel_supply.abm_annual_metrics import add_loss, annualize, overlap_pieces
from steel_supply.hazard_dynamic.reproduction_grid import resolve_run_path

OUT = data_path('results/postprocessing/formal_abm_metrics')


def persistent_onset(series: pd.Series, threshold: float = 0.0) -> int | None:
    ordered = series.sort_index()
    for year in ordered.index:
        if bool((ordered.loc[year:] > threshold).all()):
            return int(year)
    return None


def chosen_steps(run: Path, years: set[int] | None = None) -> list[dict[str, object]]:
    metadata = json.loads((run / "run_metadata.json").read_text(encoding="utf-8"))
    result: list[dict[str, object]] = []
    for item in metadata["steps"]:
        start = date.fromisoformat(item["start_date"])
        end = date.fromisoformat(item["end_date_exclusive"])
        if years is None or any(year in years for year, _ in overlap_pieces(start, end)):
            result.append(item)
    return result


def final_demand_steps(run: Path, years: set[int] | None = None, by_country: bool = False) -> pd.DataFrame:
    selected = chosen_steps(run, years)
    paths = [run / "flows" / f"flows_step_{int(item['index'])}.parquet" for item in selected]
    dataset = ds.dataset([str(path) for path in paths], format="parquet")
    columns = ["step", "mass"] + (["entity_to"] if by_country else [])
    table = dataset.to_table(
        columns=columns,
        filter=ds.field("sector_to") == "S1",
        use_threads=True,
    )
    frame = table.to_pandas()
    if by_country:
        totals = frame.groupby(["step", "entity_to"], as_index=False)["mass"].sum()
        totals = totals.rename(columns={"entity_to": "country"})
    else:
        totals = frame.groupby("step", as_index=False)["mass"].sum()
    meta = pd.DataFrame(selected).rename(columns={"index": "step"})
    result = totals.merge(meta, on="step", how="left", validate="many_to_one")
    result.attrs["step_calendar"] = selected
    return result


def add_fulfilment(annual: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
    endpoints = annual.loc[annual["year"].isin([2023, 2100])].copy()
    wide = endpoints.pivot_table(
        index=groups,
        columns="year",
        values="annual_mass",
        aggfunc="first",
    ).reset_index()
    wide = wide.loc[wide[2023].gt(0)].copy()
    wide["fulfilment_percent"] = 100.0 * wide[2100].fillna(0.0) / wide[2023]
    wide["loss_percent"] = 100.0 - wide["fulfilment_percent"]
    return wide


metrics = json.loads((data_path('results/postprocessing/formal_abm_metrics/selected_metrics.json')).read_text(encoding="utf-8"))

# Scan the full capacity=1.0 trajectory to verify its onset year.
catalog = pd.read_csv(data_path('results/postprocessing/formal_abm_metrics/scenario_catalog.csv'))
capacity_one = catalog.loc[
    catalog.climate.eq("ssp370")
    & catalog.trade.eq("free-trade")
    & catalog.inventory.eq(4.0)
    & catalog.capacity.eq(1.0)
    & catalog.random_ratio.isna()
].iloc[0]
print("Reading full capacity=1.0 trajectory...", flush=True)
capacity_annual = add_loss(annualize(final_demand_steps(resolve_run_path(capacity_one.run_folder))))
capacity_annual.to_csv(data_path('results/postprocessing/formal_abm_metrics/capacity_1_0_full_annual.csv'), index=False)
capacity_series = capacity_annual.drop_duplicates("year").set_index("year")["loss_percent"]
metrics["capacity"]["1.0"]["persistent_positive_onset"] = persistent_onset(capacity_series)
metrics["capacity"]["1.0"]["persistent_over_0_01_onset"] = persistent_onset(
    capacity_series, 0.01
)

# Recompute the named trade country-level final-demand endpoints.
named = catalog.loc[
    catalog.climate.eq("ssp370")
    & catalog.inventory.eq(4.0)
    & catalog.capacity.eq(1.1)
    & catalog.random_ratio.isna()
    & catalog.trade.isin(["free-trade", "us-cn-decoupling", "bloc-confrontation", "global-decoupling"])
].sort_values("ordinal")
named_rows: list[pd.DataFrame] = []
for row in named.itertuples(index=False):
    print(f"Reading named-trade country endpoints: {row.trade}...", flush=True)
    annual = annualize(
        final_demand_steps(resolve_run_path(row.run_folder), {2023, 2100}, by_country=True),
        ["country"],
        years={2023, 2100},
    )
    fulfilled = add_fulfilment(annual, ["country"])
    fulfilled["trade"] = row.trade
    fulfilled["ordinal"] = int(row.ordinal)
    named_rows.append(fulfilled)
named_country = pd.concat(named_rows, ignore_index=True)
named_country.to_csv(data_path('results/postprocessing/formal_abm_metrics/named_trade_country_final_demand.csv'), index=False)
metrics["named_trade_country"] = {
    trade: {
        country: float(group.loc[group.country.eq(country), "loss_percent"].iloc[0])
        for country in ["CHN", "USA", "CAN", "AUS"]
        if country in set(group.country)
    }
    for trade, group in named_country.groupby("trade")
}

(data_path('results/postprocessing/formal_abm_metrics/extended_selected_metrics.json')).write_text(
    json.dumps(metrics, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
print(json.dumps({
    "e2eci": metrics["e2eci"],
    "capacity_1_0": metrics["capacity"]["1.0"],
    "named_trade_country": metrics["named_trade_country"],
}, ensure_ascii=False, indent=2), flush=True)
