"""Step 2: build manuscript ABM metrics from the sealed 411-run grid."""

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

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from scipy import stats


REPO = bundle_root()
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from steel_supply.abm_annual_metrics import add_loss, annualize, overlap_pieces

ROOT = data_path('results/simulation')
OUT = data_path('results/postprocessing/formal_abm_metrics')
OUT.mkdir(parents=True, exist_ok=True)


def command_args(command: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for index, value in enumerate(command[:-1]):
        if value.startswith("--"):
            result[value[2:].replace("-", "_")] = command[index + 1]
    return result


from steel_supply.hazard_dynamic.reproduction_grid import load_catalogue

catalog = load_catalogue()
portable_catalog = catalog.copy()
portable_catalog["run_folder"] = portable_catalog.run_folder.map(
    lambda p: Path(p).relative_to(data_path()).as_posix())
portable_catalog.to_csv(data_path('results/postprocessing/formal_abm_metrics/scenario_catalog.csv'), index=False)


def select_one(**criteria: object) -> pd.Series:
    mask = pd.Series(True, index=catalog.index)
    for key, value in criteria.items():
        if value is None:
            mask &= catalog[key].isna()
        else:
            mask &= catalog[key].eq(value)
    selected = catalog.loc[mask]
    if len(selected) != 1:
        raise RuntimeError(f"Expected one match for {criteria}, got {len(selected)}")
    return selected.iloc[0]


def step_totals_for_run(
    run: Path,
    years: set[int] | None = None,
    *,
    country_stage: bool = False,
) -> pd.DataFrame:
    metadata = json.loads((run / "run_metadata.json").read_text(encoding="utf-8"))
    steps = metadata["steps"]
    chosen: list[dict[str, object]] = []
    for item in steps:
        start = date.fromisoformat(item["start_date"])
        end = date.fromisoformat(item["end_date_exclusive"])
        if years is None or any(year in years for year, _ in overlap_pieces(start, end)):
            chosen.append(item)
    paths = [run / "flows" / f"flows_step_{int(item['index'])}.parquet" for item in chosen]
    if not paths or not all(path.exists() for path in paths):
        missing = [str(path) for path in paths if not path.exists()]
        raise FileNotFoundError(f"Missing selected flows: {missing[:3]}")
    dataset = ds.dataset([str(path) for path in paths], format="parquet")
    if not country_stage:
        table = dataset.to_table(
            columns=["step", "mass"],
            filter=ds.field("sector_to") == "S1",
            use_threads=True,
        )
        frame = table.to_pandas()
        totals = frame.groupby("step", as_index=False)["mass"].sum()
    else:
        stage_labels = {
            "P1": "Mining",
            "P2": "Ironmaking",
            "P3": "Steelmaking",
            "P4": "Casting",
            "P5": "Forming",
            "P6": "Manufacturing",
            "S1": "Final demand",
        }
        table = dataset.to_table(
            columns=["step", "mass", "entity_from", "entity_to", "sector_from", "sector_to"],
            filter=(ds.field("sector_from") == "P1")
            | ds.field("sector_to").isin(["P2", "P3", "P4", "P5", "P6", "S1"]),
            use_threads=True,
        )
        frame = table.to_pandas()
        frames: list[pd.DataFrame] = []
        mining = frame.loc[frame["sector_from"].eq("P1"), ["step", "mass", "entity_from"]].copy()
        mining["country"] = mining.pop("entity_from")
        mining["stage"] = stage_labels["P1"]
        frames.append(mining)
        for sector in ["P2", "P3", "P4", "P5", "P6", "S1"]:
            part = frame.loc[frame["sector_to"].eq(sector), ["step", "mass", "entity_to"]].copy()
            part["country"] = part.pop("entity_to")
            part["stage"] = stage_labels[sector]
            frames.append(part)
        totals = (
            pd.concat(frames, ignore_index=True)
            .groupby(["step", "country", "stage"], as_index=False)["mass"]
            .sum()
        )
    meta = pd.DataFrame(chosen).rename(columns={"index": "step"})
    result = totals.merge(meta, on="step", how="left", validate="many_to_one")
    result.attrs["step_calendar"] = chosen
    return result


def annotate(frame: pd.DataFrame, scenario: pd.Series) -> pd.DataFrame:
    result = frame.copy()
    for column in ["ordinal", "climate", "trade", "inventory", "capacity", "random_ratio", "random_seed"]:
        result[column] = scenario[column]
    return result


print("Catalogued 411 sealed scenarios.", flush=True)

# Full annual trajectories needed by Fig. 5a-b.
full_scenarios: list[pd.Series] = []
for climate in ["picontrol", "ssp126", "ssp370", "ssp585"]:
    full_scenarios.append(select_one(climate=climate, trade="free-trade", inventory=4.0, capacity=1.1, random_ratio=None))
for inventory in range(1, 9):
    candidate = select_one(climate="ssp370", trade="free-trade", inventory=float(inventory), capacity=1.1, random_ratio=None)
    if int(candidate.ordinal) not in {int(item.ordinal) for item in full_scenarios}:
        full_scenarios.append(candidate)

full_frames: list[pd.DataFrame] = []
seasonal_frames: list[pd.DataFrame] = []
for index, scenario in enumerate(full_scenarios, start=1):
    print(f"Full trajectory {index}/{len(full_scenarios)}: scenario {int(scenario.ordinal)}", flush=True)
    steps = step_totals_for_run(Path(str(scenario.run_folder)))
    annual = annotate(add_loss(annualize(steps)), scenario)
    full_frames.append(annual)
    if scenario.climate == "ssp370" and scenario.trade == "free-trade":
        seasonal = steps.copy()
        seasonal["interval_mid_date"] = pd.to_datetime(seasonal["start_date"]) + pd.to_timedelta(seasonal["step_days"] / 2, unit="D")
        seasonal["day_of_year"] = seasonal["interval_mid_date"].dt.dayofyear + seasonal["interval_mid_date"].dt.hour / 24.0
        baseline_daily_mass = (
            annual.loc[annual.year.eq(2023), "annual_mass"].iloc[0]
            / annual.loc[annual.year.eq(2023), "coverage_days"].iloc[0]
        )
        seasonal["normalized_seven_day_mass"] = (
            seasonal["mass"] * 7.0 / seasonal["step_days"]
        )
        seasonal["loss_percent"] = 100.0 * (
            1.0 - seasonal["normalized_seven_day_mass"] / (baseline_daily_mass * 7.0)
        )
        seasonal_frames.append(annotate(seasonal, scenario))
full_annual = pd.concat(full_frames, ignore_index=True)
full_annual.to_csv(data_path('results/postprocessing/formal_abm_metrics/full_annual_climate_inventory.csv'), index=False)
pd.concat(seasonal_frames, ignore_index=True).to_csv(data_path('results/postprocessing/formal_abm_metrics/ssp370_inventory_step_series.csv'), index=False)

# Capacity sensitivity and named trade scenarios need 2023 and 2100 only.
endpoint_scenarios: list[pd.Series] = []
for capacity in [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]:
    endpoint_scenarios.append(select_one(climate="ssp370", trade="free-trade", inventory=4.0, capacity=capacity, random_ratio=None))
for trade in ["us-cn-decoupling", "bloc-confrontation", "global-decoupling"]:
    endpoint_scenarios.append(select_one(climate="ssp370", trade=trade, inventory=4.0, capacity=1.1, random_ratio=None))
endpoint_frames: list[pd.DataFrame] = []
for index, scenario in enumerate(endpoint_scenarios, start=1):
    print(f"Endpoint deterministic {index}/{len(endpoint_scenarios)}: scenario {int(scenario.ordinal)}", flush=True)
    steps = step_totals_for_run(Path(str(scenario.run_folder)), {2023, 2100})
    endpoint_frames.append(annotate(add_loss(annualize(steps, years={2023, 2100})), scenario))
endpoints = pd.concat(endpoint_frames, ignore_index=True)
endpoints.to_csv(data_path('results/postprocessing/formal_abm_metrics/capacity_and_named_trade_endpoints.csv'), index=False)

# All 190 random-disruption replicates.
random_catalog = catalog.loc[catalog.trade.eq("random-disruption")].sort_values(["random_ratio", "random_seed"])
random_frames: list[pd.DataFrame] = []
for index, scenario in enumerate(random_catalog.itertuples(index=False), start=1):
    if index == 1 or index % 10 == 0 or index == len(random_catalog):
        print(f"Random endpoints {index}/{len(random_catalog)}: scenario {int(scenario.ordinal)}", flush=True)
    steps = step_totals_for_run(Path(str(scenario.run_folder)), {2023, 2100})
    series = annotate(add_loss(annualize(steps, years={2023, 2100})), pd.Series(scenario._asdict()))
    random_frames.append(series)
random_endpoints = pd.concat(random_frames, ignore_index=True)
random_endpoints.to_csv(data_path('results/postprocessing/formal_abm_metrics/random_trade_endpoints.csv'), index=False)

random_2100 = random_endpoints.loc[random_endpoints.year.eq(2100)].copy()
summary_rows: list[dict[str, object]] = []
for ratio, group in random_2100.groupby("random_ratio"):
    values = group["loss_percent"].to_numpy(dtype=float)
    sem = stats.sem(values)
    half = float(stats.t.ppf(0.975, len(values) - 1) * sem)
    summary_rows.append({
        "random_ratio": ratio,
        "n": len(values),
        "mean_loss_percent": float(values.mean()),
        "sd": float(values.std(ddof=1)),
        "ci95_low": float(values.mean() - half),
        "ci95_high": float(values.mean() + half),
    })
random_summary = pd.DataFrame(summary_rows)
random_summary.to_csv(data_path('results/postprocessing/formal_abm_metrics/random_trade_summary.csv'), index=False)

# Recompute the Fig. 6 country-stage profiles and E2ECI association.
baseline = select_one(climate="ssp370", trade="free-trade", inventory=4.0, capacity=1.1, random_ratio=None)
print(f"Country-stage endpoints: scenario {int(baseline.ordinal)}", flush=True)
country_steps = step_totals_for_run(Path(str(baseline.run_folder)), {2023, 2100}, country_stage=True)
country_annual = add_loss(annualize(country_steps, groups=["country", "stage"], years={2023, 2100}), groups=["country", "stage"])
country_annual.to_csv(data_path('results/postprocessing/formal_abm_metrics/country_stage_annual.csv'), index=False)
wide = country_annual.pivot_table(
    index=["country", "stage"],
    columns="year",
    values="annual_mass",
    aggfunc="first",
).reset_index()
wide = wide.loc[wide[2023].gt(0)].copy()
wide["fulfilment_percent"] = 100.0 * wide[2100].fillna(0.0) / wide[2023]
wide["loss_percent"] = 100.0 - wide["fulfilment_percent"]
wide.to_csv(data_path('results/postprocessing/formal_abm_metrics/country_stage_fulfilment.csv'), index=False)

e2eci_scores = pd.read_csv(
    data_path('results/postprocessing/e2eci/seven_stage_e2eci_country_scores.csv')
)
e2eci = e2eci_scores[["Country", "E2ECI"]].drop_duplicates().rename(columns={"Country": "country"})
final = wide.loc[wide.stage.eq("Final demand"), ["country", "fulfilment_percent", "loss_percent"]]
e2eci_new = e2eci.merge(final, on="country", how="inner", validate="one_to_one")
r, p = stats.pearsonr(e2eci_new["E2ECI"], e2eci_new["fulfilment_percent"])
e2eci_new.to_csv(data_path('results/postprocessing/formal_abm_metrics/e2eci_with_formal_satisfaction.csv'), index=False)


def persistent_onset(series: pd.Series, threshold: float = 0.0) -> int | None:
    ordered = series.sort_index()
    for year in ordered.index:
        if bool((ordered.loc[year:] > threshold).all()):
            return int(year)
    return None


metrics: dict[str, object] = {
    "formal_grid_total": int(len(catalog)),
    "annualization_note": (
        "Each recorded step mass is allocated across calendar years in proportion "
        "to the step's actual overlap days, then summed as annual_mass. "
        "loss_percent compares annual_mass / coverage_days with the 2023 daily mean, "
        "so leap years do not create artificial gains. Endpoint tables contain only "
        "2023 and 2100."
    ),
    "climate": {},
    "inventory": {},
    "capacity": {},
    "named_trade": {},
    "random_trade": {},
    "e2eci": {
        "n": int(len(e2eci_new)),
        "pearson_r": float(r),
        "pearson_p": float(p),
        "ceiling_count": int((e2eci_new.fulfilment_percent >= 100.0 - 1e-9).sum()),
    },
}
for climate in ["picontrol", "ssp126", "ssp370", "ssp585"]:
    series = full_annual.loc[
        full_annual.climate.eq(climate) & full_annual.inventory.eq(4.0),
        ["year", "loss_percent"],
    ].drop_duplicates("year").set_index("year")["loss_percent"]
    metrics["climate"][climate] = {
        "loss_2100": float(series.at[2100]),
        "persistent_positive_onset": persistent_onset(series),
        "persistent_over_0_01_onset": persistent_onset(series, 0.01),
    }
for inventory in range(1, 9):
    series = full_annual.loc[
        full_annual.climate.eq("ssp370") & full_annual.inventory.eq(float(inventory)),
        ["year", "loss_percent"],
    ].drop_duplicates("year").set_index("year")["loss_percent"]
    metrics["inventory"][str(inventory)] = {
        "loss_2100": float(series.at[2100]),
        "persistent_positive_onset": persistent_onset(series),
        "persistent_over_0_01_onset": persistent_onset(series, 0.01),
    }
for capacity in [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]:
    row = endpoints.loc[
        endpoints.trade.eq("free-trade") & endpoints.capacity.eq(capacity) & endpoints.year.eq(2100)
    ].iloc[0]
    metrics["capacity"][str(capacity)] = {
        "loss_2100": float(row.loss_percent),
    }
for trade in ["free-trade", "us-cn-decoupling", "bloc-confrontation", "global-decoupling"]:
    source = full_annual if trade == "free-trade" else endpoints
    row = source.loc[source.trade.eq(trade) & source.climate.eq("ssp370") & source.inventory.eq(4.0) & source.capacity.eq(1.1) & source.year.eq(2100)].iloc[0]
    metrics["named_trade"][trade] = {
        "loss_2100": float(row.loss_percent),
    }
for ratio in [0.05, 0.15, 0.20, 0.50, 0.95]:
    row = random_summary.loc[np.isclose(random_summary.random_ratio, ratio)].iloc[0]
    metrics["random_trade"][str(ratio)] = row.to_dict()

vip = wide.loc[wide.country.isin(["CHN", "USA", "AUS", "IND", "BRA", "CAN"])].copy()
vip.to_csv(data_path('results/postprocessing/formal_abm_metrics/vip_stage_fulfilment.csv'), index=False)
metrics["vip_stage_fulfilment"] = {
    country: {
        row.stage: float(row.fulfilment_percent)
        for row in group.sort_values("stage").itertuples(index=False)
    }
    for country, group in vip.groupby("country")
}

(data_path('results/postprocessing/formal_abm_metrics/selected_metrics.json')).write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
