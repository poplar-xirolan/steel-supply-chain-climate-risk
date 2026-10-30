"""Shared reproducible loaders and aggregations for manuscript ABM figures.

The formal simulation grid contains intentionally repeated deterministic parameter
combinations and intentionally replicated random-disruption experiments.  Figure code
therefore uses one deterministic run (the lexicographically first folder) and all random
replicates.  Raw flow rows are never averaged directly: flows are first summed within a
simulation step, then calendar-year rates are calculated with day-overlap weighting.
"""

from __future__ import annotations
from steel_supply.pipeline_paths import science_path, bundle_root, data_path

import re
import warnings
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd
import pyarrow.dataset as pyarrow_dataset
import pyarrow.parquet as pyarrow_parquet


SIMULATION_START_DATE = date(2023, 1, 1)
STEP_DAYS = 7
NORMALIZED_STEPS_PER_YEAR = 52.0
FLOW_STEP_METADATA_COLUMNS = (
    "step_start_date",
    "step_end_date_exclusive",
    "step_days",
    "time_scale_ratio",
)


def _as_date(value: str | date | datetime) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return datetime.strptime(value, "%Y-%m-%d").date()


def _float_string(value: float | int) -> str:
    """Match the current simulation-folder convention, including 1.0 and 4.0."""

    return str(float(value))


def _trade_part(trade_scenario: str, random_disruption_ratio: float | None) -> str:
    if trade_scenario != "random-disruption":
        return trade_scenario
    if random_disruption_ratio is None:
        raise ValueError("random_disruption_ratio is required for random-disruption")
    return f"{trade_scenario}{str(float(random_disruption_ratio))}"


def build_run_prefix(
    *,
    hazard_type: str,
    climate_scenario: str,
    trade_scenario: str,
    inventory_ratio: float,
    capacity_ratio: float,
    random_disruption_ratio: float | None = None,
) -> str:
    """Build the exact parameter prefix shared by all sensitivity-ID folders."""

    return (
        f"{hazard_type}_{climate_scenario}_"
        f"abun-cap{_float_string(capacity_ratio)}_"
        f"{_trade_part(trade_scenario, random_disruption_ratio)}_"
        f"inv{_float_string(inventory_ratio)}_"
    )


def discover_run_folders(
    root_dir: str | Path,
    *,
    hazard_type: str,
    climate_scenario: str,
    trade_scenario: str,
    inventory_ratio: float,
    capacity_ratio: float,
    random_disruption_ratio: float | None = None,
) -> list[Path]:
    """Return matching legacy-direct or verified formal-grid run folders."""

    root = Path(root_dir)
    prefix = build_run_prefix(
        hazard_type=hazard_type,
        climate_scenario=climate_scenario,
        trade_scenario=trade_scenario,
        inventory_ratio=inventory_ratio,
        capacity_ratio=capacity_ratio,
        random_disruption_ratio=random_disruption_ratio,
    )
    direct_matches = [
        path for path in root.iterdir() if path.is_dir() and path.name.startswith(prefix)
    ]
    nested_matches = [
        path
        for path in root.glob("scenario_*/attempt-*/engine_output/*")
        if path.is_dir()
        and path.name.startswith(prefix)
        and (path.parent.parent / "PASS.json").is_file()
    ]
    matches = sorted(
        direct_matches if direct_matches else nested_matches,
        key=lambda path: (path.name, str(path)),
    )
    if not matches:
        raise FileNotFoundError(f"No simulation run matches prefix: {prefix}")
    return matches


def select_run_folders(
    root_dir: str | Path,
    *,
    hazard_type: str,
    climate_scenario: str,
    trade_scenario: str,
    inventory_ratio: float,
    capacity_ratio: float,
    random_disruption_ratio: float | None = None,
    run_policy: str | None = None,
) -> list[Path]:
    """Select one deterministic run or all intentional random replicates.

    ``run_policy=None`` implements the manuscript rule:
    ``first`` for deterministic scenarios and ``all`` for random-disruption.
    Folder-name sorting makes "first" stable across operating systems.
    """

    matches = discover_run_folders(
        root_dir,
        hazard_type=hazard_type,
        climate_scenario=climate_scenario,
        trade_scenario=trade_scenario,
        inventory_ratio=inventory_ratio,
        capacity_ratio=capacity_ratio,
        random_disruption_ratio=random_disruption_ratio,
    )
    policy = run_policy or ("all" if trade_scenario == "random-disruption" else "first")
    if policy not in {"first", "all"}:
        raise ValueError("run_policy must be 'first', 'all', or None")
    if policy == "all":
        return matches
    if len(matches) > 1:
        warnings.warn(
            f"Found {len(matches)} deterministic runs; selecting first: {matches[0].name}",
            stacklevel=2,
        )
    return matches[:1]


def steps_overlapping_date_range(
    start_date: str | date | datetime,
    end_date: str | date | datetime,
) -> list[int]:
    """Return simulation steps with any overlap with an inclusive date range."""

    start = _as_date(start_date)
    end = _as_date(end_date)
    if start < SIMULATION_START_DATE:
        raise ValueError(f"start_date {start} precedes simulation start")
    if end < start:
        raise ValueError("end_date precedes start_date")
    start_step = (start - SIMULATION_START_DATE).days // STEP_DAYS
    end_step = (end - SIMULATION_START_DATE).days // STEP_DAYS
    return list(range(start_step, end_step + 1))


def _step_interval(step: int) -> tuple[date, date]:
    start = SIMULATION_START_DATE + timedelta(days=step * STEP_DAYS)
    return start, start + timedelta(days=STEP_DAYS)


def _step_interval_from_metadata(
    step: int,
    *,
    start_value: object | None = None,
    end_value: object | None = None,
    step_days_value: object | None = None,
) -> tuple[date, date]:
    """Prefer the engine's actual interval, including its six-day terminal step."""

    try:
        actual_start = pd.Timestamp(start_value).date() if pd.notna(start_value) else None
    except (TypeError, ValueError):
        actual_start = None
    try:
        actual_end = pd.Timestamp(end_value).date() if pd.notna(end_value) else None
    except (TypeError, ValueError):
        actual_end = None
    if actual_start is not None and actual_end is not None and actual_end > actual_start:
        return actual_start, actual_end
    if actual_start is not None and pd.notna(step_days_value):
        actual_days = int(step_days_value)
        if actual_days > 0:
            return actual_start, actual_start + timedelta(days=actual_days)
    return _step_interval(step)


def _range_overlap_interval(
    step_start: date,
    step_end: date,
    range_start: date,
    range_end_inclusive: date,
) -> tuple[date, date, int]:
    range_end = range_end_inclusive + timedelta(days=1)
    overlap_start = max(step_start, range_start)
    overlap_end = min(step_end, range_end)
    overlap_days = max(0, (overlap_end - overlap_start).days)
    return overlap_start, overlap_end, overlap_days


def _range_overlap(
    step: int,
    range_start: date,
    range_end_inclusive: date,
) -> tuple[date, date, int]:
    step_start, step_end = _step_interval(step)
    return _range_overlap_interval(
        step_start, step_end, range_start, range_end_inclusive
    )


def load_simulation_flows(
    root_dir: str | Path,
    hazard_type: str,
    climate_scenario: str,
    trade_scenario: str,
    inventory_ratio: float,
    capacity_ratio: float,
    start_date: str | date,
    end_date: str | date,
    random_disruption_ratio: float | None = None,
    target_columns: Sequence[str] | None = None,
    run_policy: str | None = None,
    strict: bool = True,
) -> pd.DataFrame:
    """Load selected flow steps and attach reproducible run/date metadata.

    For an exact calendar-year request, a cross-year boundary step is retained with its
    true overlap dates.  Downstream seasonal plots therefore use the real midpoint rather
    than relabeling a late-December step as early January.
    """

    range_start = _as_date(start_date)
    range_end = _as_date(end_date)
    steps = steps_overlapping_date_range(range_start, range_end)
    folders = select_run_folders(
        root_dir,
        hazard_type=hazard_type,
        climate_scenario=climate_scenario,
        trade_scenario=trade_scenario,
        inventory_ratio=inventory_ratio,
        capacity_ratio=capacity_ratio,
        random_disruption_ratio=random_disruption_ratio,
        run_policy=run_policy,
    )

    requested_columns = list(target_columns) if target_columns else None
    if requested_columns is not None:
        for required in ("step", "year"):
            if required not in requested_columns:
                requested_columns.append(required)

    frames: list[pd.DataFrame] = []
    missing: list[tuple[str, int]] = []
    for folder in folders:
        sensitivity_id = folder.name.split("sensitivity-id-")[-1]
        flows_dir = folder / "flows"
        for step in steps:
            path = flows_dir / f"flows_step_{step}.parquet"
            if not path.exists():
                missing.append((folder.name, step))
                continue
            parquet_file = pyarrow_parquet.ParquetFile(path)
            available_columns = set(parquet_file.schema.names)
            if requested_columns is None:
                read_columns = None
            else:
                read_columns = list(requested_columns)
                read_columns.extend(
                    column
                    for column in FLOW_STEP_METADATA_COLUMNS
                    if column in available_columns and column not in read_columns
                )
            frame = parquet_file.read(columns=read_columns).to_pandas()
            start_value = (
                frame["step_start_date"].iloc[0]
                if "step_start_date" in frame.columns and not frame.empty
                else None
            )
            end_value = (
                frame["step_end_date_exclusive"].iloc[0]
                if "step_end_date_exclusive" in frame.columns and not frame.empty
                else None
            )
            step_days_value = (
                frame["step_days"].iloc[0]
                if "step_days" in frame.columns and not frame.empty
                else None
            )
            step_start, step_end = _step_interval_from_metadata(
                step,
                start_value=start_value,
                end_value=end_value,
                step_days_value=step_days_value,
            )
            overlap_start, overlap_end, overlap_days = _range_overlap_interval(
                step_start, step_end, range_start, range_end
            )
            if overlap_days <= 0:
                continue
            frame["sensitivity_id"] = sensitivity_id
            frame["run_folder"] = folder.name
            frame["step_start_date"] = pd.Timestamp(step_start)
            frame["step_end_date"] = pd.Timestamp(step_end)
            frame["range_overlap_start"] = pd.Timestamp(overlap_start)
            frame["range_overlap_end"] = pd.Timestamp(overlap_end)
            frame["range_overlap_days"] = overlap_days
            frames.append(frame)

    if missing:
        maximum_requested_step = max(steps)
        internal_missing = [(folder, step) for folder, step in missing if step < maximum_requested_step]
        message = f"Missing {len(missing)} requested flow files; first: {missing[0]}"
        if strict and internal_missing:
            raise FileNotFoundError(message)
        warnings.warn(message, stacklevel=2)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_global_s1_step_totals(
    root_dir: str | Path,
    hazard_type: str,
    climate_scenario: str,
    trade_scenario: str,
    inventory_ratio: float,
    capacity_ratio: float,
    start_date: str | date,
    end_date: str | date,
    random_disruption_ratio: float | None = None,
    run_policy: str | None = None,
    strict: bool = True,
) -> pd.DataFrame:
    """Stream flow files and retain only one global S1 total per run-step.

    This is the preferred path for global manuscript figures: it avoids concatenating
    tens of millions of transaction rows before aggregation.
    """

    range_start = _as_date(start_date)
    range_end = _as_date(end_date)
    steps = steps_overlapping_date_range(range_start, range_end)
    folders = select_run_folders(
        root_dir,
        hazard_type=hazard_type,
        climate_scenario=climate_scenario,
        trade_scenario=trade_scenario,
        inventory_ratio=inventory_ratio,
        capacity_ratio=capacity_ratio,
        random_disruption_ratio=random_disruption_ratio,
        run_policy=run_policy,
    )

    rows: list[dict[str, object]] = []
    missing: list[tuple[str, int]] = []
    for folder in folders:
        sensitivity_id = folder.name.split("sensitivity-id-")[-1]
        existing_paths: list[Path] = []
        for step in steps:
            path = folder / "flows" / f"flows_step_{step}.parquet"
            if not path.exists():
                missing.append((folder.name, step))
                continue
            existing_paths.append(path)

        if not existing_paths:
            continue
        dataset = pyarrow_dataset.dataset(
            [str(path) for path in existing_paths], format="parquet"
        )
        metadata_columns = [
            column
            for column in FLOW_STEP_METADATA_COLUMNS
            if column in dataset.schema.names
        ]
        table = dataset.to_table(
            columns=["sector_to", "mass", "year", "step", *metadata_columns],
            filter=pyarrow_dataset.field("sector_to") == "S1",
            use_threads=True,
        )
        selected = table.to_pandas()
        grouping_columns = ["step", "year", *metadata_columns]
        step_totals = (
            selected.groupby(grouping_columns, as_index=False, dropna=False)["mass"]
            .sum()
            .sort_values("step")
        )
        for result in step_totals.itertuples(index=False):
            step = int(result.step)
            step_start, step_end = _step_interval_from_metadata(
                step,
                start_value=getattr(result, "step_start_date", None),
                end_value=getattr(result, "step_end_date_exclusive", None),
                step_days_value=getattr(result, "step_days", None),
            )
            overlap_start, overlap_end, overlap_days = _range_overlap_interval(
                step_start, step_end, range_start, range_end
            )
            if overlap_days <= 0:
                continue
            rows.append(
                {
                    "sensitivity_id": sensitivity_id,
                    "run_folder": folder.name,
                    "step": step,
                    "file_year": int(result.year),
                    "step_start_date": pd.Timestamp(step_start),
                    "step_end_date": pd.Timestamp(step_end),
                    "range_overlap_start": pd.Timestamp(overlap_start),
                    "range_overlap_end": pd.Timestamp(overlap_end),
                    "range_overlap_days": overlap_days,
                    "mass": float(result.mass),
                }
            )

    if missing:
        maximum_requested_step = max(steps)
        internal_missing = [(folder, step) for folder, step in missing if step < maximum_requested_step]
        message = f"Missing {len(missing)} requested flow files; first: {missing[0]}"
        if strict and internal_missing:
            raise FileNotFoundError(message)
        warnings.warn(message, stacklevel=2)
    return pd.DataFrame(rows)


def aggregate_step_mass(
    frame: pd.DataFrame,
    *,
    group_columns: Sequence[str] = (),
    mass_column: str = "mass",
) -> pd.DataFrame:
    """Sum flow rows to the run-step grain before any averaging."""

    if frame.empty:
        return pd.DataFrame()
    required = {
        "sensitivity_id",
        "step",
        "step_start_date",
        "step_end_date",
        "range_overlap_start",
        "range_overlap_end",
        "range_overlap_days",
        mass_column,
        *group_columns,
    }
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(f"Missing columns for step aggregation: {sorted(missing)}")
    keys = [
        "sensitivity_id",
        "step",
        "step_start_date",
        "step_end_date",
        "range_overlap_start",
        "range_overlap_end",
        "range_overlap_days",
        *group_columns,
    ]
    return frame.groupby(keys, as_index=False, dropna=False)[mass_column].sum()


def global_s1_step_totals(frame: pd.DataFrame) -> pd.DataFrame:
    """Return one global S1 total per run and simulation step."""

    return aggregate_step_mass(frame.loc[frame["sector_to"] == "S1"].copy())


def _calendar_year_pieces(step_totals: pd.DataFrame, mass_column: str) -> pd.DataFrame:
    """Split each requested step mass across calendar years by day overlap."""

    base = step_totals.copy()
    base["step_start_date"] = pd.to_datetime(base["step_start_date"])
    base["step_end_date"] = pd.to_datetime(base["step_end_date"])
    base["range_overlap_start"] = pd.to_datetime(base["range_overlap_start"])
    base["range_overlap_end"] = pd.to_datetime(base["range_overlap_end"])
    base["effective_start"] = base[["step_start_date", "range_overlap_start"]].max(axis=1)
    base["effective_end"] = base[["step_end_date", "range_overlap_end"]].min(axis=1)
    base["step_days"] = (base["step_end_date"] - base["step_start_date"]).dt.days
    if base["step_days"].le(0).any():
        raise ValueError("Simulation steps must have a positive duration")
    base["year"] = base["effective_start"].dt.year
    boundary = pd.to_datetime((base["year"] + 1).astype(str) + "-01-01")
    first_end = pd.concat([base["effective_end"], boundary], axis=1).min(axis=1)
    base["calendar_overlap_days"] = (first_end - base["effective_start"]).dt.days
    base["allocated_mass"] = (
        base[mass_column] * base["calendar_overlap_days"] / base["step_days"]
    )

    crossing = base.loc[base["effective_end"].dt.year > base["effective_start"].dt.year].copy()
    if not crossing.empty:
        crossing["year"] = crossing["effective_end"].dt.year
        crossing_start = pd.to_datetime(crossing["year"].astype(str) + "-01-01")
        crossing["calendar_overlap_days"] = (
            crossing["effective_end"] - crossing_start
        ).dt.days
        crossing["allocated_mass"] = (
            crossing[mass_column]
            * crossing["calendar_overlap_days"]
            / crossing["step_days"]
        )
        base = pd.concat([base, crossing], ignore_index=True)
    return base.loc[base["calendar_overlap_days"] > 0].copy()


def annualize_step_totals(
    step_totals: pd.DataFrame,
    *,
    group_columns: Sequence[str] = (),
    mass_column: str = "mass",
) -> pd.DataFrame:
    """Allocate duration-scaled step mass to calendar years without duplication.

    ``annual_mass`` conserves the input step totals.  ``mean_step_mass`` expresses that
    mass per seven covered days, so a terminal step shorter than seven days is comparable
    with regular simulation steps.  ``normalized_annual_mass`` scales that rate to 52
    equivalent seven-day steps for presentation.
    """

    if step_totals.empty:
        return pd.DataFrame()
    pieces = _calendar_year_pieces(step_totals, mass_column)
    run_year_keys = ["sensitivity_id", "year"]
    coverage = (
        pieces[
            ["sensitivity_id", "step", "year", "calendar_overlap_days"]
        ]
        .drop_duplicates()
        .groupby(run_year_keys, as_index=False)["calendar_overlap_days"]
        .sum()
        .rename(columns={"calendar_overlap_days": "coverage_days"})
    )
    keys = ["sensitivity_id", *group_columns, "year"]
    annual = (
        pieces.groupby(keys, as_index=False, dropna=False)["allocated_mass"]
        .sum()
        .rename(columns={"allocated_mass": "annual_mass"})
    )
    annual = annual.merge(coverage, on=run_year_keys, how="left", validate="many_to_one")
    annual["mean_step_mass"] = (
        annual["annual_mass"] * 7.0 / annual["coverage_days"]
    )
    annual["normalized_annual_mass"] = (
        annual["mean_step_mass"] * NORMALIZED_STEPS_PER_YEAR
    )
    days_in_year = pd.to_datetime(annual["year"].astype(str) + "-12-31").dt.dayofyear
    annual["coverage_fraction"] = annual["coverage_days"] / days_in_year
    return annual


def annual_global_s1(frame: pd.DataFrame) -> pd.DataFrame:
    """Convenience wrapper for the manuscript's global final-demand metric."""

    return annualize_step_totals(global_s1_step_totals(frame))


def add_baseline_loss(
    annual: pd.DataFrame,
    *,
    baseline_year: int = 2023,
    group_columns: Sequence[str] = (),
    value_column: str = "mean_step_mass",
) -> pd.DataFrame:
    """Add a within-run loss percentage relative to a baseline year."""

    keys = ["sensitivity_id", *group_columns]
    baseline = annual.loc[annual["year"] == baseline_year, keys + [value_column]].rename(
        columns={value_column: "baseline_mass"}
    )
    if baseline.empty:
        raise ValueError(f"No baseline rows found for {baseline_year}")
    result = annual.merge(baseline, on=keys, how="left", validate="many_to_one")
    result["loss_percent"] = 100.0 * (
        1.0 - result[value_column] / result["baseline_mass"]
    )
    return result


def seasonal_global_s1(frame: pd.DataFrame) -> pd.DataFrame:
    """Return true interval midpoints and global S1 totals for seasonal plots."""

    return seasonal_step_series(global_s1_step_totals(frame))


def seasonal_step_series(step_totals: pd.DataFrame) -> pd.DataFrame:
    """Attach true dates and a seven-day-equivalent mass to step totals."""

    totals = step_totals.copy()
    step_start = pd.to_datetime(totals["step_start_date"])
    step_end = pd.to_datetime(totals["step_end_date"])
    step_days = (step_end - step_start).dt.days.astype(float)
    if step_days.le(0).any():
        raise ValueError("Simulation steps must have a positive duration")
    totals["normalized_step_mass"] = totals["mass"] * STEP_DAYS / step_days
    overlap_start = pd.to_datetime(totals["range_overlap_start"])
    overlap_days = totals["range_overlap_days"].astype(float)
    totals["interval_mid_date"] = overlap_start + pd.to_timedelta(
        overlap_days / 2.0, unit="D"
    )
    totals["day_of_year"] = (
        totals["interval_mid_date"].dt.dayofyear
        + totals["interval_mid_date"].dt.hour / 24.0
    )
    return totals


def sensitivity_id_from_folder(folder: str | Path) -> str:
    match = re.search(r"sensitivity-id-([A-Za-z0-9-]+)$", Path(folder).name)
    return match.group(1) if match else "unknown"

