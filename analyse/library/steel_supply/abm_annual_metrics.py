from steel_supply.pipeline_paths import science_path, bundle_root, data_path
"""Mass-conserving annual totals and calendar-normalized ABM loss rates."""

from collections import defaultdict
from datetime import date

import pandas as pd


def overlap_pieces(start: date, end: date) -> list[tuple[int, int]]:
    pieces = []
    while start < end:
        boundary = min(date(start.year + 1, 1, 1), end)
        pieces.append((start.year, (boundary - start).days))
        start = boundary
    return pieces


def annualize(
    step_totals: pd.DataFrame,
    groups: list[str] | None = None,
    *,
    years: set[int] | None = None,
) -> pd.DataFrame:
    """Split actual step mass; an endpoint request must not emit adjacent years.

    Coverage follows the loaded simulation calendar, not a country's nonzero
    trades. Loaders provide the selected metadata even for zero-flow steps.
    """
    groups = groups or []
    calendar = pd.DataFrame(step_totals.attrs.get("step_calendar", []))
    if calendar.empty:
        calendar = step_totals[["start_date", "end_date_exclusive"]].drop_duplicates()
    coverage = defaultdict(int)
    for row in calendar.itertuples(index=False):
        for year, days in overlap_pieces(date.fromisoformat(row.start_date), date.fromisoformat(row.end_date_exclusive)):
            coverage[year] += days
    values = defaultdict(float)
    for row in step_totals.itertuples(index=False):
        group_values = tuple(getattr(row, group) for group in groups)
        for year, days in overlap_pieces(date.fromisoformat(row.start_date), date.fromisoformat(row.end_date_exclusive)):
            if years is None or year in years:
                values[(*group_values, year)] += float(row.mass) * days / int(row.step_days)
    rows = [
        {**dict(zip(groups, key[:-1])), "year": key[-1],
         "annual_mass": mass, "coverage_days": coverage[key[-1]]}
        for key, mass in values.items()
    ]
    return pd.DataFrame(rows, columns=[*groups, "year", "annual_mass", "coverage_days"]).sort_values(
        [*groups, "year"]
    ).reset_index(drop=True)


def add_loss(frame: pd.DataFrame, groups: list[str] | None = None) -> pd.DataFrame:
    """Loss relative to the 2023 daily mean; annual_mass remains a conserved sum."""
    groups = groups or []
    result = frame.copy()
    result["mean_daily_mass"] = result["annual_mass"] / result["coverage_days"]
    base = result.loc[result.year.eq(2023), [*groups, "mean_daily_mass"]].rename(
        columns={"mean_daily_mass": "baseline_daily_mass"}
    )
    if base.empty:
        raise ValueError("No baseline rows found for 2023")
    result = (
        result.merge(base, on=groups, how="left", validate="many_to_one")
        if groups else result.assign(baseline_daily_mass=float(base.iloc[0]["baseline_daily_mass"]))
    )
    result["loss_percent"] = 100.0 * (1.0 - result["mean_daily_mass"] / result["baseline_daily_mass"])
    return result.drop(columns="baseline_daily_mass")

