"""Step 1: build the formal seven-stage product-form E2ECI and HHI diagnostic.

The script reads the canonical 2023 MFA and writes the current E2ECI tables to
``data/results/postprocessing/e2eci``.  The ABM relationship is produced by the
separate formal ABM postprocessor so this builder never imports a stale outcome
table.
"""

from __future__ import annotations

from pathlib import Path as _DataPath
import sys as _data_sys
_ANALYSE_ROOT = next(p for p in _DataPath(__file__).resolve().parents if (p / 'run_analyse.py').is_file())
if str(_ANALYSE_ROOT / 'library') not in _data_sys.path:
    _data_sys.path.insert(0, str(_ANALYSE_ROOT / 'library'))
from steel_supply.pipeline_paths import data_path, science_path, bundle_root


import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


PROJECT_ROOT = bundle_root()
OUTPUT_ROOT = data_path('results/postprocessing/e2eci')
TABLES = OUTPUT_ROOT
NOTES = data_path('results/postprocessing/e2eci/notes')

MFA_PATH = data_path('results/mfa/2023.parquet')

STAGES = ("P1", "P2", "P3", "P4", "P5", "P6", "P7")
CAPACITY_FACTORS = {stage: (1.0 if stage == "P7" else 1.1) for stage in STAGES}
VIP_COUNTRIES = ("CHN", "USA", "AUS", "IND", "BRA", "CAN")
REQUIRED_MFA_COLUMNS = {
    "entity_from",
    "entity_to",
    "sector_from",
    "sector_to",
    "mass",
    "unit",
    "extra",
}
REQUIRED_NON_NULL_COLUMNS = {
    "entity_from",
    "entity_to",
    "sector_from",
    "sector_to",
    "mass",
    "unit",
}


def save_table(frame: pd.DataFrame, stem: str) -> None:
    """Save a review-friendly CSV and a type-preserving Parquet copy."""

    frame.to_csv(TABLES / f"{stem}.csv", index=False)
    frame.to_parquet(TABLES / f"{stem}.parquet", index=False)


def validate_mfa(mfa: pd.DataFrame) -> None:
    missing = REQUIRED_MFA_COLUMNS - set(mfa.columns)
    if missing:
        raise ValueError(f"MFA is missing required columns: {sorted(missing)}")
    if mfa[list(REQUIRED_NON_NULL_COLUMNS)].isna().any().any():
        raise ValueError("MFA contains nulls in required identifier or measure fields")
    if not np.isfinite(mfa["mass"].to_numpy(float)).all():
        raise ValueError("MFA mass contains non-finite values")
    if (mfa["mass"] < 0).any():
        raise ValueError("MFA mass contains negative values")
    if mfa["unit"].nunique() != 1:
        raise ValueError("MFA contains multiple mass units")


def build_demand(mfa: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return positive M8-S1 country demand and a zero-demand audit."""

    rows = mfa[
        mfa["sector_from"].eq("M8") & mfa["sector_to"].eq("S1")
    ].copy()
    if rows.empty:
        raise ValueError("No M8-S1 final-demand rows were found")
    if (rows["entity_from"] != rows["entity_to"]).any():
        raise ValueError("M8-S1 includes off-diagonal country flows")

    demand_all = (
        rows.groupby("entity_to", as_index=False)["mass"]
        .sum()
        .rename(columns={"entity_to": "Country", "mass": "final_demand"})
        .sort_values("Country")
    )
    zero_demand = demand_all[demand_all["final_demand"].le(0)].copy()
    demand = demand_all[demand_all["final_demand"].gt(0)].copy()
    if demand.empty:
        raise ValueError("No country has positive final demand")
    demand["demand_share"] = demand["final_demand"] / demand["final_demand"].sum()
    return demand, zero_demand


def build_modeled_capacity(mfa: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the ABM's baseline real-process capacity rule for P1--P7."""

    countries = sorted(set(mfa["entity_from"]) | set(mfa["entity_to"]))
    grid = pd.MultiIndex.from_product(
        [countries, STAGES], names=["Country", "Stage"]
    ).to_frame(index=False)

    inflow = (
        mfa[mfa["sector_to"].isin(STAGES)]
        .groupby(["entity_to", "sector_to"], as_index=False)["mass"]
        .sum()
        .rename(
            columns={
                "entity_to": "Country",
                "sector_to": "Stage",
                "mass": "total_inflow",
            }
        )
    )
    outflow = (
        mfa[mfa["sector_from"].isin(STAGES)]
        .groupby(["entity_from", "sector_from"], as_index=False)["mass"]
        .sum()
        .rename(
            columns={
                "entity_from": "Country",
                "sector_from": "Stage",
                "mass": "total_outflow",
            }
        )
    )
    capacity = grid.merge(inflow, on=["Country", "Stage"], how="left").merge(
        outflow, on=["Country", "Stage"], how="left"
    )
    capacity[["total_inflow", "total_outflow"]] = capacity[
        ["total_inflow", "total_outflow"]
    ].fillna(0.0)
    capacity["baseline_throughput"] = capacity[
        ["total_inflow", "total_outflow"]
    ].max(axis=1)
    capacity["capacity_factor"] = capacity["Stage"].map(CAPACITY_FACTORS)
    capacity["modeled_capacity"] = (
        capacity["baseline_throughput"] * capacity["capacity_factor"]
    )
    world = capacity.groupby("Stage")["modeled_capacity"].sum()
    if set(world.index) != set(STAGES) or (world <= 0).any():
        raise ValueError("Every P1-P7 stage must have positive global capacity")
    capacity["world_stage_capacity"] = capacity["Stage"].map(world)
    capacity["capacity_share"] = (
        capacity["modeled_capacity"] / capacity["world_stage_capacity"]
    )
    return capacity.sort_values(["Country", "Stage"])


def build_stage_adequacy(
    capacity: pd.DataFrame, demand: pd.DataFrame
) -> pd.DataFrame:
    """Create the explicit positive-demand-country by seven-stage grid."""

    stage = capacity.merge(
        demand[["Country", "final_demand", "demand_share"]],
        on="Country",
        how="inner",
        validate="many_to_one",
    )
    expected_rows = len(demand) * len(STAGES)
    if len(stage) != expected_rows:
        raise ValueError(
            f"Incomplete country-stage grid: expected {expected_rows}, got {len(stage)}"
        )
    stage["adequacy_raw"] = stage["capacity_share"] / stage["demand_share"]
    stage["adequacy_capped"] = stage["adequacy_raw"].clip(lower=0, upper=1)
    stage["adequacy_pct"] = 100 * stage["adequacy_capped"]
    stage["capacity_absent"] = stage["modeled_capacity"].eq(0)
    return stage.sort_values(["Country", "Stage"])


def aggregate_indices(stage: pd.DataFrame, demand: pd.DataFrame) -> pd.DataFrame:
    """Calculate A, entropy balance, HHI balance, and both product scores."""

    country_order = demand["Country"].tolist()
    adequacy = stage.pivot(
        index="Country", columns="Stage", values="adequacy_capped"
    ).reindex(index=country_order, columns=STAGES)
    if adequacy.isna().any().any():
        raise ValueError("Stage-adequacy grid contains missing cells")

    values = adequacy.to_numpy(float)
    scale = values.mean(axis=1)
    row_sum = values.sum(axis=1)
    shares = np.divide(
        values,
        row_sum[:, None],
        out=np.zeros_like(values),
        where=row_sum[:, None] > 0,
    )

    log_terms = np.zeros_like(shares)
    positive = shares > 0
    log_terms[positive] = shares[positive] * np.log(shares[positive])
    entropy = -log_terms.sum(axis=1)
    entropy_balance = np.divide(
        entropy,
        math.log(len(STAGES)),
        out=np.zeros_like(entropy),
        where=row_sum > 0,
    )
    entropy_balance = np.clip(entropy_balance, 0, 1)

    hhi_raw = np.square(shares).sum(axis=1)
    hhi_normalized = (hhi_raw - 1 / len(STAGES)) / (1 - 1 / len(STAGES))
    hhi_normalized = np.where(row_sum > 0, hhi_normalized, 1.0)
    hhi_normalized = np.clip(hhi_normalized, 0, 1)
    hhi_balance = np.where(row_sum > 0, 1 - hhi_normalized, 0.0)

    country = demand.set_index("Country").copy()
    for column_number, stage_name in enumerate(STAGES):
        country[f"adequacy_{stage_name}"] = values[:, column_number]
        country[f"share_{stage_name}"] = shares[:, column_number]
    country["A_scale"] = scale
    country["B_entropy"] = entropy_balance
    country["effective_stage_count"] = np.where(row_sum > 0, np.exp(entropy), 0.0)
    country["hhi_raw"] = hhi_raw
    country["hhi_normalized_concentration"] = hhi_normalized
    country["B_hhi"] = hhi_balance
    country["e2eci_entropy_product"] = np.clip(
        100 * scale * entropy_balance, 0, 100
    )
    country["e2eci_hhi_product"] = np.clip(100 * scale * hhi_balance, 0, 100)
    country["E2ECI"] = country["e2eci_entropy_product"]
    country["E2ECI_HHI_diagnostic"] = country["e2eci_hhi_product"]
    country["hhi_minus_entropy_score"] = (
        country["e2eci_hhi_product"] - country["e2eci_entropy_product"]
    )
    country["positive_stage_count"] = (values > 0).sum(axis=1)
    country["full_stage_count"] = (values >= 1 - 1e-12).sum(axis=1)
    country["zero_stage_count"] = (values == 0).sum(axis=1)
    country["minimum_stage_adequacy"] = values.min(axis=1)
    country["maximum_stage_adequacy"] = values.max(axis=1)
    country["vip_country"] = country.index.isin(VIP_COUNTRIES)

    country["rank_entropy"] = country["e2eci_entropy_product"].rank(
        ascending=False, method="average"
    )
    country["rank_hhi"] = country["e2eci_hhi_product"].rank(
        ascending=False, method="average"
    )
    country["rank_shift_hhi_minus_entropy"] = (
        country["rank_hhi"] - country["rank_entropy"]
    )
    country["absolute_rank_shift"] = country[
        "rank_shift_hhi_minus_entropy"
    ].abs()
    return country.reset_index().sort_values(
        ["e2eci_entropy_product", "Country"], ascending=[False, True]
    )


def correlation_pair(x: pd.Series, y: pd.Series) -> dict[str, float | int]:
    valid = pd.concat([x, y], axis=1).dropna()
    if len(valid) < 3 or valid.iloc[:, 0].nunique() < 2 or valid.iloc[:, 1].nunique() < 2:
        return {"n": len(valid), "pearson_r": np.nan, "spearman_rho": np.nan}
    return {
        "n": len(valid),
        "pearson_r": float(stats.pearsonr(valid.iloc[:, 0], valid.iloc[:, 1]).statistic),
        "spearman_rho": float(stats.spearmanr(valid.iloc[:, 0], valid.iloc[:, 1]).statistic),
    }


def build_diagnostics(
    country: pd.DataFrame,
    stage: pd.DataFrame,
    capacity: pd.DataFrame,
    mfa: pd.DataFrame,
    zero_demand: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    descriptive_rows: list[dict[str, float | int | str]] = []
    for metric in (
        "A_scale",
        "B_entropy",
        "B_hhi",
        "e2eci_entropy_product",
        "e2eci_hhi_product",
    ):
        values = country[metric]
        descriptive_rows.append(
            {
                "metric": metric,
                "n": int(values.notna().sum()),
                "mean": float(values.mean()),
                "sd": float(values.std()),
                "min": float(values.min()),
                "p25": float(values.quantile(0.25)),
                "median": float(values.median()),
                "p75": float(values.quantile(0.75)),
                "p90": float(values.quantile(0.90)),
                "max": float(values.max()),
                "zero_count": int(values.eq(0).sum()),
                "ceiling_count": int(values.ge(1 - 1e-12).sum())
                if metric.startswith(("A_", "B_"))
                else int(values.ge(100 - 1e-10).sum()),
            }
        )
    descriptives = pd.DataFrame(descriptive_rows)

    comparison_rows: list[dict[str, float | int | str]] = []
    score_pair = correlation_pair(
        country["e2eci_entropy_product"], country["e2eci_hhi_product"]
    )
    comparison_rows.append(
        {
            "comparison": "entropy_product_vs_hhi_product",
            **score_pair,
            "mean_absolute_score_difference": float(
                country["hhi_minus_entropy_score"].abs().mean()
            ),
            "p90_absolute_score_difference": float(
                country["hhi_minus_entropy_score"].abs().quantile(0.90)
            ),
            "max_absolute_score_difference": float(
                country["hhi_minus_entropy_score"].abs().max()
            ),
            "median_absolute_rank_shift": float(country["absolute_rank_shift"].median()),
            "p90_absolute_rank_shift": float(country["absolute_rank_shift"].quantile(0.90)),
            "max_absolute_rank_shift": float(country["absolute_rank_shift"].max()),
        }
    )
    if "loss_pct" in country:
        for metric in ("e2eci_entropy_product", "e2eci_hhi_product"):
            pair = correlation_pair(country[metric], country["loss_pct"])
            comparison_rows.append(
                {
                    "comparison": f"{metric}_vs_main_simulation_loss",
                    **pair,
                    "mean_absolute_score_difference": np.nan,
                    "p90_absolute_score_difference": np.nan,
                    "max_absolute_score_difference": np.nan,
                    "median_absolute_rank_shift": np.nan,
                    "p90_absolute_rank_shift": np.nan,
                    "max_absolute_rank_shift": np.nan,
                }
            )
    comparisons = pd.DataFrame(comparison_rows)

    stage_summary = (
        stage.groupby("Stage", as_index=False)
        .agg(
            world_stage_capacity=("world_stage_capacity", "first"),
            positive_capacity_countries_in_scored_sample=(
                "modeled_capacity", lambda x: int((x > 0).sum())
            ),
            zero_capacity_countries_in_scored_sample=(
                "modeled_capacity", lambda x: int((x == 0).sum())
            ),
            countries_at_full_adequacy=("adequacy_capped", lambda x: int((x >= 1 - 1e-12).sum())),
            median_raw_adequacy=("adequacy_raw", "median"),
            median_capped_adequacy=("adequacy_capped", "median"),
            mean_capped_adequacy=("adequacy_capped", "mean"),
        )
        .sort_values("Stage")
    )

    checks = pd.DataFrame(
        [
            {"check": "mfa_rows", "value": len(mfa), "passed": True},
            {
                "check": "exact_duplicate_mfa_rows",
                "value": int(mfa.duplicated().sum()),
                "passed": not mfa.duplicated().any(),
            },
            {
                "check": "negative_mass_rows",
                "value": int((mfa["mass"] < 0).sum()),
                "passed": not (mfa["mass"] < 0).any(),
            },
            {
                "check": "positive_demand_countries",
                "value": country["Country"].nunique(),
                "passed": country["Country"].nunique() == len(country),
            },
            {
                "check": "complete_global_capacity_reference_entities",
                "value": len(set(mfa["entity_from"]) | set(mfa["entity_to"])),
                "passed": len(set(mfa["entity_from"]) | set(mfa["entity_to"]))
                >= country["Country"].nunique(),
            },
            {
                "check": "zero_demand_entities_excluded",
                "value": len(zero_demand),
                "passed": True,
            },
            {
                "check": "country_stage_rows",
                "value": len(stage),
                "passed": len(stage) == len(country) * len(STAGES),
            },
            {
                "check": "maximum_stage_capacity_share_sum_error",
                "value": float(
                    (capacity.groupby("Stage")["capacity_share"].sum() - 1)
                    .abs()
                    .max()
                ),
                "passed": bool(
                    (capacity.groupby("Stage")["capacity_share"].sum() - 1)
                    .abs()
                    .max()
                    < 1e-12
                ),
            },
            {
                "check": "demand_share_sum",
                "value": float(country["demand_share"].sum()),
                "passed": abs(float(country["demand_share"].sum()) - 1) < 1e-12,
            },
            {
                "check": "distinct_stages",
                "value": stage["Stage"].nunique(),
                "passed": set(stage["Stage"]) == set(STAGES),
            },
            {
                "check": "finite_main_scores",
                "value": int(np.isfinite(country["e2eci_entropy_product"]).sum()),
                "passed": np.isfinite(country["e2eci_entropy_product"]).all(),
            },
            {
                "check": "main_score_range_0_100",
                "value": int(country["e2eci_entropy_product"].between(0, 100).sum()),
                "passed": country["e2eci_entropy_product"].between(0, 100).all(),
            },
            {
                "check": "hhi_score_range_0_100",
                "value": int(country["e2eci_hhi_product"].between(0, 100).sum()),
                "passed": country["e2eci_hhi_product"].between(0, 100).all(),
            },
            {
                "check": "all_six_vip_countries_present",
                "value": int(country["vip_country"].sum()),
                "passed": set(country.loc[country["vip_country"], "Country"])
                == set(VIP_COUNTRIES),
            },
        ]
    )
    return descriptives, comparisons, stage_summary, checks


def synthetic_edge_cases() -> pd.DataFrame:
    """Document the score's deterministic behavior in easily interpreted cases."""

    cases = {
        "seven stages fully adequate": np.ones(7),
        "six adequate, one absent": np.array([1, 1, 1, 1, 1, 1, 0], float),
        "three adequate, four absent": np.array([1, 1, 1, 0, 0, 0, 0], float),
        "seven balanced at half demand": np.full(7, 0.5),
        "one adequate, six absent": np.array([1, 0, 0, 0, 0, 0, 0], float),
        "all stages absent": np.zeros(7),
    }
    rows = []
    for name, values in cases.items():
        scale = float(values.mean())
        total = float(values.sum())
        if total == 0:
            entropy_balance = 0.0
            hhi_balance = 0.0
        else:
            shares = values / total
            positive = shares[shares > 0]
            entropy_balance = float(
                np.clip(-(positive * np.log(positive)).sum() / math.log(7), 0, 1)
            )
            hhi = float(np.square(shares).sum())
            hhi_balance = float(1 - (hhi - 1 / 7) / (1 - 1 / 7))
        rows.append(
            {
                "case": name,
                "A_scale": scale,
                "B_entropy": entropy_balance,
                "B_hhi": hhi_balance,
                "e2eci_entropy_product": 100 * scale * entropy_balance,
                "e2eci_hhi_product": 100 * scale * hhi_balance,
            }
        )
    return pd.DataFrame(rows)




def write_summary(
    country: pd.DataFrame,
    comparisons: pd.DataFrame,
    stage_summary: pd.DataFrame,
    checks: pd.DataFrame,
) -> None:
    pair = comparisons.iloc[0]
    vip_columns = [
        "Country",
        "A_scale",
        "B_entropy",
        "e2eci_entropy_product",
        "B_hhi",
        "e2eci_hhi_product",
        "rank_entropy",
        "rank_hhi",
        "zero_stage_count",
    ]
    vip = country[country["vip_country"]][vip_columns].sort_values(
        "e2eci_entropy_product", ascending=False
    )
    largest = country.nlargest(10, "absolute_rank_shift")[[
        "Country",
        "e2eci_entropy_product",
        "e2eci_hhi_product",
        "rank_entropy",
        "rank_hhi",
        "rank_shift_hhi_minus_entropy",
        "zero_stage_count",
    ]]

    loss_lines = []
    for row in comparisons.iloc[1:].itertuples():
        loss_lines.append(
            f"- {row.comparison}: Pearson r={row.pearson_r:.4f}; "
            f"Spearman rho={row.spearman_rho:.4f}; n={int(row.n)}."
        )

    text = f"""# Seven-stage product E2ECI diagnostic summary

## Formula agreement

- Countries with positive 2023 MFA final demand: {len(country)}.
- Entropy-product versus HHI-product: Pearson r={pair.pearson_r:.4f};
  Spearman rho={pair.spearman_rho:.4f}.
- Mean absolute score difference: {pair.mean_absolute_score_difference:.3f} points;
  90th percentile: {pair.p90_absolute_score_difference:.3f}; maximum:
  {pair.max_absolute_score_difference:.3f}.
- Median absolute rank change: {pair.median_absolute_rank_shift:.1f} places;
  90th percentile: {pair.p90_absolute_rank_shift:.1f}; maximum:
  {pair.max_absolute_rank_shift:.1f}.

These comparisons are diagnostics of aggregation-form dependence. They are not
used to select the main formula after observing its association with the outcome.

## Association with the existing main simulation outcome

{chr(10).join(loss_lines) if loss_lines else '- Loss file was unavailable; no outcome comparison was run.'}

## Six established focal countries

```
{vip.to_string(index=False, float_format=lambda value: f'{value:.3f}')}
```

## Largest rank changes

```
{largest.to_string(index=False, float_format=lambda value: f'{value:.3f}')}
```

## Stage coverage

```
{stage_summary.to_string(index=False, float_format=lambda value: f'{value:.4g}')}
```

## Data checks

All checks passed: {bool(checks['passed'].all())}.

```
{checks.to_string(index=False)}
```

## Interpretation boundary

The score measures the combination of average demand-normalized adequacy and
cross-stage balance. It is not a maximum feasible domestic-output fraction. For
example, a synthetic country with six fully adequate stages and one absent stage
scores about 78.9 under the entropy product by construction.
"""
    (data_path('results/postprocessing/e2eci/notes/diagnostic_summary.md')).write_text(text, encoding="utf-8")


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    NOTES.mkdir(parents=True, exist_ok=True)

    mfa = pd.read_parquet(MFA_PATH)
    validate_mfa(mfa)
    demand, zero_demand = build_demand(mfa)
    capacity = build_modeled_capacity(mfa)
    stage = build_stage_adequacy(capacity, demand)
    country = aggregate_indices(stage, demand)

    descriptives, comparisons, stage_summary, checks = build_diagnostics(
        country, stage, capacity, mfa, zero_demand
    )
    if not checks["passed"].all():
        failed = checks.loc[~checks["passed"], "check"].tolist()
        raise RuntimeError(f"Data checks failed: {failed}")

    rank_shifts = country.sort_values(
        ["absolute_rank_shift", "Country"], ascending=[False, True]
    )
    vip = country[country["vip_country"]].sort_values(
        "e2eci_entropy_product", ascending=False
    )

    save_table(capacity, "all_country_modeled_capacity")
    save_table(stage, "positive_demand_country_stage_adequacy")
    save_table(country, "seven_stage_e2eci_country_scores")
    save_table(vip, "seven_stage_e2eci_vip_countries")
    save_table(rank_shifts, "entropy_hhi_rank_shifts")
    save_table(descriptives, "metric_descriptive_statistics")
    save_table(comparisons, "entropy_hhi_diagnostic_comparisons")
    save_table(stage_summary, "stage_data_summary")
    save_table(checks, "data_quality_checks")
    save_table(zero_demand, "excluded_zero_demand_entities")
    save_table(synthetic_edge_cases(), "synthetic_formula_edge_cases")

    # Optional diagnostic rendering is not part of this numerical export.
    write_summary(country, comparisons, stage_summary, checks)

    print(comparisons.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print("\nVIP countries")
    print(
        vip[
            [
                "Country",
                "A_scale",
                "B_entropy",
                "e2eci_entropy_product",
                "B_hhi",
                "e2eci_hhi_product",
                "rank_entropy",
                "rank_hhi",
                "zero_stage_count",
            ]
        ].to_string(index=False, float_format=lambda value: f"{value:.3f}")
    )


if __name__ == "__main__":
    main()
