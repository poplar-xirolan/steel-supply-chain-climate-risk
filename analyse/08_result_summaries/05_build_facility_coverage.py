"""Step 5: quantify pre-scaling facility coverage against the 2023 MFA."""

from __future__ import annotations

from pathlib import Path as _DataPath
import sys as _data_sys
_ANALYSE_ROOT = next(p for p in _DataPath(__file__).resolve().parents if (p / 'run_analyse.py').is_file())
if str(_ANALYSE_ROOT / 'library') not in _data_sys.path:
    _data_sys.path.insert(0, str(_ANALYSE_ROOT / 'library'))
from steel_supply.pipeline_paths import data_path, science_path, bundle_root


from pathlib import Path
import sys

import pandas as pd

ROOT = bundle_root()
sys.path.insert(0, str(ROOT))

from steel_supply.postprocessing.facility_coverage import (
    StageSpec,
    select_gem_operating_facilities,
    select_spglobal_2023_facilities,
    summarize_stage,
)


INPUT = data_path('input/spatialization')
RESULTS = data_path('results')
OUTPUT_DIR = data_path('results/postprocessing/facility_coverage')


def markdown_table(summary: pd.DataFrame) -> str:
    lines = [
        "# Pre-scaling facility coverage and MFA correspondence",
        "",
        "| Data source | Stage | Source measure | Facilities | Economies represented | MFA production represented (%) | Source/MFA global ratio | Pearson r | Spearman rho | Median MFA/source scaling factor |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.source} | {row.stage} | {row.source_measure} | "
            f"{row.facilities_retained:,} | {row.economies_represented}/{row.producing_economies} | "
            f"{row.production_coverage_pct:.1f} | "
            f"{row.source_to_represented_mfa_ratio:.3f} | "
            f"{row.pearson_r:.3f} | {row.spearman_rho:.3f} | "
            f"{row.mfa_to_source_scaling_median:.3f} |"
        )
    lines.extend(
        [
            "",
            "The comparison uses the unscaled Fe-equivalent facility quantities retained by the formal spatialization pipeline. S&P Global supplies 2023 mine production. GEM supplies nominal annual ironmaking and steelmaking capacity, so its source/MFA ratios describe capacity relative to production rather than measurement error. Production coverage is the share of global 2023 MFA production located in economies represented by at least one retained facility. Correlations and scaling-factor summaries use represented MFA-producing economies only.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    mfa = pd.read_parquet(data_path('results/mfa/2023.parquet'))

    sp_basic = pd.read_parquet(data_path('input/spatialization/spglobal_iron_mine/basic_info.parquet'))
    sp_production = pd.read_parquet(data_path('input/spatialization/spglobal_iron_mine/production.parquet'))
    gem_basic = pd.read_parquet(data_path('input/spatialization/global_energy_monitor_iron_steel/basic_info.parquet'))
    gem_capacity = pd.read_parquet(data_path('input/spatialization/global_energy_monitor_iron_steel/capacity.parquet'))

    stages = [
        (
            StageSpec(
                "S&P Global",
                "Iron-ore mining",
                "2023 production",
                "P1",
                "M1",
                data_path('results/facilities/mine_scaled.parquet'),
                "fe_eq_weight_kg",
                "fe_eq_weight_kg_scaled",
            ),
            select_spglobal_2023_facilities(sp_basic, sp_production),
        ),
        (
            StageSpec(
                "Global Energy Monitor",
                "Ironmaking",
                "Nominal annual capacity",
                "P2",
                "M2",
                data_path('results/facilities/iron_factory_scaled.parquet'),
                "fe_eq_weight_kg",
                "fe_eq_weight_kg_scaled",
            ),
            select_gem_operating_facilities(gem_basic, gem_capacity, facility_type="Iron"),
        ),
        (
            StageSpec(
                "Global Energy Monitor",
                "Steelmaking",
                "Nominal annual capacity",
                "P3",
                "M3",
                data_path('results/facilities/steel_factory_scaled.parquet'),
                "fe_eq_weight_kg",
                "fe_eq_weight_kg_scaled",
            ),
            select_gem_operating_facilities(gem_basic, gem_capacity, facility_type="Steel"),
        ),
    ]

    summaries: list[dict[str, object]] = []
    country_tables: list[pd.DataFrame] = []
    for spec, eligible in stages:
        facilities = pd.read_parquet(spec.result_path)
        summary, by_country = summarize_stage(
            mfa=mfa,
            facilities=facilities,
            eligible_source_ids=eligible,
            spec=spec,
        )
        summaries.append(summary)
        country_tables.append(by_country)

    summary_table = pd.DataFrame(summaries)
    country_table = pd.concat(country_tables, ignore_index=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_table.to_csv(data_path('results/postprocessing/facility_coverage/facility_coverage_summary.csv'), index=False)
    country_table.to_csv(data_path('results/postprocessing/facility_coverage/facility_coverage_by_country.csv'), index=False)
    (data_path('results/postprocessing/facility_coverage/facility_coverage_summary.md')).write_text(
        markdown_table(summary_table), encoding="utf-8"
    )
    print(summary_table.to_string(index=False))


if __name__ == "__main__":
    main()
