"""Recalculate and draw Figure 2 from the supplied annual global loss data.

No reference image, reference statistics, original repository, Parquet reader,
notebook kernel, climate input or ABM output is used at runtime.
"""

from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from common.figure_style import LIGHT_INK, apply_figure_style, check_reference_font, panel_label

INDUSTRIES = ['iron-mine', 'iron-factory', 'steel-factory']
HAZARDS = ['heatwave', 'tropical-cyclone', 'wildfire']
SCENARIOS = ['picontrol', 'ssp126', 'ssp370', 'ssp585']
TARGET_YEARS = [2035, 2060, 2100]
INPUT_YEARS = range(2018, 2101)
KEYS = ['industry', 'hazard', 'scenario', 'year']
VALUES = ['baseline_production_fe_kg', 'production_loss_fe_kg', 'annual_loss_fraction']
INDUSTRY_LABELS = {'iron-mine': 'Iron mines', 'iron-factory': 'Ironmaking facilities', 'steel-factory': 'Steel plants'}
HAZARD_LABELS = {'heatwave': 'Heatwave', 'tropical-cyclone': 'Tropical cyclone', 'wildfire': 'Wildfire'}
SCENARIO_LABELS = {'picontrol': 'Pre-industrial', 'ssp126': 'SSP1-2.6', 'ssp370': 'SSP3-7.0', 'ssp585': 'SSP5-8.5'}
DEFAULT_DATA = HERE.parents[1] / 'data/Figure_2/annual_global_losses.csv'


def read_annual_data(path: Path) -> pd.DataFrame:
    """Read explicit annual positions, rejecting corruption and silent omissions."""
    data = pd.read_csv(path, float_precision='round_trip')
    required = set(KEYS + VALUES + ['availability'])
    if not required.issubset(data.columns):
        raise ValueError(f'Missing columns: {sorted(required - set(data.columns))}')
    if data[KEYS].isna().any().any() or data.duplicated(KEYS).any():
        raise ValueError('Annual keys must be present and unique.')
    expected = set(itertools.product(INDUSTRIES, HAZARDS, SCENARIOS, INPUT_YEARS))
    observed = set(data[KEYS].itertuples(index=False, name=None))
    if observed != expected:
        raise ValueError('Expected all 2988 annual positions (2018-2100), including explicitly unavailable records.')
    if not data.availability.isin(['available', 'unavailable']).all():
        raise ValueError('Unknown availability status.')
    for column in VALUES:
        data[column] = pd.to_numeric(data[column], errors='raise')
    available = data.availability == 'available'
    measured = data.loc[available, VALUES]
    if not np.isfinite(measured.to_numpy()).all():
        raise ValueError('Available records must have finite production, loss and fraction.')
    if not data.loc[~available, VALUES].isna().all().all():
        raise ValueError('Unavailable records must retain missing numeric values, not zeros.')
    if (measured.baseline_production_fe_kg <= 0).any() or (measured.production_loss_fe_kg < 0).any():
        raise ValueError('Invalid baseline production or loss.')
    if not measured.annual_loss_fraction.between(0, 1).all():
        raise ValueError('Annual loss fractions must be in [0, 1].')
    quotient = measured.production_loss_fe_kg / measured.baseline_production_fe_kg
    if not np.allclose(quotient, measured.annual_loss_fraction, rtol=1e-12, atol=0):
        raise ValueError('Annual loss fractions disagree with the supplied loss/production totals.')
    return data.sort_values(KEYS).reset_index(drop=True)


def _period_stats(frame: pd.DataFrame, center_year: int):
    selected = frame[(frame.availability == 'available') & frame.year.between(center_year - 5, center_year + 5)]
    values = selected.sort_values('year').annual_loss_fraction.to_numpy()
    count = len(values)
    if count == 0:
        return np.nan, np.nan, 0
    mean = np.mean(values)
    half_width = stats.sem(values) * stats.t.ppf(0.975, count - 1) if count > 1 else 0.0
    return mean, half_width, count


def compute_plot_values(data: pd.DataFrame) -> pd.DataFrame:
    """Original available-year means/t-intervals divided by the 2023 baseline."""
    records = []
    for industry in INDUSTRIES:
        for hazard in HAZARDS:
            group = data[(data.industry == industry) & (data.hazard == hazard)]
            baseline, _, baseline_count = _period_stats(group[group.scenario == 'ssp370'], 2023)
            if baseline_count != 11 or not np.isfinite(baseline) or baseline <= 0:
                raise ValueError(f'Invalid 2023 baseline for {industry}/{hazard}; cannot normalize.')
            for target_year in TARGET_YEARS:
                for scenario in SCENARIOS:
                    mean, ci, count = _period_stats(group[group.scenario == scenario], target_year)
                    records.append(dict(
                        industry=industry, hazard=hazard, target_year=target_year, scenario=scenario,
                        mean_risk_ratio=mean / baseline, temporal_95ci_half_width=ci / baseline,
                        baseline_scenario='ssp370', baseline_center_year=2023,
                        requested_window_years=11, available_years=count,
                    ))
    return pd.DataFrame(records)


def render_figure(data: pd.DataFrame):
    """Retain the adopted three-by-three scenario-strip layout."""
    apply_figure_style()
    limits = {}
    for hazard in HAZARDS:
        frame = data[data.hazard == hazard]
        upper = (frame.mean_risk_ratio + frame.temporal_95ci_half_width).max(skipna=True)
        limits[hazard] = math.ceil(max(1.20, float(upper) * 1.07) * 4) / 4
    fills = {'picontrol': '#E7E7E7', 'ssp126': '#C8DFEB', 'ssp370': '#75AECA', 'ssp585': '#174F72'}
    edges = {'picontrol': '#777777', 'ssp126': '#79AFCB', 'ssp370': '#4A87A6', 'ssp585': '#174F72'}
    fig, axes = plt.subplots(3, 3, figsize=(6.15, 4.02), sharey=True)
    y_positions = np.arange(len(TARGET_YEARS), dtype=float)
    offsets = np.linspace(-0.27, 0.27, len(SCENARIOS))

    for row, industry in enumerate(INDUSTRIES):
        for col, hazard in enumerate(HAZARDS):
            ax = axes[row, col]
            for year_index, year in enumerate(TARGET_YEARS):
                panel = data[(data.industry == industry) & (data.hazard == hazard) & (data.target_year == year)].set_index('scenario')
                for offset, scenario in zip(offsets, SCENARIOS):
                    record = panel.loc[scenario]
                    y = y_positions[year_index] + offset
                    if np.isfinite(record.mean_risk_ratio):
                        ax.barh(y, record.mean_risk_ratio, height=0.19, color=fills[scenario], edgecolor='none', linewidth=0, zorder=2)
                        ax.errorbar(record.mean_risk_ratio, y, xerr=record.temporal_95ci_half_width, fmt='none', ecolor=edges[scenario], elinewidth=0.42, capsize=0.9, capthick=0.42, zorder=4)
                    else:
                        ax.barh(y, limits[hazard], height=0.19, color='#F4F4F4', edgecolor='#B7B7B7', linewidth=0.45, linestyle=(0, (2.0, 1.6)), zorder=2)
            ax.axvline(1.0, color=LIGHT_INK, linewidth=0.45, linestyle=(0, (2, 1.7)), zorder=1)
            ax.set_xlim(0, limits[hazard])
            ax.set_ylim(len(TARGET_YEARS) - 0.58, -0.58)
            ax.set_yticks(y_positions, [str(year) for year in TARGET_YEARS])
            ax.grid(False)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(col == 0)
            for side in ('left', 'bottom'):
                ax.spines[side].set_color('#555555')
                ax.spines[side].set_linewidth(0.42)
            ax.tick_params(axis='x', length=2.0, width=0.42)
            ax.tick_params(axis='y', left=col == 0, labelleft=col == 0, length=2.0, width=0.42)
            if row < len(INDUSTRIES) - 1:
                ax.spines['bottom'].set_visible(False)
                ax.tick_params(axis='x', bottom=False, labelbottom=False)
            else:
                ax.set_xlabel('Relative hazard impact')
            panel_label(ax, row * len(HAZARDS) + col, x=-0.10, y=1.04)
            if row == 0:
                ax.set_title(HAZARD_LABELS[hazard], fontweight='bold')
            if col == 0:
                ax.set_ylabel(INDUSTRY_LABELS[industry], labelpad=13, fontweight='bold')

    fig.legend(handles=[Patch(facecolor=fills[s], edgecolor='none', label=SCENARIO_LABELS[s]) for s in SCENARIOS],
               loc='upper center', bbox_to_anchor=(0.54, 0.997), ncol=4, handlelength=1.0, columnspacing=1.3)
    fig.subplots_adjust(left=0.105, right=0.995, top=0.89, bottom=0.105, wspace=0.17, hspace=0.22)
    return fig


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=DEFAULT_DATA, help='Annual global loss CSV; defaults to the bundled input.')
    parser.add_argument('--output-dir', type=Path, default=HERE / 'output', help='Destination for PNG, PDF and recalculated statistics.')
    args = parser.parse_args(argv)
    data = read_annual_data(args.data)
    values = compute_plot_values(data)
    check_reference_font()
    fig = render_figure(values)
    destination = args.output_dir.resolve()
    protected = (HERE / 'reference').resolve()
    if destination == protected or protected in destination.parents:
        plt.close(fig)
        raise ValueError('Do not overwrite the accepted reference directory.')
    destination.mkdir(parents=True, exist_ok=True)
    apply_figure_style()
    try:
        fig.savefig(destination / 'Figure_2.png', dpi=600, bbox_inches='tight', facecolor='white')
        fig.savefig(destination / 'Figure_2.pdf', bbox_inches='tight', facecolor='white')
    finally:
        plt.close(fig)
    values.to_csv(destination / 'Figure_2_plot_values.csv', index=False, float_format='%.17g', na_rep='')
    missing = int((values.available_years == 0).sum())
    incomplete = int(values.available_years.between(1, 10).sum())
    print(f'Calculated {len(values)} plot positions: {missing} unavailable, {incomplete} with fewer than 11 available years.')
    print(f'Wrote Figure_2.png, Figure_2.pdf and Figure_2_plot_values.csv to {destination}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
