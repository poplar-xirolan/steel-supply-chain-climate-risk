"""Redraw Supplementary Figure 2 from its bundled annual ABM summaries."""
from __future__ import annotations

import argparse
import calendar
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PLOTTING_ROOT = Path(__file__).resolve().parent.parent
if str(PLOTTING_ROOT) not in sys.path:
    sys.path.insert(0, str(PLOTTING_ROOT))
from common.figure_style import apply_figure_style, check_reference_font

CAPACITY_RATIOS = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
YEARS = list(range(2023, 2101))
REQUIRED_COLUMNS = ['year', 'annual_mass', 'coverage_days', 'mean_daily_mass', 'loss_percent', 'capacity']
ZERO_TOLERANCE_PP = 1e-10


def load_capacity_data(path: Path) -> pd.DataFrame:
    """Validate the inherited input contract and return all annual plotted values.

    The CSV's original loss ratios, including floating-point residues, are
    retained. Recomputed ratios are an independent consistency check, not a
    substitute for absent data. Each capacity setting uses its own 2023 rate.
    """
    frame = pd.read_csv(path)
    missing = set(REQUIRED_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f'Missing input fields: {sorted(missing)}')
    for column in REQUIRED_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors='raise')
    if not np.isfinite(frame[REQUIRED_COLUMNS].to_numpy(dtype=float)).all():
        raise ValueError('Capacity annual values must be finite; missing values are not zero')
    if not frame.year.eq(np.floor(frame.year)).all():
        raise ValueError('Calendar years must be integers')
    frame['year'] = frame.year.astype(int)
    expected = pd.MultiIndex.from_product([CAPACITY_RATIOS, YEARS], names=['capacity', 'year'])
    index = pd.MultiIndex.from_frame(frame[['capacity', 'year']])
    if index.has_duplicates or set(index) != set(expected):
        raise ValueError('Expected one record per capacity/year: six settings, 2023–2100')
    if frame.annual_mass.lt(0).any():
        raise ValueError('Annual delivery mass cannot be negative')
    days = np.array([366 if calendar.isleap(year) else 365 for year in frame.year])
    if not np.array_equal(frame.coverage_days.to_numpy(), days):
        raise ValueError('Annual coverage does not match the complete calendar year')
    if not np.allclose(frame.mean_daily_mass, frame.annual_mass / days, rtol=1e-12, atol=1e-12):
        raise ValueError('Annual delivery mass and mean daily delivery disagree')
    baseline = frame.loc[frame.year.eq(2023)].set_index('capacity').mean_daily_mass
    if baseline.le(0).any():
        raise ValueError('Every capacity setting needs a positive 2023 baseline')
    expected_loss = 100 * (1 - frame.mean_daily_mass / frame.capacity.map(baseline))
    if not np.allclose(frame.loss_percent, expected_loss, rtol=1e-10, atol=1e-10):
        raise ValueError('Loss ratios disagree with delivery relative to the matching 2023 daily rate')
    return frame.pivot(index='year', columns='capacity', values='loss_percent').reindex(
        index=YEARS, columns=CAPACITY_RATIOS)


def create_figure(capacity_data: pd.DataFrame):
    """Retain the adopted figure's construction, scales and six line styles."""
    apply_figure_style()
    capacity_style = {
        1.0: dict(color='#D55E00', linewidth=1.15, linestyle='-', zorder=4),
        1.1: dict(color='#0072B2', linewidth=1.15, linestyle='-', zorder=5),
        1.2: dict(color='#8C8C8C', linewidth=0.7, linestyle=(0, (1, 1)), zorder=2),
        1.3: dict(color='#A0A0A0', linewidth=0.7, linestyle=(0, (3, 1)), zorder=2),
        1.4: dict(color='#B4B4B4', linewidth=0.7, linestyle=(0, (5, 2)), zorder=2),
        1.5: dict(color='#C6C6C6', linewidth=0.7, linestyle=(0, (7, 2)), zorder=2),
    }
    fig, ax = plt.subplots(figsize=(6.15, 2.75))
    for ratio in reversed(CAPACITY_RATIOS):
        series = capacity_data[ratio]
        ax.plot(series.index, series.values, label=f'{ratio:.1f}', **capacity_style[ratio])
    ax.set_facecolor('white')
    ax.grid(False)
    for side in ['top', 'right']:
        ax.spines[side].set_visible(False)
    for side in ['left', 'bottom']:
        ax.spines[side].set_color('#444444')
        ax.spines[side].set_linewidth(0.45)
    ax.tick_params(direction='out', length=2.2, width=0.45, pad=2)
    ax.set_xlabel('Year')
    ax.set_ylabel('Final-demand loss relative to 2023 (%)')
    ax.set_xlim(2023, 2100)
    ax.set_ylim(0, None)
    handles, labels = ax.get_legend_handles_labels()
    order = np.argsort([float(label) for label in labels])
    ax.legend([handles[i] for i in order], [labels[i] for i in order],
              title='Capacity ratio', loc='upper left', ncol=3,
              handlelength=2.2, columnspacing=1.0)
    if np.abs(capacity_data[[1.2, 1.3, 1.4, 1.5]].to_numpy()).max() < ZERO_TOLERANCE_PP:
        ax.annotate('Ratios 1.2-1.5 overlap at zero', xy=(2062, 0),
                    xytext=(2050, max(ax.get_ylim()[1] * 0.11, 0.8)),
                    fontsize=6.2, color='#666666',
                    arrowprops=dict(arrowstyle='-', color='#9A9A9A', linewidth=0.6))
    fig.tight_layout(pad=0.7)
    return fig


def main(argv=None):
    folder = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=folder.parents[1] / 'data/Supplementary_Figure_2')
    parser.add_argument('--output-dir', type=Path, default=folder / 'output')
    args = parser.parse_args(argv)
    try:
        data = load_capacity_data(args.data_dir / 'capacity_sensitivity_annual.csv')
    except (ValueError, OSError) as error:
        parser.error(str(error))
    check_reference_font()
    fig = create_figure(data)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Reapply the same global export defaults used by the adopted renderer.
    apply_figure_style()
    stem = args.output_dir / 'Supplementary_Figure_2'
    fig.savefig(stem.with_suffix('.png'), dpi=600, bbox_inches='tight', facecolor='white')
    fig.savefig(stem.with_suffix('.pdf'), bbox_inches='tight', facecolor='white')
    fig.savefig(stem.with_suffix('.svg'), bbox_inches='tight', facecolor='white')
    fig.savefig(args.output_dir / 'Supplementary_Figure_2_preview.png', dpi=240,
                bbox_inches='tight', facecolor='white')
    plt.close(fig)
    data.to_csv(args.output_dir / 'source_capacity_annual_loss.csv')
    print(f'Saved Supplementary Figure 2: {len(data)} years × {len(data.columns)} capacity settings')
    print(f'Output directory: {args.output_dir.resolve()}')


if __name__ == '__main__':
    main()
