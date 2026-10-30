"""Redraw accepted Figure 5 from bundled annual and actual-interval summaries."""

from __future__ import annotations

import argparse
import calendar
import json
from pathlib import Path
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DEFAULT_DATA = HERE.parents[1] / 'data/Figure_5'
WEEKS = list(range(1, 9))
CLIMATES = ['picontrol', 'ssp126', 'ssp370', 'ssp585']
ANNUAL_FIELDS = ['year','annual_mass','coverage_days','loss_percent','climate','trade','inventory','capacity']
STEP_FIELDS = ['step','mass','start_date','end_date_exclusive','step_days','loss_percent','climate','trade','inventory','capacity']
BLUE = "#176A93"
ORANGE = "#CF631D"
DARK_ORANGE = "#A43719"
INK = "#25323C"
MUTED = "#62717A"

def validate_annual(frame: pd.DataFrame) -> None:
    """Check all 11 settings, every year and the 2023 daily-delivery ratio."""
    if set(frame.columns) != set(ANNUAL_FIELDS) or frame.isna().any().any():
        raise ValueError('Missing or unexpected annual fields')
    numeric = frame[['year','annual_mass','coverage_days','loss_percent','inventory','capacity']].to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or frame.annual_mass.lt(0).any():
        raise ValueError('Non-finite annual data or negative delivery mass')
    if not (frame.trade.eq('free-trade') & frame.capacity.eq(1.1)).all():
        raise ValueError('Unexpected annual trade/capacity setting')
    settings = {(climate,4) for climate in CLIMATES} | {('ssp370',week) for week in WEEKS}
    expected = {(climate,week,year) for climate,week in settings for year in range(2023,2101)}
    actual = list(frame[['climate','inventory','year']].itertuples(index=False,name=None))
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError('Incomplete or duplicate annual climate/inventory/year grid')
    days = np.array([366 if calendar.isleap(int(year)) else 365 for year in frame.year])
    if not np.array_equal(frame.coverage_days,days):
        raise ValueError('Annual coverage must include every calendar day, including leap days')
    baseline = frame[frame.year.eq(2023)].set_index(['climate','inventory'])
    baseline_daily = baseline.annual_mass / baseline.coverage_days
    if baseline_daily.le(0).any():
        raise ValueError('Every setting requires a positive 2023 baseline')
    keys = pd.MultiIndex.from_frame(frame[['climate','inventory']])
    expected_loss = 100.0 * (1.0 - (frame.annual_mass.to_numpy() / days) / baseline_daily.reindex(keys).to_numpy())
    if not np.allclose(frame.loss_percent,expected_loss,rtol=0,atol=1e-10):
        raise ValueError('Annual loss is inconsistent with delivery/coverage relative to 2023')
    if frame.loss_percent.lt(-1e-10).any() or frame.loss_percent.gt(100+1e-10).any():
        raise ValueError('Annual loss outside the accepted range')


def validate_steps(frame: pd.DataFrame, annual: pd.DataFrame) -> None:
    """Require each true interval for every inventory; do not bridge gaps."""
    if set(frame.columns) != set(STEP_FIELDS) or frame.isna().any().any():
        raise ValueError('Missing or unexpected interval fields')
    numeric = frame[['step','mass','step_days','loss_percent','inventory','capacity']].to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or frame.mass.lt(0).any():
        raise ValueError('Non-finite interval data or negative delivery mass')
    if not (frame.climate.eq('ssp370') & frame.trade.eq('free-trade') & frame.capacity.eq(1.1)).all():
        raise ValueError('Unexpected interval scenario setting')
    if set(frame.inventory) != set(WEEKS) or frame.duplicated(['inventory','step']).any():
        raise ValueError('Missing inventory setting or duplicate interval identity')
    starts = pd.date_range('2099-12-27','2100-12-26',freq='7D')
    ends = pd.date_range('2100-01-03',periods=len(starts),freq='7D')
    ends = ends.where(ends < pd.Timestamp('2101-01-01'),pd.Timestamp('2101-01-01'))
    expected_calendar = set(zip(starts,ends))
    for week,group in frame.groupby('inventory'):
        actual = list(zip(group.start_date,group.end_date_exclusive))
        if len(actual) != len(expected_calendar) or set(actual) != expected_calendar:
            raise ValueError(f'Incomplete, duplicate or shifted 2100 interval calendar for inventory {week}')
    durations = (frame.end_date_exclusive-frame.start_date).dt.total_seconds().to_numpy()/86400
    if not np.array_equal(frame.step_days,durations):
        raise ValueError('Step duration does not match the actual dates')
    baseline = annual[annual.year.eq(2023)&annual.climate.eq('ssp370')].set_index('inventory')
    daily = (baseline.annual_mass/baseline.coverage_days).reindex(frame.inventory).to_numpy()
    expected_loss = 100.0 * (1.0 - (frame.mass.to_numpy()*7.0/frame.step_days.to_numpy())/(daily*7.0))
    if not np.allclose(frame.loss_percent,expected_loss,rtol=0,atol=1e-10):
        raise ValueError('Interval loss is inconsistent with delivery and the 2023 baseline')
    if frame.loss_percent.lt(-1e-10).any() or frame.loss_percent.gt(100+1e-10).any():
        raise ValueError('Interval loss outside the accepted range')


def persistent_onsets(frame: pd.DataFrame) -> dict:
    if list(frame.index) != list(range(2023, 2101)):
        raise ValueError("Expected complete annual evidence for 2023–2100")
    values = frame.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("Missing annual loss cannot be treated as zero")
    mask = np.logical_and.accumulate((values > 0)[::-1], axis=0)[::-1]
    return {
        key: int(frame.index[mask[:, j]][0]) if mask[:, j].any() else None
        for j, key in enumerate(frame.columns)
    }


def load_evidence(data_dir: Path = DEFAULT_DATA):
    # Keep the adopted renderer's parsing semantics. Extraction preserves the
    # original CSV field strings rather than re-serializing parsed floats.
    raw = pd.read_csv(Path(data_dir)/'annual_summary.csv')
    validate_annual(raw)
    # Formal floating-point residue at zero is <1e-10 percentage points.
    # This is numerical normalization, not a positive-loss threshold.
    annual = raw.copy()
    annual.loc[annual.loss_percent.abs() < 1e-10, "loss_percent"] = 0.0
    climate = annual[annual.inventory.eq(4)].pivot(
        index="year", columns="climate", values="loss_percent"
    ).loc[:, CLIMATES]
    inventory = annual[annual.climate.eq("ssp370")].pivot(
        index="year", columns="inventory", values="loss_percent"
    ).loc[:, WEEKS]
    inventory.columns = WEEKS
    steps = pd.read_csv(Path(data_dir)/'within_year_2100.csv',
                        parse_dates=['start_date','end_date_exclusive'])
    validate_steps(steps,raw)
    start, end = pd.Timestamp("2100-01-01"), pd.Timestamp("2101-01-01")
    steps = steps.sort_values(['inventory','start_date']).copy()
    steps["overlap_start"] = steps.start_date.clip(lower=start)
    steps["overlap_end"] = steps.end_date_exclusive.clip(upper=end)
    steps["elapsed_start"] = (steps.overlap_start - start).dt.total_seconds() / 86400
    steps["elapsed_end"] = (steps.overlap_end - start).dt.total_seconds() / 86400
    steps["overlap_days"] = steps.elapsed_end - steps.elapsed_start
    steps["midpoint_day_of_year"] = (steps.elapsed_start + steps.elapsed_end) / 2 + 1
    seasonal = steps.pivot(index="midpoint_day_of_year", columns="inventory", values="loss_percent").loc[:, WEEKS]
    seasonal.columns = WEEKS
    baseline_steps = steps[steps.inventory.eq(4)].sort_values("elapsed_start")
    left, right = baseline_steps.elapsed_start.to_numpy(), baseline_steps.elapsed_end.to_numpy()
    np.testing.assert_array_equal(left[1:], right[:-1])
    assert left[0] == 0 and right[-1] == 365
    edges = np.r_[left, right[-1]]
    assert len(edges) == 54 and np.all(np.diff(edges) > 0)
    for week, group in steps.groupby("inventory"):
        group = group.sort_values("elapsed_start")
        np.testing.assert_array_equal(group.elapsed_start, left)
        np.testing.assert_array_equal(group.elapsed_end, right)
        assert group.overlap_days.sum() == 365
    if not np.isfinite(seasonal.to_numpy()).all():
        raise ValueError("Missing weekly loss cannot be silently bridged")
    np.testing.assert_allclose(
        np.average(seasonal, axis=0, weights=np.diff(edges)),
        inventory.loc[2100], rtol=0, atol=1e-10,
    )
    climate_onsets, inventory_onsets = persistent_onsets(climate), persistent_onsets(inventory)
    if not (climate.loc[2023:2066] == 0).all().all() or not climate[['picontrol','ssp126']].eq(0).all().all():
        raise ValueError('The fixed zero-period annotations do not describe these data')
    return climate, inventory, seasonal, edges, steps, climate_onsets, inventory_onsets


def style_axis(ax, *, grid=True):
    ax.spines[["top", "right"]].set_visible(False)
    for spine in ax.spines.values():
        spine.set_color("#A3ADB3")
        spine.set_linewidth(0.45)
    ax.tick_params(length=2.2, width=0.45, pad=2.5, colors=INK)
    if grid:
        ax.grid(axis="y", color="#E6E9EB", linewidth=0.4, zorder=0)
    ax.set_axisbelow(True)


def make_figure(climate, inventory, seasonal, edges, climate_onsets, inventory_onsets):
    width, height = 180, 173
    bottom_extension = 8
    fig = plt.figure(figsize=(width / 25.4, height / 25.4))

    def axes(x, y, w, h):
        return fig.add_axes([x / width, (y + bottom_extension) / height, w / width, h / height])

    def text(x, y, value, **kwargs):
        return fig.text(x / width, (y + bottom_extension) / height, value, **kwargs)

    def title(letter, x, y, label):
        text(x - 10, y, letter, fontsize=9, fontweight="bold", va="baseline")
        text(x, y, label, fontsize=7.4, fontweight="bold", va="baseline")

    cmap = LinearSegmentedColormap.from_list(
        "shortage_orange", ["#FFFAF3", "#FCE5C7", "#F3BA79", "#E48942", "#C14B21", "#76291B"]
    )
    annual_norm = Normalize(0, 12)
    seasonal_norm = Normalize(0, 21)
    if inventory.to_numpy().max() > annual_norm.vmax or seasonal.to_numpy().max() > seasonal_norm.vmax:
        raise ValueError("Colour limits must contain every plotted loss")

    # a: compact late-century view, with the omitted all-zero period disclosed.
    a = axes(16, 103, 66, 48)
    title("a", 16, 158, "Climate pathways")
    for key, colour, linestyle in [("ssp585", DARK_ORANGE, "-"), ("ssp370", ORANGE, (0, (4, 1.2)))]:
        a.plot(climate.index, climate[key], color=colour, linestyle=linestyle, lw=1.2)
        value = climate.at[2100, key]
        a.plot(2100, value, "o", color=colour, ms=3, clip_on=False)
        label_y = 12.9 if key == "ssp585" else 3.55
        a.text(2100.5, label_y, f"{'SSP5-8.5' if key == 'ssp585' else 'SSP3-7.0'}  {value:.1f}%",
               color=colour, ha="right", va="center", fontsize=6.5, fontweight="bold")
    a.axhline(0, color="#7D8A91", lw=0.6, linestyle=(0, (2, 2)), clip_on=False)
    a.text(2068, 10.1, "Pre-industrial / SSP1-2.6: 0%\n2023–2066: all pathways 0%",
           fontsize=5.6, color=MUTED, linespacing=1.4, va="top")
    a.set(xlim=(2067, 2101), ylim=(0, 14), xticks=[2070, 2080, 2090, 2100], yticks=[0, 4, 8, 12],
          xlabel="Year", ylabel="Annual final-demand loss (%)")
    style_axis(a)
    text(16, 153, "Four-week inventory", color=MUTED, fontsize=6.0)

    # b: all annual inventory data plus the actual persistent onsets and endpoints.
    b_bottom, b_height = 113.5, 37.5
    b = axes(101, b_bottom, 60, b_height)
    title("b", 101, 158, "Inventory buffering")
    text(101, 153, "SSP3-7.0", color=MUTED, fontsize=6.0)
    text(161, 153, "Onset in blue", color=BLUE, fontsize=5.9, ha="right")
    annual_edges = np.arange(2022.5, 2101, 1)
    inv_edges = np.arange(0.5, 9, 1)
    mesh = b.pcolormesh(annual_edges, inv_edges, inventory.T, cmap=cmap, norm=annual_norm, shading="flat", rasterized=True)
    # Existing staircase definition: positive every year from onset through 2100.
    boundary = [max([w for w in WEEKS if inventory_onsets[w] is not None and inventory_onsets[w] <= year], default=0) + 0.5 for year in inventory.index]
    b.stairs(boundary, annual_edges, baseline=None, color=BLUE, lw=0.85, linestyle=(0, (3, 1.5)), clip_on=False)
    for week in WEEKS:
        onset = inventory_onsets[week]
        if onset is not None:
            b.text(onset - 2, week, str(onset), color=BLUE, ha="right", va="center", fontsize=5.8,
                   fontweight="bold" if week == 4 else "normal")
        text(170, b_bottom + b_height * (8.5 - week) / 8, f"{inventory.at[2100, week]:.1f}",
             color=BLUE if week == 4 else INK, fontsize=6.4, ha="center", va="center",
             fontweight="bold" if week == 4 else "normal")
    b.add_patch(Rectangle((2022.5, 3.5), 78, 1, fill=False, edgecolor=BLUE, lw=0.7))
    b.set(xlim=(2022.5, 2100.5), ylim=(8.5, 0.5), yticks=WEEKS, xticks=[2023, 2050, 2075, 2100],
          xlabel="Year", ylabel="Inventory (weeks)")
    b.tick_params(length=0, pad=3)
    b.spines[:].set_visible(False)
    b.get_yticklabels()[3].set_color(BLUE)
    b.get_yticklabels()[3].set_fontweight("bold")
    text(170, 153, "2100\n(%)", fontsize=5.9, color=MUTED, ha="center", va="bottom", linespacing=1.1)

    # Independent annual scale; the within-year heatmap has its own colourbar.
    cbar_ax = axes(105, 97.9, 49, 1.8)
    cb = fig.colorbar(mesh, cax=cbar_ax, orientation="horizontal", ticks=[0, 3, 6, 9, 12])
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, pad=1.5, labelsize=5.7)
    text(105, 101.9, "Final-demand loss (%)", fontsize=6.0, color=MUTED)

    # c: unsmoothed seven-day trajectory and the complete inventory comparison.
    c = axes(16, 43, 145, 34)
    title("c", 16, 83, "Within-year shortages under SSP3-7.0, 2100")
    mean, peak = inventory.at[2100, 4], seasonal[4].max()
    peak_index = int(np.argmax(seasonal[4].to_numpy()))
    peak_midpoint = (edges[peak_index] + edges[peak_index + 1]) / 2
    c.stairs(seasonal[4], edges, baseline=0, fill=True, color="#F8E5D2", lw=0)
    c.stairs(seasonal[4], edges, baseline=None, color=ORANGE, lw=1.15)
    c.axhline(mean, color=BLUE, linestyle=(0, (4, 2)), lw=0.7)
    c.text(4, 14.2, "Four-week inventory", color=ORANGE, fontsize=6.7, fontweight="bold", va="top")
    c.text(4, mean + 0.55, f"Annual mean  {mean:.1f}%", color=BLUE, fontsize=6.6)
    c.annotate(f"Peak  {peak:.1f}%", (peak_midpoint, peak), (peak_midpoint - 43, 14.4),
               color=DARK_ORANGE, fontsize=6.7, ha="left",
               arrowprops={"arrowstyle": "-", "lw": 0.55, "color": DARK_ORANGE})
    months = [(pd.Timestamp(2100, m, 1) - pd.Timestamp(2100, 1, 1)).days for m in [1, 4, 7, 10, 12]]
    c.set(xlim=(0, 365), ylim=(0, 15.5), yticks=[0, 5, 10, 15], xticks=months,
          ylabel="Final-demand loss (%)")
    c.tick_params(axis="x", bottom=False, labelbottom=False)
    style_axis(c)
    c.spines["bottom"].set_visible(False)

    h = axes(16, 12, 145, 24)
    seasonal_mesh = h.pcolormesh(edges, inv_edges, seasonal.T, cmap=cmap, norm=seasonal_norm, shading="flat", rasterized=True)
    h.add_patch(Rectangle((0, 3.5), 365, 1, fill=False, edgecolor=BLUE, lw=0.7))
    h.set(xlim=(0, 365), ylim=(8.5, 0.5), yticks=WEEKS, xticks=months,
          xticklabels=["Jan", "Apr", "Jul", "Oct", "Dec"], ylabel="Inventory (weeks)")
    h.spines[:].set_visible(False)
    h.tick_params(length=0, pad=2.7, labelsize=6)
    h.get_yticklabels()[3].set_color(BLUE)
    h.get_yticklabels()[3].set_fontweight("bold")
    for week in WEEKS:
        text(170, 12 + 24 * (8.5 - week) / 8, f"{seasonal[week].max():.1f}",
             color=BLUE if week == 4 else INK, ha="center", va="center", fontsize=6.1,
             fontweight="bold" if week == 4 else "normal")
    text(170, 37.5, "Peak\n(%)", fontsize=5.9, color=MUTED, ha="center", va="bottom", linespacing=1.1)
    text(16, 38.5, "All inventory settings", fontsize=5.9, color=MUTED)
    text(161, 38.5, "Blue outline: four-week baseline", fontsize=5.9, color=BLUE, ha="right")
    seasonal_cbar_ax = axes(64, -3, 49, 1.8)
    seasonal_cb = fig.colorbar(seasonal_mesh, cax=seasonal_cbar_ax, orientation="horizontal",
                              ticks=[0, 5, 10, 15, 21])
    seasonal_cb.outline.set_visible(False)
    seasonal_cb.ax.tick_params(length=0, pad=1.5, labelsize=5.7)
    text(64, 1, "Final-demand loss (%)", fontsize=6.0, color=MUTED)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=HERE/'output')
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    try:
        font_manager.findfont('Arial',fallback_to_default=False)
    except ValueError:
        warnings.warn('Arial is unavailable; font fallback may change the reference layout.',stacklevel=2)
    climate, inventory, seasonal, edges, steps, co, io = load_evidence(args.data_dir)
    rc = {"font.family": "Arial", "font.size": 6.7, "text.color": INK, "axes.labelcolor": INK,
          "xtick.labelsize": 6.2, "ytick.labelsize": 6.2, "axes.labelsize": 6.7,
          "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "savefig.bbox": None}
    with plt.rc_context(rc):
        fig = make_figure(climate, inventory, seasonal, edges, co, io)
        for suffix, dpi in [(".png", 600), ("_preview.png", 240), (".pdf", 600), (".svg", 600)]:
            fig.savefig(out / f"Figure_5{suffix}", dpi=dpi, facecolor="white", bbox_inches=None)
        plt.close(fig)
    climate.to_csv(out / "source_climate_annual.csv")
    inventory.to_csv(out / "source_inventory_annual.csv")
    seasonal.to_csv(out / "source_inventory_within_year_2100.csv")
    steps.to_csv(out / "source_2100_step_calendar_overlap.csv", index=False)
    endpoints = pd.DataFrame({"inventory_weeks": WEEKS, "persistent_onset": [io[w] for w in WEEKS],
                              "annual_loss_2100_percent": inventory.loc[2100].to_numpy(),
                              "peak_step_loss_2100_percent": seasonal.max().to_numpy()})
    endpoints.to_csv(out / "source_inventory_endpoints.csv", index=False)
    print(json.dumps({"output": str(out), "inventory_endpoints": endpoints.to_dict("records")}, indent=2))


if __name__ == "__main__":
    main()
