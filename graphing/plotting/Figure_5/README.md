# Figure 5 — dynamic shortages and inventory buffering

One command independently redraws the complete accepted Figure 5 from the two
bundled numerical tables. No original project package, ABM transactions,
simulation folders, old plotting caches, notebooks, selected_metrics.json or
network connection is needed. Figure 5 retains its own adopted v3 styling; it
does not use the common style module used by Figures 2–4.

## Install and run

Tested in the existing Windows/Python 3.13.5 environment with the three pinned
dependencies in requirements.txt. From the reproduce_export root:

```text
python -m pip install -r graphing/plotting/Figure_5/requirements.txt
python graphing/plotting/Figure_5/plot_figure5.py
```

Paths resolve relative to the script, not the working directory. Keep the
plotting/Figure_5 and data/Figure_5 relative locations. Optional --data-dir PATH
and --output-dir PATH override the defaults. Only NumPy, pandas and Matplotlib
are needed; no SciencePlots, SciPy, PyArrow, LaTeX or draw.io is required.

Arial must be installed for reference typography and is not bundled. If absent,
the script warns and Matplotlib uses a fallback, which may change the layout.
A fresh installation on another machine has not been tested.

## Inputs

Both files are in ../../data/Figure_5/:

- annual_summary.csv: all 858 annual observations for 2023–2100. Four climate
  pathways at four-week inventory and SSP3-7.0 at inventories of 1–8 weeks give
  11 distinct settings, with their shared four-week setting represented once.
- within_year_2100.csv: all 424 records whose true simulation intervals overlap
  2100, for SSP3-7.0 at all eight inventory settings.

The CSVs retain delivery mass and coverage/dates alongside the adopted loss
ratios. The script checks those ratios against delivery per day relative to
each setting's 2023 baseline before plotting. It rejects incomplete setting/year
grids, duplicate records, missing or non-finite values, invalid mass/coverage,
incomplete or shifted interval calendars and inconsistent loss ratios. Genuine
zero losses are retained; missing values are not filled with zero.

## Computation and display

- Panel a uses four-week inventory; all four climate pathways and all 78 years
  are retained in the input. The accepted display focuses on 2067–2100, with
  the omitted all-zero period and two all-zero pathways explicitly disclosed.
- Panel b shows every annual value for each inventory. The blue boundary and
  labels are recalculated as the first positive year followed by positive
  annual loss in every remaining year through 2100. An isolated earlier loss
  stays in the heatmap but does not move this boundary. The original numerical
  normalization maps annual absolute loss below 1e-10 percentage points to zero.
- Panel c uses real, unsmoothed interval values. Intervals are clipped to 2100,
  giving 53 intervals: first overlap two days, 51 seven-day intervals, last
  interval six days. The day-weighted mean is checked against the annual loss;
  an unweighted average of the 53 values is not the annual mean.
- Annual and within-year colour bars remain independent (0–12% and 0–21%).
  Onset years, 2100 endpoints, peaks and interval edges are calculated anew,
  not read from an endpoint table. The four-week setting remains highlighted.

## Outputs

output/ contains:

- Figure_5.png: accepted 180 × 173 mm canvas, 600 dpi, 4251 × 4086 pixels.
- Figure_5_preview.png: 240 dpi preview.
- Figure_5.pdf and Figure_5.svg: vector text/lines with rasterized heatmap meshes.
- source_climate_annual.csv and source_inventory_annual.csv: recalculated panel
  matrices, including the complete annual window.
- source_inventory_within_year_2100.csv: recalculated interval/inventory matrix.
- source_2100_step_calendar_overlap.csv: input intervals with clipping and
  elapsed-day/overlap fields added.
- source_inventory_endpoints.csv: eight recalculated onset/endpoint/peak rows.

Rerunning replaces these output files, not the input data or reference/ files.
reference/ contains unchanged accepted PNG/PDF/SVG files for comparison; the
script never reads them. Preparation tests and temporary verification records
are not execution dependencies and are kept outside the export package.

See ../../data/Figure_5/README.md for field definitions, units and provenance.
This is reproduction of the figure from model-derived summaries, not a rerun
of the climate assessment or ABM.
