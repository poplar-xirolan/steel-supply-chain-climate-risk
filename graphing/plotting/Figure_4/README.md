# Figure 4 — seasonal and cumulative contributions to global production loss

This folder independently redraws the accepted nine-panel Figure 4. The supplied
data retain all relevant countries, not only the displayed top five. The script
recalculates the ranking, Other, global seasonal composition, percentages and
cumulative endpoints; it does not read reference images or reference statistics.

## Install and run

Tested with Python 3.13.5 on Windows and the four dependencies in
`requirements.txt`. From the `reproduce_export/` root:

```text
python -m pip install -r graphing/plotting/Figure_4/requirements.txt
python graphing/plotting/Figure_4/plot_figure4.py
```

The existing environment was tested; a fresh installation on another machine was
not tested. Arial is required for the reference typography but is not bundled.
Missing Arial triggers a warning and font fallback, which may change the layout.
SciencePlots can emit Matplotlib deprecation warnings in this pinned environment.

Paths resolve relative to the script, not the working directory. Keep the
`graphing/plotting/common/` directory and the relative `plotting/`–`data/` layout.
Optional `--data PATH` and `--output-dir PATH` arguments override the defaults.
No Jupyter, SciPy, PyArrow, GeoPandas, LaTeX, network, facility database or
original project package is needed to run this script.

## Input and outputs

The only runtime data input is:

`../../data/Figure_4/country_seasonal_losses.csv`

Outputs in `output/`:

- `Figure_4.png`: 600 dpi, reference dimensions 3675 × 2703 pixels.
- `Figure_4.pdf`: vector plot with embedded font subsets.
- `Figure_4_plot_values.csv`: all 63 recalculated display rows.

`reference/` holds the unchanged accepted PNG/PDF for comparison. Generated
outputs are included after verification and can be regenerated; rerunning
replaces those three output files, not the input or reference files.

## Definitions retained from the accepted analysis

The scenario is SSP3-7.0; the displayed year 2060 represents 2055–2065. The input
records each country's absolute loss for Apr–Sep and Oct–Mar in each calendar
year. Here Oct–Mar combines **Jan–Mar and Oct–Dec of the same year**; it is not
a shifted cross-year interval. These are fixed calendar halves applied globally,
not hemisphere-specific summer and winter definitions.

For each country, losses in each half-year are summed over the eleven years and
divided by eleven. Countries are ranked by the sum of their two mean half-year
losses. The five highest are displayed; exact ties keep alphabetical country-code
order. Other is the remainder of all countries after subtracting those five.

Every coloured segment is divided by the panel's global **total** mean annual
loss (both half-years), not by that country's own losses or production. Each
country row begins at the previous country's cumulative endpoint. The printed
numbers are cumulative endpoints, not the individual country's percentage share.
The global row separately shows the global two-half-year composition.

This is a ratio of period-average absolute losses, not an unweighted average of
annual percentage contributions. No 2023 subtraction or normalization is applied.
Missing rows are rejected rather than filled with zero. Genuine zero losses are
retained. At the CSV-reading boundary, the complete country sets are checked
against the accepted 27/46/77-country stage coverage, so a wholly omitted country
or an unexpected replacement code is also rejected. A hypothetical zero-global-loss panel has an undefined composition and
is shown as N/A; all nine supplied panels have positive global loss.

See `../../data/Figure_4/README.md` for fields, units and provenance. This is
figure-level reproduction from derived inputs, not a climate-model or ABM rerun.
