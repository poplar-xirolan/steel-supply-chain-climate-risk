# Figure 2 - climate-related production losses

This folder redraws the accepted Figure 2 from the supplied **annual global
loss data**, recalculating its means and temporal confidence intervals. It does
not read the reference images or reference statistics to produce the figure.

## Requirements

The tested environment is **Python 3.13.5 on Windows**, with the five direct
dependencies pinned in `requirements.txt`. Install them in a Python environment:

```text
python -m pip install -r graphing/plotting/Figure_2/requirements.txt
```

The command above is run from the `reproduce_export` root. This release was
tested in the existing environment, not by a fresh installation on another machine.
Install **Arial** for the reference typography. If Arial is unavailable, the
script warns and uses an available fallback; exact layout matching is then not
expected. Font files are not redistributed.

SciencePlots is an explicit dependency rather than a silent optional fallback.
Its `nature` and `no-latex` styles are followed by the accepted figure settings.
Jupyter, LaTeX, draw.io, PyArrow, climate-model files and ABM trajectories are
not needed to run this script. SciencePlots may emit a Matplotlib deprecation
warning in the pinned environment; it does not prevent rendering.

## Run

From the `reproduce_export` root:

```text
python graphing/plotting/Figure_2/plot_figure2.py
```

Alternatively, run `python plot_figure2.py` inside this folder. Defaults are
resolved from the script's location, so the working directory does not control
which data are read. Retain the relative layout of `plotting/` and `data/`, and
keep the sibling `plotting/common/` directory.

The script reads only `../../data/Figure_2/annual_global_losses.csv` and writes:

- `output/Figure_2.png` - 600 dpi; reference dimensions: 3676 x 2333 pixels.
- `output/Figure_2.pdf` - vector plot with embedded font subsets.
- `output/Figure_2_plot_values.csv` - all 108 recalculated plot positions.

Optional `--data PATH` and `--output-dir PATH` arguments select other input and
output locations. Re-running replaces these three generated output files, not
the input data or `reference/` files. `reference/` contains the unchanged
accepted PNG/PDF for comparison. The generated outputs are included here after
the release check; they can also be regenerated with the command above.

## Statistical definitions retained from the accepted analysis

For each stage, hazard, scenario and year, the annual global loss fraction is
the sum of country-day production losses divided by the sum of corresponding
baseline production. It is **not** an unweighted mean of country loss rates.

The baseline is the SSP3-7.0 mean for 2018-2028, referred to as the **2023
baseline**. Target labels are 2035, 2060 and 2100, representing the available
annual values within center year +/- 5 years. The data end in 2100, so the last
window contains 2095-2100 (six years). For each period, the script calculates
the arithmetic mean and a two-sided 95% Student-t interval across its available
annual fractions, then divides both mean and interval half-width by the
corresponding baseline mean. Baseline uncertainty is not propagated again.
These are temporal intervals, **not climate-model ensemble uncertainty**.

SSP1-2.6 tropical-cyclone input is unavailable before 2061. Its 2035 slots remain
missing, and its 2060 estimates use 2061-2065. Missing input is never filled with
zero. `available_years` records the actual count for every plotted position.
There are 105 available estimates and three unavailable slots in the reference.

Invalid, duplicate or silently omitted input records fail explicitly. All nine
supplied baseline means are positive and have 11 annual values; the script
rejects an invalid baseline instead of inventing a normalizing denominator.

See `../../data/Figure_2/README.md` for fields, units and source provenance.
This is a figure-level reproduction, not a rerun of facility-level risk models.
