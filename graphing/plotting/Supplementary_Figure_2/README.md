# Supplementary Figure 2 — production-capacity sensitivity

This entry redraws the current SI's capacity-sensitivity figure from the complete
bundled annual numerical table. It does not use the original project package,
Notebook, plotting cache, ABM transactions, workbook, reference images or network.
Keep the shared `graphing/plotting/common/` folder: the inherited figure style
is an explicit dependency, as for Figures 2–4.

## Install and run

Tested in the existing Windows/Python 3.13.5 environment. From reproduce_export:

```text
python -m pip install -r graphing/plotting/Supplementary_Figure_2/requirements.txt
python graphing/plotting/Supplementary_Figure_2/plot_supplementary_figure2.py
```

The four direct dependencies are NumPy, pandas, Matplotlib and SciencePlots.
No PyArrow, SciPy, seaborn, LaTeX or draw.io is required. The existing pinned
environment was used without upgrades; a fresh installation on a different
machine has not been tested. SciencePlots 2.2.2 can emit Matplotlib API
deprecation warnings under 3.11; these are not suppressed by this package.

Paths resolve relative to the script, not the working directory. Keep the
`plotting/Supplementary_Figure_2/`, `plotting/common/` and
`data/Supplementary_Figure_2/` relative locations. Optional `--data-dir PATH` and
`--output-dir PATH` override the defaults. Arial is used for the reference
typography; it is not bundled. If unavailable, the script warns before falling
back to another font, which may change text metrics and layout.

## Input and calculations

`../../data/Supplementary_Figure_2/capacity_sensitivity_annual.csv` contains all
468 records: six capacity multipliers (1.0–1.5 at 0.1 intervals), each with all
78 calendar years from 2023 through 2100. The setting is multi-hazard SSP3-7.0,
free trade and a four-week inventory-to-capacity target. The multiplier varies
for expandable production stages P1–P6; P7's end-of-life scrap supply stays at
1.0. Absolute inventory targets also increase with capacity at the fixed
four-week ratio, as in the manuscript model; this is not a separate experiment
holding absolute inventory mass constant.

The loader retains the source loss ratios and checks them against annual mass,
calendar coverage and the corresponding setting's positive 2023 daily delivery.
It rejects missing settings/years, duplicates, non-finite entries, incomplete
leap-year coverage, negative mass and inconsistent ratios. Missing observations
are not filled with zero, dropped, interpolated or smoothed.

Tiny floating-point residues are retained in the input and numerical output.
The existing plot's lower y-limit is zero; the annotation describing the 1.2–1.5
curves as overlapping zero uses an absolute tolerance of 1e-10 percentage points.
It is omitted if new input data no longer support that statement. This check
does not overwrite data values. See the data README for fields and provenance.

## Outputs and comparison

`output/` contains:

- `Supplementary_Figure_2.png`: 600 dpi, 3624 × 1587 pixels in the checked environment.
- `Supplementary_Figure_2_preview.png`: 240 dpi preview.
- `Supplementary_Figure_2.pdf` and `Supplementary_Figure_2.svg`: vector lines and labels.
- `source_capacity_annual_loss.csv`: complete year × capacity matrix of plotted percentages.

The adopted 6.15 × 2.75 inch construction and tight-cropping settings are
retained; cropping determines the final exported canvas size. Colours, line
styles, line order, legend, axes and annotation placement are unchanged.
`reference/` contains the unchanged adopted PNG/PDF for comparison only; the
script never reads them. Rerunning replaces output files, not input/reference
files. Preparation helpers, tests and temporary logs are outside this package
and are not execution dependencies.

This is plotting-level reproduction from model-derived annual results, not a
rerun of climate assessment, upstream reconstruction or the ABM.
