# Figure 3 — geographic distribution of production risks

One command redraws the complete accepted Figure 3, including the nine country
maps, four facility insets, locator boxes and seven facility labels. The script
uses only the data bundled in `graphing/data/Figure_3/` and the sibling
`plotting/common/` style. It does not require the original repository, full
facility databases, notebooks, previously rendered images, or simulation files.

## Install and run

The tested environment is Python 3.13.5 on Windows. From `reproduce_export/`:

```text
python -m pip install -r graphing/plotting/Figure_3/requirements.txt
python graphing/plotting/Figure_3/plot_figure3.py
```

Keep `plot_figure3.py`, `render_composite.py` and `map_geometry.py` together.
Defaults resolve relative to the script, so another working directory is also
supported. `--data-dir PATH` and `--output-dir PATH` override these defaults.

Install Arial for exact typography; fonts are not bundled. Missing Arial produces
a warning and a fallback font, which can change layout. No LaTeX, online basemap,
draw.io or network connection is needed for plotting. SciencePlots can emit
Matplotlib deprecation warnings in the pinned environment without preventing
rendering. The existing environment was tested; a fresh installation on another
machine was not tested.

## Outputs

`output/` contains:

- `Figure_3.png`, `Figure_3.pdf`, `Figure_3.svg`: complete thirteen-panel figure.
- `Figure_3_country_grid.png`: newly drawn nine-map intermediate.
- `Figure_3_country_values.csv`: recalculated country/global ratios and coverage.
- `Figure_3_facility_values.csv`: recalculated inset facility/global ratios.

The reference dimensions are 4514 × 2231 pixels at 600 dpi. As in the adopted
figure, the country grid is embedded as a raster inside PDF/SVG; facility labels
and leader lines remain vector. The grid is **recomputed and redrawn on every
run**, not copied from `reference/`. Re-running replaces only generated outputs.
The unchanged accepted images in `reference/` are comparison material, never
calculation or rendering inputs.

## Scientific scope

The scenario is SSP3-7.0; the displayed year 2060 represents 2055–2065. Country
annual loss fractions are total annual loss divided by annual baseline production.
The arithmetic mean of these annual fractions is divided by its corresponding
global period mean. This is **not** a relative increase from 2023, an unweighted
average of countries, or a single pooled multi-year loss/production ratio.

Facility insets use the same period and their own all-facility global denominator.
Only inset facilities are included, but that denominator is supplied separately
and is never estimated from the subset. Missing country production remains
missing; zero loss with positive production remains a valid zero. All available
series in the supplied package have eleven annual observations.

Insets: j Brazil mining/heatwave; k Australia mining/tropical cyclone; l India
ironmaking/heatwave; m Indonesia steelmaking/wildfire. Point size is constant.
The seven labels retain the accepted selection based on reported 2023 ore
production for mines and operating nominal capacity for iron/steel facilities.
The released plotting data preserve that fixed selection, not the proprietary
attributes used to rank facilities. See the data README for sources and limits.

## Redistribution status

This local package includes the author-approved small inset subset, not a full
facility database. Inclusion is not an independent determination that third-party
redistribution is permitted. The actual S&P agreement, GEM source-version terms,
and boundary-source terms need to be checked before public release. This does not
affect local reproducibility and does not imply a licence to the source databases.
