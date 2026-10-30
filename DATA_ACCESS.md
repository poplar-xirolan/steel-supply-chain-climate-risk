# Input sources and sharing boundaries

## Included, explicitly supplied data

`analyse/data/manifest.csv` records each supplied scientific data file, its source identity,
SHA-256 checksum, size and role. `source` is descriptive provenance, **not** a
runtime path. Analysis never opens the author's original `data/` directory.

`provided/` contains shared study results and comparison data. These files are
retained unchanged; the full workflow never substitutes them for newly computed
MFA, climate losses, simulation summaries or figure inputs.

- `provided/mfa/`: final 2023 MFA in kg Fe-equivalent, with its model node codes.
- `provided/country_hazards/`: daily national production/loss constraints, including
  single hazards and the actual multi-hazard-max inputs used by the dynamic model.
- `provided/coverage/`: scientific source availability information. SSP1-2.6
  tropical-cyclone data before 2061 are unavailable, not zero risk.
- `provided/abm_summaries/`: annual, within-year and national summary results.
  `source_run_id` identifies a source scenario; it is not a path to a hidden run.
- `provided/figure_metrics/`: actual-output components and capacity sensitivity.
- `provided/dscci_statistics/`: reference tables; the workflow computes new DSCCI
  outputs from the newly reconstructed MFA rather than copying reference results.
- `provided/plot_auxiliary/Figure_3/`: previously approved regional point and label
  data, annual derived losses and simplified boundaries, retained for comparison.
  The workflow rebuilds its own plotting inputs from source and computed data.
- `definitions/`: country mapping, model trade categories and end-use shares.
- `usgs/`: **four research-curated USGS-named workbook worksheets**. These are
  retained source tables, not claimed to be the original USGS publication files.
  Exact worksheet names and provenance are in the manifest. No new web edition
  has been substituted for the study inputs.

The Fig. 3 subset contains the already accepted 44 mine plotting points and the
necessary open-source iron/steel points. This bounded inclusion does not extend
the author's approval to the complete S&P database. The package does not make a
general assertion that a particular row count creates a redistribution exemption.

## Source inputs required for the full workflow

Place the following **lawfully obtained, study-version** inputs beneath
`analyse/data/external/`. The paths are the file contract of this model, not
promises that publishers distribute data in exactly these formats.

```text
mfa/
  mfa_inputs.xlsx
  trade_raw/*.parquet
spatialization/
  world_administration_esri.gpkg
  spglobal_iron_mine/basic_info.parquet
  spglobal_iron_mine/production.parquet
  global_energy_monitor_iron_steel/basic_info.parquet
  global_energy_monitor_iron_steel/capacity.parquet
isimip/
  tasmax/*.nc
  tas/*.nc
  hurs/*.nc
  wind/*.nc
  wildfire/*.nc
isimip_download_lists/          # five supplied lists
```

The original facility schemas are documented in
`analyse/00_prepare_inputs/INPUT_SCHEMAS.md`. They are normalized source records,
not derived MFA/hazard/model outputs. Preserve source identifiers, units, versions
and calendars when preparing them; do not replace missing observations with zero.

The mixed workbook contains USGS, World Steel Association, UN Comtrade mappings,
and study definitions. It and the raw trade mirror are not republished wholesale.
Users reconstructing the raw path must have that input workbook or recreate its
documented sheets from the relevant sources under their terms. The workbook
structure is listed in `analyse/data/external/workbook_sheets.csv`.
The workflow starts from these study-version normalized source tables and the
listed climate files. Obtaining the inputs and converting provider downloads to
the documented source formats are the user's responsibility, not an automated
one-click reconstruction supplied by this package.

Source access points (not runtime downloads): [UN Comtrade](https://comtradeplus.un.org/),
[USGS iron and steel statistics](https://www.usgs.gov/centers/national-minerals-information-center/iron-and-steel-statistics-and-information),
[World Steel Association](https://worldsteel.org/steel-topics/statistics/),
[S&P Global](https://www.spglobal.com/market-intelligence/),
[Global Energy Monitor](https://globalenergymonitor.org/projects/global-iron-and-steel-tracker/),
and [ISIMIP](https://data.isimip.org/).
Consult the manuscript/SI references and source terms for the study's cited products.

The five exact climate collection URL lists are supplied. Downloading is separate,
explicit and optional; the collection observed in the author workspace is about
10.67 TB. Source access, availability and authentication remain the source's
responsibility; no credentials are embedded or managed here.

```text
python analyse/00_prepare_inputs/01_download_isimip.py --dataset tasmax
```

## What is deliberately not included

Complete proprietary coordinates/attributes, mixed-source workbooks, raw trade
mirrors, bulk Worldsteel tables, the full NetCDF collection, extraction caches,
millions of ABM transaction/agent files, old dispatcher permissions, heartbeats,
private audit logs and author-machine paths are not required public inputs.
All generated intermediates have a declared producer in the full workflow.

Full-facility derived losses have **not** automatically been released merely
because identifiers can be anonymized. Additional sharing needs a source-specific
permission decision. Country aggregates and the previously accepted figure subset
remain included under the agreed release boundary.
