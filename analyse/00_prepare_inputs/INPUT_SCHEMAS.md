# Initial source formats

These are the study's normalized **source-input** contracts. They do not depend on
an MFA result, climate-extraction cache or simulation output. Prepare records from
the cited source versions, preserving identifiers and units. The package does not
silently infer source-specific mappings from a current vendor download.

## MFA

`mfa/trade_raw/*.parquet` contains:

`trade_year`, `reporter_comtrade_id`, `partner_comtrade_id`, `flow_code`,
`commodity_code`, `monetary_usd`, `weight_unified_kg`.

Raw-source identities and the workbook's required sheet names are in
`data/external/required_sources.csv` and `data/external/workbook_sheets.csv`.
Do not retain only 2023 when preparing the trade inputs: the original cleaning
uses the surrounding historical observations for outlier correction.
`mfa_inputs.xlsx` supplies country/category mappings, production/consumption
statistics, ore grades and end-use shares. The exported mapping/USGS CSVs document
parts of that workbook; they are not falsely advertised as a complete replacement
for its non-redistributed Worldsteel and mixed-source inputs.

## Facility sources

Both `basic_info.parquet` files use:

`ID_internal`, `Geo_lat`, `Geo_lon`, `Geo_accuracy`, `Name`, `Src_name`, `Src_id`,
`Raw_json`.

Coordinates are geographic latitude/longitude in degrees. Source identity fields
and nested attributes stay local; the export does not supply the complete S&P
table. `ID_internal` must link unambiguously to production/capacity records.

S&P mine `production.parquet` uses:

`ID_internal`, `ID_prod_internal`, `Unit`, `Is_ore_weight`, `Grade`, `Prod_year`,
`Sampling_year`, `Production`, `Raw_json`.

The retained study production rows use `Thousand metric tons per year`; source
ore weight is not iron-equivalent mass. The existing ore-grade conversion is kept.

GEM `capacity.parquet` uses:

`ID_internal`, `ID_cap_internal`, `Type`, `Sub_type`, `Unit`, `Commissioning_year`,
`Status`, `Sampling_year`, `Capacity`, `Raw_json`.

The existing code selects iron/steel operating units and aggregates their source
capacity. Nominal capacity is not reported actual production. National production
allocation and rescaling occur in step 02, not in these source records.

The two extraction-location tables that previously lived in `climate_facilities/`
were exact copies of these two `basic_info` tables. `prepare-facilities` now creates
those copies under `work/prepared_climate_facilities/`; no pre-made location cache
is required.

## Boundaries and climate

`world_administration_esri.gpkg` must contain geographic country geometry and the
`ISO_CC` alpha-2 code used by the original national mapping. Its study geometry and
country mapping, rather than a newly substituted boundary edition, determine the
accepted map data.

The five climate directories hold the exact collections named in the supplied URL
lists. Keep original filenames, variables, units, coordinate grids and calendars.
The extraction code recognizes the ISIMIP filename components and validates the
selected scientific coverage. The lists include the years surrounding the paper's
2023 baseline and 2060 representative window; do not arbitrarily trim them.

There is no assertion that an absent file or unavailable cyclone period means
zero exposure. Missing/invalid scientific coverage fails rather than being repaired
by a fabricated zero.
