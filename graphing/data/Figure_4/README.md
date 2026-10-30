# Figure 4 source data

## Files

- `country_seasonal_losses.csv`: the complete numerical input used by the script.
- `source_files.csv`: provenance for the 99 original country-day files, including
  their basenames, SHA-256 hashes and coverage counts. This table is documentation,
  not an execution dependency; the original files are not needed to draw the figure.

The numerical CSV contains **9,900 rows**, covering SSP3-7.0, 2055–2065, three
production stages, three single hazards and two calendar halves per country/year.
There are 27 mining, 46 ironmaking and 77 steelmaking countries in each corresponding
stage/hazard/year. Each has both half-year records, including genuine zero losses.

| Field | Meaning |
|---|---|
| `stage` | `iron-mine`, `iron-factory` or `steel-factory`; displayed as Iron mines, Ironmaking facilities and Steel plants. |
| `hazard` | `heatwave`, `tropical-cyclone` or `wildfire`. These are separate hazards, not the multi-hazard maximum combination. |
| `scenario` | `ssp370` (SSP3-7.0). |
| `year` | Calendar year, 2055–2065 inclusive. |
| `country` | Country alpha-3 code; all countries in the relevant production-stage data, not only the five displayed. |
| `season` | `Apr-Sep` or `Oct-Mar`, defined below. |
| `loss_kg` | Absolute production loss summed within the half-year, in kg Fe-equivalent. Not a fraction or percentage. |
| `days_covered` | Number of country-day records included in the half-year sum. |

Apr–Sep always contains 183 days. Oct–Mar contains 182 days in a common year and
183 in a leap year (2056, 2060, 2064). The latter combines January–March and
October–December **within the same calendar year**. No hemisphere-specific season
assignment is made. Missing observations are not encoded as zero.

## Extraction and provenance

The source is the project's formal `hazard_country_level_loss` outputs. Each
country-day `production_loss_kg_scaled` value was assigned to a calendar half
using its year and day of year, then summed by country and half-year using the
same pandas grouping as the adopted Figure 4 notebook. All 99 inputs had complete
country-day coverage and unique country/day keys. There were 1,808,100 source
country-day records before this lossless-for-the-figure seasonal aggregation.

The plotting script calculates eleven-year means, global denominators, top-five
selection, remainder and percentages from these seasonal sums. It does not need
absolute production, facility IDs/coordinates, original third-party facility
attributes, model-grid files, ABM outputs, maps, or the MFA workbook. The data
are derived country-level model results, not observed production losses.

Units and upstream scientific assumptions follow the manuscript Methods. The
supplied data are sufficient to redraw this figure; reconstruction of the original
daily hazard/model outputs belongs to the separate analysis workflow.
