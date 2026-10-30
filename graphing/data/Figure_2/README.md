# Source data for Figure 2

These are model-derived figure inputs, not raw observations. They were
assembled from the formal country-level, single-hazard results used by the
accepted analysis. No facility coordinates, proprietary facility records or
raw climate-model arrays are included.

## annual_global_losses.csv

One row per production stage, hazard, scenario and year, for 2018-2100:
**2,988 positions, 2,859 available and 129 explicitly unavailable**.

| Field | Meaning |
|---|---|
| `industry` | `iron-mine`, `iron-factory` or `steel-factory`; displayed as Iron mines, Ironmaking facilities and Steel plants. |
| `hazard` | `heatwave`, `tropical-cyclone` or `wildfire`; these are separate hazards, not their combined maximum. |
| `scenario` | `picontrol`, `ssp126`, `ssp370` or `ssp585`. |
| `year` | Calendar year of the formal annual input file. |
| `baseline_production_fe_kg` | Sum of `daily_fe_eq_weight_kg_scaled` over all country-day rows in that formal file, in kg Fe-equivalent. This is the baseline denominator, not realized production after climate losses. |
| `production_loss_fe_kg` | Sum of `production_loss_kg_scaled` over those rows, in kg Fe-equivalent. |
| `annual_loss_fraction` | Loss sum divided by baseline-production sum; dimensionless fraction, not percent. |
| `availability` | `available` or `unavailable`. |
| `source_file` | Basename of the formal yearly country-level Parquet used for this row. No local absolute path is included. |
| `source_sha256` | SHA-256 of that source file at release assembly. |

Rows marked unavailable retain blank numeric and provenance fields. All 129
correspond to SSP1-2.6 tropical cyclones in 2018-2060 across the three stages.
They represent absent source coverage, not zero physical risk.

Aggregation uses the original pandas summation and original global
`compute_loss_percent` function, without changing the daily values or annual
weighting. The upstream file convention is
`{industry}_{hazard}_{year}_{scenario}.parquet` in the formal
`results/hazard_country_level_loss/` output. A full analysis rebuild must
produce those scientific results before assembling equivalent global annual
inputs. This plotting package uses the supplied annual CSV directly and never
reads assembly helpers, audit logs or the original project directory.

Numbers are serialized with 17 significant digits. The plotting script reads
with round-trip floating-point precision so exporting/importing the CSV does
not introduce avoidable rounding changes.

## reference_plot_values.csv

The 108 stage-hazard-target-year-scenario positions calculated with the original
notebook's period-statistic and normalization code. This file is a comparison
reference only: the standalone script does not read it when drawing.

| Field | Meaning |
|---|---|
| `industry`, `hazard`, `scenario` | Same categories as the annual input. |
| `target_year` | 2035, 2060 or 2100. |
| `mean_risk_ratio` | Period-mean annual loss fraction divided by its stage-hazard 2023 baseline mean. |
| `temporal_95ci_half_width` | Half-width of the temporal 95% t interval, divided by the same baseline mean. |
| `baseline_scenario` | `ssp370`. |
| `baseline_center_year` | 2023. |
| `requested_window_years` | 11 (center year +/- 5). |
| `available_years` | Actual annual sample count: 0, 5, 6 or 11 in this release. |

The three zero-count positions have blank mean and interval fields. The 2060
SSP1-2.6 cyclone estimates use five years; all 2100 estimates use six years.
The intervals quantify variation across available annual fractions, not
between climate models. Further calculation details are in the plotting README.
