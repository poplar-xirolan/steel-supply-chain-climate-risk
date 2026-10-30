# Figure 5 source data

The two numerical CSVs are derived global ABM results, not observed losses.
They are sufficient to redraw Figure 5 without raw ABM transaction files.
source_files.csv records source basenames, SHA-256 hashes and their roles. The
original selected_metrics.json was used only as an independent verification
source and is not supplied or required at runtime.

## annual_summary.csv — 858 rows

Coverage: 2023–2100 inclusive, 78 years for each of 11 settings. The settings
are picontrol, ssp126, ssp370 and ssp585 with four-week inventory, plus ssp370
with 1–3- and 5–8-week inventory. Every setting uses free-trade and capacity 1.1.

| Field | Definition |
|---|---|
| year | Calendar year, including leap-year coverage. |
| annual_mass | Global final-demand delivery summed over the calendar year, kg Fe-equivalent. Actual step masses are allocated across year boundaries in proportion to overlap days by upstream postprocessing. |
| coverage_days | Number of covered days: 365 or 366, not assumed uniformly 365. |
| loss_percent | 100 × (1 − annual_mass / coverage_days / corresponding 2023 daily delivery). Units are percent, not a fractional rate. |
| climate | picontrol (pre-industrial), ssp126 (SSP1-2.6), ssp370 (SSP3-7.0), or ssp585 (SSP5-8.5). |
| trade | free-trade. |
| inventory | Inventory setting in weeks, 1–8. |
| capacity | Model capacity multiplier, 1.1. |

The 2023 baseline is matched within each climate/inventory setting, not taken
from another setting. The CSV preserves unrounded source values, including tiny
floating-point residues around zero. Original numeric/date field strings are
copied unchanged; the plotting loader retains the adopted pandas float parsing
under the pinned environment. The plotting script applies the adopted
absolute 1e-10-percentage-point annual roundoff normalization, after checking
the loss/mass/coverage relationship. It does not delete isolated positive years.

## within_year_2100.csv — 424 rows

Coverage: SSP3-7.0, free trade, capacity multiplier 1.1, inventories 1–8 weeks.
There are 53 actual intervals for each inventory. The first interval starts
2099-12-27 and ends 2100-01-03; only two days contribute to the 2100 display.
The final interval is 2100-12-26 to 2101-01-01, lasting six days. End dates
are exclusive. Together the clipped intervals cover all 365 days of 2100.

| Field | Definition |
|---|---|
| step | Original simulation step identifier; not a day-of-year value. |
| mass | Actual global final-demand delivery for the whole original interval, kg Fe-equivalent; the first mass is not pre-clipped to its two-day overlap. |
| start_date | Inclusive original interval start. |
| end_date_exclusive | Exclusive original interval end. |
| step_days | Actual original interval duration, seven days except the final six-day step. |
| loss_percent | 100 × (1 − (mass × 7 / step_days) / (corresponding 2023 daily delivery × 7)). This is the daily-normalized interval loss, not an annual loss. |
| climate, trade, inventory, capacity | Same definitions as above. |

The plot uses unchanged interval loss ratios over each interval's clipped
duration. Its annual mean is weighted by overlap days; averaging 53 ratios
equally would incorrectly weight the partial boundary intervals. No interpolation,
smoothing or resampling to an artificial 52-week calendar is used.

## Extraction

annual_summary.csv selects only the eight relevant fields from the formal
full_annual_climate_inventory.csv. within_year_2100.csv selects only the ten
relevant fields and intervals intersecting 2100 from the formal
ssp370_inventory_step_series.csv. Unrelated scenario ordinals, random fields,
and precomputed midpoint/normalization fields were removed. No rows needed by
the figure were removed, and no records were inferred from missing data.

The adopted v3 provenance hashes matched these formal source files before
extraction. Source model assumptions and scenario definitions remain those of
the manuscript Methods. Reconstruction of those formal ABM summaries belongs
to the separate analysis workflow, not this plotting-only package.
