# Supplementary Figure 2 source data

`capacity_sensitivity_annual.csv` is the complete derived global ABM annual
table used for this figure, not an observational data series. It is copied
byte-for-byte from the formal capacity-sensitivity annual product: 468 records,
six capacity settings, 78 years per setting. No fields or rows were removed.
Its SHA-256 is
`7c48b364dccdf1dd5e3dd90adf80d9654889f248f8ceb66de908209ee24acf5e`.

## Fixed setting and parameter meaning

All records use multi-hazard SSP3-7.0, unrestricted international matching,
a four-week inventory target, a seven-day simulation step, the fixed 2023
industrial structure, and the complete 2023–2100 simulation window.
The varied capacity-to-baseline-flow ratio κ is 1.0, 1.1, 1.2, 1.3, 1.4 or 1.5
for expandable production stages P1–P6; κ = 1.1 is the main baseline setting.
The end-of-life scrap-supply stage P7 remains at 1.0. Because inventory targets
scale with inherent capacity, their absolute mass changes when κ changes even
though the four-week inventory-to-capacity ratio is fixed.

## Fields

| Field | Meaning and unit |
|---|---|
| year | Calendar year, 2023–2100 inclusive. |
| annual_mass | Annual simulated final-demand delivery to S1, kg Fe-equivalent. Upstream postprocessing allocates cross-year simulation-step masses in proportion to calendar overlap days. |
| coverage_days | Covered calendar days, 365 or 366. Leap years are not treated as 365 days. |
| mean_daily_mass | annual_mass / coverage_days, kg Fe-equivalent per day. |
| loss_percent | 100 × (1 − mean_daily_mass / the same capacity setting's mean_daily_mass in 2023), in percent (%), not a fraction. |
| capacity | Varied model capacity multiplier κ; dimensionless. |

The CSV preserves all original numeric field strings, including floating-point
residues of order 1e-14 near zero. The plotting loader retains the adopted
pandas numeric parsing under the pinned environment, checks the mass/rate/loss
relationship and keeps all six trajectories. No missing value is replaced by
zero. Zero-overlap annotation and the plot's zero lower bound do not modify the
underlying values.

## Provenance and scope

`source_files.csv` records source basenames, SHA-256 hashes and roles. The
annual table's original provenance receipt and six run-metadata hashes were
checked during preparation. Its numerical trajectories also agree with the
adopted plotting cache within floating-point precision, and the κ = 1.1 series
agrees with Figure 5's SSP3-7.0/four-week series.

The large original per-step hash receipt and plotting cache are not supplied
or needed at runtime. Unchanged adopted PNG/PDF files are in the plotting
folder's `reference/`, for comparison only. The script reads only the annual
numerical CSV, not those reference images or this provenance listing.
The preparation checks did not rerun the ABM or rebuild the annual table from
raw transaction files; that belongs to the separate analysis workflow.
