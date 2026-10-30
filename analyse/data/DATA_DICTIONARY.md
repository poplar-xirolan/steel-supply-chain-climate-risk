# Included data dictionary

The file-level inventory is `manifest.csv`. Input roles are explicit:
**provided**, **external original source**, **reference** or **model definition**.
Generated output belongs only to `work/` and `result/`, never to `data/provided/`.

| Area | Identity / measures | Interpretation |
|---|---|---|
| `provided/mfa/2023.parquet` | `entity_from/to`, `sector_from/to`, `mass`, `unit`, `extra` | Physical Fe-equivalent kg flows; `extra` retains process-type distinctions |
| `provided/country_hazards/*.parquet` | `country_a3`, `day_of_year`, `daily_fe_eq_weight_kg_scaled`, `production_loss_fraction`, `production_loss_kg_scaled` | Daily national stage production and loss; filename gives stage, hazard, year, scenario |
| `provided/coverage/` | scenario/period/source availability | Scientific missing-coverage policy, not an old run authorization |
| `provided/abm_summaries/full_annual_climate_inventory.csv` | year, annual mass, covered days, daily mean, loss %, scenario and parameter columns | Actual scenario results, calendar-normalized loss relative to 2023 |
| `provided/abm_summaries/ssp370_inventory_step_series.csv` | step dates, days, mass, loss %, inventory setting | Weekly intervals; partial/cross-year intervals are retained |
| `provided/abm_summaries/*country*.csv` | country/stage/period quantities and fulfilment | Distinguish stage production from final-demand satisfaction |
| `provided/abm_summaries/*endpoints.csv` | scenario, parameters, 2023/2100 quantities | Baseline and endpoint both retained; random restrictions retain replicate seed |
| `provided/figure_metrics/production_components_annual.csv` | country, sector, actual output, throughput, outbound delivery, year | Different physical quantities; Figure 6 uses actual output |
| `provided/figure_metrics/capacity_sensitivity_annual.csv` | capacity factor × year, annual mass, days and loss % | 6 capacity settings × 78 years; Supplementary Figure 2 |
| `provided/dscci_statistics/dscci/` | Country, Stage, capacity, demand, adequacy and DSCCI | Reference data; seven stages and existing entropy/product formula |
| `provided/dscci_statistics/statistics/` | estimates, intervals, P values, leave-one-out country | Reference statistical outputs, not treated as model inputs |
| `provided/plot_auxiliary/Figure_3/` | point IDs/coordinates, annual derived fractions, labels, geometry | Bounded previously approved plot subset; IDs are local to the supplied plot data |
| `definitions/` | country and trade category mappings, end-use shares | Model definitions; individual numerical coefficients also remain in the scientific scripts |
| `usgs/` | columns as recorded in the original curated worksheets | Four study-source tables; exact names recorded in the manifest |

Stages: `iron-mine` = mining, `iron-factory` = ironmaking, `steel-factory` =
steelmaking. Scenario codes `picontrol`, `ssp126`, `ssp370`, `ssp585` are preserved
from the input products and scientific code. `multi-hazard-max` is the actual
coupled loss input, not the sum of the three single-hazard rates.

MFA model codes used in the pipeline:

| Code | Production stage |
|---|---|
| P1 | Mining |
| P2 | Ironmaking |
| P3 | Steelmaking |
| P4 | Casting |
| P5 | Forming |
| P6 | Manufacturing |
| P7 | Old-scrap supply/recovery |

Market/stock nodes M1–M9 and S1 retain the exact network/engine definitions;
S1 is the final-demand destination. Refer to the authored system definition in
Supplementary Figure 1 and the source code for the detailed market/flow topology.
Do not reinterpret P4 as an independent steel-production capacity when reading
the merged display profile in Figure 6d.

`E2ECI`/`e2eci` in legacy column/file names denotes the manuscript's **DSCCI**.
`source_run_id` is descriptive provenance only. Private source directories,
execution contracts, reference artwork or source-code hashes are never used to
substitute expected results for new calculations.
