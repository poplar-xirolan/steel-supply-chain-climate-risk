# Figure 6 numerical inputs and provenance

These are derived MFA/ABM results, not observed losses or original commercial
facility records. source_files.csv lists original project-relative sources,
hashes and verification roles. None of those original files or paths is a
runtime dependency; the two scripts read only the six CSVs in this folder.
Only relevant columns/rows were selected. All retained numeric field strings
were copied unchanged rather than parsed and re-serialized during extraction.

## country_capability.csv — 168 rows

| Field | Meaning |
|---|---|
| country | Economy alpha-3 code. |
| dscci | Existing seven-stage domestic supply chain capability index, 0–100. The source's legacy column name was E2ECI. |
| final_demand_satisfaction | Simulated 2100 final-demand delivery relative to corresponding 2023 delivery, percent. |
| final_demand_loss | 100 minus satisfaction, percentage points on the same 0–100 scale. |
| mfa_final_demand_2023 | Domestic 2023 MFA M8-to-S1 annual final demand, kg Fe-equivalent; also used to check d's capacity denominator. |

Scenario: multi-hazard-max, SSP3-7.0, free trade, inventory four weeks, model
capacity factor 1.1. The original tiny satisfaction residuals within 1e-8
percentage points of 0/100 are tied to those endpoints at runtime, matching the
adopted statistics; ordinary observations and CSV fields are not rounded.

## random_trade_endpoints.csv — 380 rows

19 levels (0.05, 0.10, ..., 0.95) with 10 seeds (2026081100–2026081109) each,
and two endpoint years per realization. Fields:

- year: 2023 or 2100; coverage_days: 365 in both years.
- annual_mass: actual global annual final-demand delivery, kg Fe-equivalent.
- loss_percent: 100 × (1 − (annual_mass / coverage_days) / own 2023 daily delivery).
- climate: ssp370; trade: random-disruption; inventory: 4 weeks; capacity: 1.1.
- random_ratio: fraction of potential foreign partners excluded, not a percent.
- random_seed: realization identifier; not a country-resampling seed.

The baseline is matched to each realization. The script uses all 190 runs to
recalculate 19 means and sample-SD/Student-t intervals. Recalculation from CSV
values may differ from the old serialized mean/CI table at floating-point
roundoff (approximately 1e-14 percentage points); inputs are not changed to force
old summary-table bytes.

## global_trade_endpoints.csv — 8 rows

Same endpoint mass/day/loss fields and fixed SSP3-7.0/inventory/capacity settings
as above, without random fields. The four trade values are free-trade,
us-cn-decoupling, bloc-confrontation and global-decoupling. Both endpoint years
are retained. Each scenario uses its own 2023 daily baseline; global decoupling
does not share the other scenarios' denominator.

## country_trade_endpoints.csv — 670 rows

Fields: country, 2023, 2100, fulfilment_percent, loss_percent, trade. The two
year-named fields are annual final-demand delivery in kg Fe-equivalent. The
recorded fulfilment is 100 × 2100 / 2023, with both years covering 365 days.
Loss is 100 minus fulfilment. Free trade, US–China decoupling and group
restriction each retain all 168 economies. Global decoupling retains the source's
166 positive-baseline records; LTU and MKD are absent, not inserted as zero rows.

63 global-decoupling rows have an original blank 2100 mass, while the source
explicitly records 0% fulfilment and 100% loss. The upstream conserving-metrics
builder groups existing S1 flows; a country with no future flow has no future
mass row, and add_fulfilment explicitly derives its recorded zero fulfilment.
These blanks are preserved verbatim, not replaced with observed or inferred
mass values. Only the exact original sparse locations/outcomes are allowed.
Any new missing location is rejected. Panel c uses the finite global endpoint
table, which is reconciled with the sum of recorded national masses; e's eight
selected countries have finite masses in both compared scenarios.

## production_components.csv — 92 rows

Fields: country, sector (P1–P6), actual_output, year (2023 or 2100). Actual
output is main-product output in kg Fe-equivalent, allocated by actual calendar
overlap. It is not replenishment receipts, outbound shipments of all commodities
or realized throughput before product yields. The source compact table was
previously reconciled with engine states and primary-product flows.

P1 is mining; P2 ironmaking; P3 steelmaking; P4 casting; P5 forming; P6
manufacturing. The eight countries are CHN, USA, IND, BRA, AUS, CAN, DEU and GBR.
The two structurally absent mining components (DEU/GBR P1) have no records in
either year. Their absence is allowed only because the matching model capacity
is zero. Missing positive-capacity output is an error.

The figure combines P3 and P4 before division: 100 × (P3 output in 2100 + P4
output in 2100) / (P3 output in 2023 + P4 output in 2023). It never averages their
two percentages. There are 40 displayed cells, of which 38 have defined ratios.

## baseline_capacity_components.csv — 48 rows

Fields: country, sector, total_inflow, total_outflow, baseline_throughput,
capacity_factor, modeled_capacity, final_demand. Mass/throughput/capacity/demand
values are annual kg Fe-equivalent. baseline_throughput is max(total_inflow,
total_outflow); modeled_capacity is 1.1 times that throughput. final_demand is
the country's domestic annual 2023 MFA M8-to-S1 final demand, not the slightly
calendar-scaled ABM baseline delivery.

All 48 component records were reconciled with the canonical 2023 MFA during
export preparation. Runtime checks the capacity formula and matches denominators
to country_capability.csv. For the combined P3/P4 cell, the two capacities are
added and divided by ONE final-demand denominator. Forty bars are drawn on a
common 0–1x scale; 17 have an overflow arrow. These bars are not DSCCI adequacy
scores, and 1x does not imply stage-specific self-sufficiency. Five display groups
do not redefine the seven-stage index.
