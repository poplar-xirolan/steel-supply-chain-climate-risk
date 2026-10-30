# Figure 6 — domestic capability, production responses and trade restrictions

This folder redraws the accepted five-panel v7 figure from six bundled numerical
tables. It requires no original project package, legacy plotting module,
simulation transactions, full MFA network, workbook, old cache or network access.
It does not read reference images or precomputed statistical summaries.

## Install and run

Tested in the existing Windows/Python 3.13.5 environment. From reproduce_export:

```text
python -m pip install -r graphing/plotting/Figure_6/requirements.txt
python graphing/plotting/Figure_6/plot_figure6.py
```

Keep plot_figure6.py and figure6_statistics.py together. Defaults resolve relative
to the script, not the current directory. Optional --data-dir PATH and
--output-dir PATH override them. The runtime dependencies are NumPy, pandas,
Matplotlib and SciPy; no SciencePlots, openpyxl, PyArrow or LaTeX is required.
The exact inherited line styles of the adopted renderer are set explicitly,
without importing the old module that used to set them as a side effect.

Arial is needed for reference typography, is not bundled, and its absence emits
a warning before font fallback. The existing pinned environment was tested; a
fresh installation on another machine has not been tested.

## Numerical inputs

All six CSVs are in ../../data/Figure_6/:

| File | Coverage |
|---|---|
| country_capability.csv | 168 economies: original DSCCI, final-demand outcome and baseline MFA demand. |
| random_trade_endpoints.csv | 190 realizations at 19 exclusion levels, each with 2023 and 2100 endpoints: 380 rows. |
| global_trade_endpoints.csv | Four named trade scenarios, each with both endpoint years: 8 rows. |
| country_trade_endpoints.csv | All 670 positive-baseline national endpoints across the named scenarios. |
| production_components.csv | Eight countries, six original production components, 2023 and 2100: 92 actual-output rows. |
| baseline_capacity_components.csv | Eight countries, six original components: 48 capacity/demand records. |

Original numeric and date-free identifier fields are copied as text unchanged
during extraction. The plotting loader retains the original pandas numeric
parsing semantics under the pinned environment. It checks country/replicate/
scenario coverage, duplicates, ratios, capacities and baselines before plotting.
Unknown missing values are rejected, not filled with zero. See the data README
for the explicitly preserved sparse zero-delivery records in the original table.

## Calculations retained from v7

- a: all 168 pairs; existing baseline seven-stage DSCCI is an input, not
  redefined using d's five display groups. The OLS line and 95% mean-fit band
  are recalculated. Pearson r, its conventional P value and paired-country
  bootstrap interval are recalculated, as are tied-rank Spearman rho and its
  two-sided pairing-permutation P value. The original 20,000 bootstrap samples,
  99,999 permutations, seed 20260913 and Spearman seed offset +1 are retained.
  The bootstrap correlation interval is not the OLS mean-fit band. Statistical
  resampling uses deterministic ISO order, retaining endpoint ties at 0/100.
- b: all 10 realizations per exclusion level are used to recalculate mean,
  sample SD and the two-sided 95% Student-t confidence interval of the mean.
  These are not interpolated model runs or country-bootstrap intervals.
- c: the four global losses are checked against each scenario's own positive
  2023 baseline and summed national endpoints. Global decoupling retains its
  different baseline and asterisk; it is not renormalized to free trade.
- d: ratios use actual main-product output, not replenishment receipts or used
  throughput. P3 and P4 are summed in mass units before forming the combined
  steelmaking/casting ratio. Capacity components are also summed; the domestic
  2023 MFA final-demand denominator is counted once, not twice. All 40 mini-bars
  retain the common 0–1x scale and arrows strictly above 1x. Germany/UK mining
  capacity is zero, while the undefined output ratios stay N/A.
- e: the same eight countries and DSCCI values are retained. Free-trade and
  group-restriction outcomes are paired only after matching their baseline
  denominators. All endpoint labels are derived from the data.

These are conditional associations within one simulated setting. Country
resampling does not model trade-network dependence, climate/ABM uncertainty or
causality. The helper rebuilds the two correlations used in this figure, not
every additional robustness analysis in the manuscript's statistical supplement.
Complete DSCCI construction and upstream model runs belong to analyse.

## Outputs

output/ contains Figure_6.png (180 × 180 mm, 600 dpi, 4251 × 4251 pixels),
Figure_6_preview.png (240 dpi), Figure_6.pdf and Figure_6.svg. It also contains:

- source_scatter.csv, source_ols_band.csv and Figure_6_statistics.json;
- source_random_restrictions.csv and source_named_scenarios.csv;
- source_country_comparison.csv;
- source_production_components_annual.csv and source_stage_fulfilment.csv;
- source_baseline_capacity_components.csv and source_baseline_capacity.csv.

Rerunning replaces these outputs, not inputs or reference assets. reference/
contains unchanged accepted PNG/PDF/SVG files for comparison only. Preparation,
tests and temporary verification logs are outside the export package and are
not needed by either script. This is plotting-level reproduction from derived
results, not a climate-model or ABM rerun.
