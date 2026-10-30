# Figure 3 source data

Scope: SSP3-7.0, 2055–2065 (displayed as 2060). These are derived numerical
inputs for reproducing the figure, not a raw-climate/model reconstruction package.

## Files and fields

| File | Fields and meaning |
|---|---|
| `country_annual.csv` | `stage`, `hazard`, `country` (ISO alpha-3 or GLOBAL), `year`, `production_kg`, `loss_kg`. Annual baseline production and climate loss in kg Fe-equivalent. Blank paired totals mean no corresponding country production record. |
| `world_boundaries.parquet` | `ISO_A3`, `geometry` only. GeoParquet with EPSG:4326 metadata, using the accepted dissolved/simplified boundaries. |
| `facility_points.csv` | `point_id` (package-only identifier), `panel`, `longitude`, `latitude` in decimal degrees, EPSG:4326. Exactly the points in the four adopted insets. |
| `facility_annual.csv` | `point_id`, `year`, `loss_fraction`. Dimensionless annual loss/production, not percentage points or absolute production. |
| `facility_global_annual.csv` | `panel`, `year`, `loss_fraction`. Annual global fraction for that panel's stage/hazard, calculated from **all** modelled facilities, not just inset points. |
| `facility_labels.csv` | `point_id`, `display_label`, `x_offset`, `y_offset`, `align`. Seven displayed names and accepted annotation placements; offsets are original-output pixels relative to each inset row. |

Country data contain 22,077 rows: 223 country/global slots × 3 stages × 3 hazards
× 11 years. The global row includes the full upstream country data, not only
countries that can be matched to a displayed boundary.

The facility subset contains 121 points: j = 21, k = 23, l = 70, m = 7. Annual
facility data contain 1,331 rows; the four global series contain 44 rows. No
facility points outside these inset countries/extents are included. Original
database IDs, raw JSON, unlabelled facility names, absolute production, capacities,
production weights and unrelated attributes are not included. The 44 S&P-derived
mine locations are distinct from the 77 GEM-derived iron/steel locations.

## Processing and provenance

Country totals were calculated from the formal country-day hazard outputs using
the original notebook's summation. Annual facility fractions were calculated from
the formal facility-day outputs; the original metric builder checked their
country aggregates against the country-day outputs. Global facility fractions
retain the full model population, even though this export discloses only insets.
Display aliases identify the accepted seven facilities without changing their
source classification. S&P mine classifications are retained, including grouped
mine entries; the export does not redefine them as individual physical mines.

The boundary source is the project's `world_administration_esri.gpkg` (`WOR_Admin`
layer), mapped from alpha-2 to alpha-3 country codes, dissolved by alpha-3, then
simplified at 0.12 degrees with `preserve_topology=False`, exactly as in the
accepted plotting code. The export contains 242 geometry rows, including 42 valid
empty geometries left by that simplification; these are skipped during drawing.
The tolerance is a display simplification, not a distance/area measurement. No
boundary repair, reprojection, alternative basemap or source workbook is bundled.
The specific boundary product edition and redistribution terms could not be
established from the file metadata; they remain to be confirmed before publishing.

The scientific and source definitions follow the manuscript Methods. Source
hashes for the files included here are in `file_hashes.json`; that manifest is
provenance/checking material, not required to execute the plotting script.

## Access and reuse

On 2026-10-29 the author explicitly approved including only the inset mine points
and plotting-required attributes to enable a complete local figure reproduction.
This limited inclusion supersedes the earlier proposed requirement that users
provide external S&P inputs for this figure only. It is **not** a verified
contractual exemption based on record count. No blanket open-data licence is
asserted over S&P, GEM or boundary-source material. Confirm the actual source
agreements/versions and required attribution before public redistribution; this
package has not been uploaded or published by this preparation step.
