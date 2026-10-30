# Figure 1 - methodological framework

`Figure_1.drawio` is the editable source for the accepted methodological
schematic. The matching PNG, PDF and SVG exports are preserved in `reference/`.
These files are provided unchanged; this folder contains no export script,
bundled application, Python environment or historical revision scripts.

## Requirements

Install **draw.io Desktop** separately from the [official draw.io website](https://www.drawio.com/).
The source and reference exports were prepared with **version 28.1.2**, which
was also used for the local export check. Ensure that **Arial** is available.
Other application versions or font substitutions may produce rendering differences.

No Python packages, model runs or external data files are needed to open and
export this source. Its graphical content is embedded. Most elements are
editable vector shapes; the country-level trade inset is an embedded PNG, so
the complete figure should not be described as entirely vector artwork.

## Open and export manually

1. Open `Figure_1.drawio` in draw.io Desktop.
2. Select the first page, **Fig1 · inputs, simulation and security analyses**.
   The source retains a blank second page; do not export all pages.
3. Use **File > Export as** and select PNG, SVG or PDF. Export the complete
   current page, not a selection. Use a light appearance, a white background,
   no grid and a border width of **20**.
4. For PNG, use **100%** zoom. The reference image is **2308 x 2154 pixels**.
   For SVG, embed images and include a copy of the diagram; the reference does
   not embed fonts. For PDF, select **Current Page** and **Crop** to the diagram,
   rather than exporting the full paper page or all pages.
5. Save new exports in `output/`, which is initially empty, or another folder
   of your choice. Keep `reference/` unchanged for comparison.

Export option names can vary with the application version. Official guidance:
[PNG](https://www.drawio.com/docs/manual/export/export-to-png/),
[SVG](https://www.drawio.com/docs/manual/export/export-to-svg/) and
[PDF](https://www.drawio.com/docs/manual/export/export-to-pdf/).

## Scope

Re-exporting this source reproduces the assembled illustration. It does **not**
recompute the underlying material flows, facility distribution or climate-risk
results shown in its insets. Consequently, this folder has no separate
`graphing/data/Figure_1/` input directory and does not supply a raw facility
coordinate database. Numerical reconstruction belongs to the analysis workflow.

If revising the illustration, retain the original reference exports and check
the new exports for changed text, missing elements, clipping and font changes.
