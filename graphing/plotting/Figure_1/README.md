# Figure 1 - localized climate risks and supply-chain propagation

`Figure_1.drawio` is the editable source for the final mechanism schematic
accepted on 3 November 2026, including the corrected supply-shortfall icon.
The matching PNG, PDF and SVG exports are preserved in `reference/`.
These files are provided unchanged; this folder contains no export script,
bundled application, Python environment or historical revision scripts.

## Requirements

Install **draw.io Desktop** separately from the [official draw.io website](https://www.drawio.com/).
The source and reference exports were prepared with **version 28.1.2**, which
was also used for the local export check. Ensure that **Arial** is available.
Other application versions or font substitutions may produce rendering differences.

No Python packages, model runs or external data files are needed to open and
export this source. Its graphical content is embedded as editable vector
shapes; the source and matching SVG/PDF contain no raster images. The country
and facility maps are part of the assembled drawing, not linked image files.

## Open and export manually

1. Open `Figure_1.drawio` in draw.io Desktop.
2. Select the single page, **Fig. 1 | Facility exposure, upstream constraints
   and final supply**. There is no blank second page.
3. Use **File > Export as** and select PNG, SVG or PDF. Export the complete
   current page, not a selection. Use a light appearance, a white background,
   no grid and a border width of **8**.
4. For PNG, use **128%** zoom. The reference image is **2210 x 2567 pixels**.
   For SVG, embed images and include a copy of the diagram; the reference does
   not embed fonts. For PDF, select **Current Page** and **Crop** to the diagram,
   rather than exporting the full paper page or all pages.
5. Save new exports in `output/`, which is initially empty, or another folder
   of your choice. Keep `reference/` unchanged for comparison.

The reference PDF is one page, uniformly scaled after cropped export to a width
of **180 mm**, without changing its artwork or proportions. A direct cropped
PDF export may therefore have a different physical page size. Set its placement
width to 180 mm when using that publication size; a PDF byte-for-byte match is
not expected because export metadata can differ.

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

The left-hand pathway illustrates how localized disruption can propagate to
final-product shortages. Country-stage constraints feed the upstream stages
of the 2023 material-flow network, followed by weekly simulation and the
analysis of national steel security. Facility-level loss profiles are schematic;
the figure is not an additional model experiment.

If revising the illustration, retain the original reference exports and check
the new exports for changed text, missing elements, clipping and font changes.
