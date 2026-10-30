"""Shared geometry for the adopted Fig. 3; no historical raster comparison."""
from pathlib import Path
import json
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.path import Path as MplPath
from matplotlib.patches import PathPatch
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import shapely
from shapely.geometry.polygon import orient
from PIL import Image
from common.figure_style import apply_figure_style as apply_nature_style, panel_label
MUTED_INK = '#666666'
MAP_RISK_COLORS = ['#2166AC','#67A9CF','#D1E5F0','#F7F7F7','#FEE8C8','#FDBB84','#E34A33','#B30000']
DPI = 600
STAGES = ["iron-mine", "iron-factory", "steel-factory"]
HAZARDS = ["heatwave", "tropical-cyclone", "wildfire"]
STAGE_LABELS = {"iron-mine":"Iron mines", "iron-factory":"Ironmaking facilities", "steel-factory":"Steel plants"}
HAZARD_LABELS = {"heatwave":"Heatwave", "tropical-cyclone":"Tropical cyclone", "wildfire":"Wildfire"}
BOUNDS = [0., .5, .8, .95, 1.05, 1.25, 2., 4.]
CASES = [
 {"letter":"j", "name":"Brazil", "country":"BRA", "stage":"iron-mine", "hazard":"heatwave", "extent":[-76,-32,-35,7]},
 {"letter":"k", "name":"Australia", "country":"AUS", "stage":"iron-mine", "hazard":"tropical-cyclone", "extent":[110,156,-46,-9]},
 {"letter":"l", "name":"India", "country":"IND", "stage":"iron-factory", "hazard":"heatwave", "extent":[67,98,5,38]},
 {"letter":"m", "name":"Indonesia", "country":"IDN", "stage":"steel-factory", "hazard":"wildfire", "extent":[94,142,-12,7]},
]
def polygon_path(polygon):
    polygon = orient(polygon, sign=1.0)
    rings = [polygon.exterior, *polygon.interiors]
    paths = []
    for ring in rings:
        coords = np.asarray(ring.coords)
        codes = np.full(len(coords), MplPath.LINETO, dtype=np.uint8)
        codes[0], codes[-1] = MplPath.MOVETO, MplPath.CLOSEPOLY
        paths.append(MplPath(coords, codes))
    return MplPath.make_compound_path(*paths)

def world_paths(geometry_path):
    frame = pd.read_parquet(geometry_path, columns=["ISO_A3", "geometry"])
    records = []
    for row in frame.itertuples(index=False):
        geom = shapely.from_wkb(row.geometry)
        if geom.is_empty:
            continue
        polygons = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
        for polygon in polygons:
            if polygon.geom_type == "Polygon" and not polygon.is_empty:
                records.append((row.ISO_A3, polygon_path(polygon), polygon.bounds))
    return records

def add_countries(ax, records, cmap, norm, values=None, extent=(-180, 180, -60, 85), focus=None):
    x0, x1, y0, y1 = extent
    present, colors, missing = [], [], []
    for code, path, bounds in records:
        a, b, c, d = bounds
        if c < x0 or a > x1 or d < y0 or b > y1:
            continue
        patch = PathPatch(path)
        if values is None:
            present.append(patch)
            colors.append("#E6E9EA" if code == focus else "#F3F4F4")
        elif code in values and np.isfinite(values[code]):
            present.append(patch)
            colors.append(cmap(norm(values[code])))
        else:
            missing.append(patch)
    if missing:
        ax.add_collection(PatchCollection(missing, facecolor="#F4F4F4", edgecolor="#B8BDC1", linewidth=0.10, hatch="////", rasterized=True, zorder=1))
    if present:
        ax.add_collection(PatchCollection(present, facecolor=colors, edgecolor="#A8AEB2", linewidth=0.13 if values is None else 0.07, rasterized=True, zorder=2))
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(values is None)
        spine.set_linewidth(0.4)
        spine.set_color("#A5ABB0")

def original_map_coordinates(geometry_path, base_png):
    """Recreate only the original axes geometry, typography and tight crop.

    The base PNG is newly generated during this run, not a reference image.
    The accepted layout and geographic limits locate the annotation boxes.
    """
    apply_nature_style()
    fig, axes = plt.subplots(3, 3, figsize=(6.15, 3.95), dpi=DPI)
    fig.subplots_adjust(wspace=0.025, hspace=0.08, left=0.06, right=0.995, top=0.92, bottom=0.16)
    geo = json.loads(pq.read_schema(geometry_path).metadata[b"geo"])
    x0, y0, x1, y1 = geo["columns"]["geometry"]["bbox"]
    industry_labels = [STAGE_LABELS[stage] for stage in STAGES]
    for i, stage in enumerate(STAGES):
        for j, hazard in enumerate(HAZARDS):
            ax = axes[i, j]
            ax.update_datalim(np.array([[x0, y0], [x1, y1]]))
            ax.autoscale_view()
            ax.set_axis_off()
            ax.set_aspect("equal")
            panel_label(ax, i * 3 + j, x=0.014, y=0.88)
            if j == 0:
                ax.text(-0.035, 0.5, industry_labels[i], transform=ax.transAxes, va="center", ha="right", fontsize=6.5, fontweight="normal", rotation=90)
            if i == 0:
                ax.set_title(HAZARD_LABELS[hazard], pad=2, fontweight="normal")
    cmap = ListedColormap(MAP_RISK_COLORS)
    norm = BoundaryNorm(BOUNDS, cmap.N, extend="max")
    cax = fig.add_axes([0.20, 0.075, 0.60, 0.024])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal", extend="max")
    cb.outline.set_linewidth(0.4)
    cb.set_label("Relative hazard impact (country / global)", labelpad=3)
    cb.set_ticks(BOUNDS)
    cb.ax.set_xticklabels(["0", "0.5", "0.8", "0.95", "1.05", "1.25", "2", "4+"])
    cb.ax.tick_params(length=1.8, width=0.4, pad=1.5)
    fig.text(0.20, 0.115, "Lower relative hazard impact", fontsize=6.2, color=MUTED_INK, ha="left", va="bottom")
    fig.text(0.80, 0.115, "Higher relative hazard impact", fontsize=6.2, color=MUTED_INK, ha="right", va="bottom")
    fig.canvas.draw()
    bbox = fig.get_tightbbox(fig.canvas.get_renderer()).padded(plt.rcParams["savefig.pad_inches"])
    expected_size = (int(bbox.width * DPI), int(bbox.height * DPI))
    with Image.open(base_png) as original:
        assert original.size == expected_size, (original.size, expected_size)
    result = {}
    for case in CASES:
        ax = axes[STAGES.index(case["stage"]), HAZARDS.index(case["hazard"])]
        a, b, c, d = case["extent"]
        points = ax.transData.transform([[a, c], [b, d]])
        points[:, 0] -= bbox.x0 * DPI
        points[:, 1] -= bbox.y0 * DPI
        result[case["letter"]] = points
    plt.close(fig)
    return result
