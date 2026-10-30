"""Accepted Figure 3 composition using this run's freshly drawn base grid."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
from PIL import Image

import map_geometry as draft
old = draft
from common.figure_style import apply_figure_style as apply_nature_style

DPI = 600
CASES = draft.CASES


def draw(sites, labels, geometry_path, base_png, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = output_dir / 'Figure_3'
    positions = old.original_map_coordinates(geometry_path, base_png)
    original = np.asarray(Image.open(base_png).convert("RGBA"))
    height, old_width = original.shape[:2]
    gap, sidebar_width = 80, 870
    sidebar_x = old_width + gap
    width = sidebar_x + sidebar_width
    apply_nature_style()
    plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42})
    fig = plt.figure(figsize=(width / DPI, height / DPI), dpi=DPI)
    fig.figimage(original, xo=0, yo=0, origin="upper", zorder=0)
    fig.add_artist(Rectangle((sidebar_x/width, 0), sidebar_width/width, 1,
                             transform=fig.transFigure, facecolor="#F1F1F1", edgecolor="none", zorder=-1))
    overlay = fig.add_axes([0, 0, old_width/width, 1], zorder=2)
    overlay.patch.set_visible(False)
    overlay.set(xlim=(0, old_width), ylim=(0, height))
    overlay.set_axis_off()
    allowed = np.zeros((height, old_width), dtype=bool)
    for case in CASES:
        (left, bottom), (right, top) = positions[case["letter"]]
        overlay.add_patch(Rectangle((left, bottom), right-left, top-bottom,
                                   fill=False, edgecolor="black", linewidth=.65))
        overlay.text(left+8, top-8, case["letter"], ha="left", va="top", fontsize=5.9,
                     fontweight="bold", color="black", bbox={"facecolor":"white", "edgecolor":"none", "alpha":.9, "pad":.25})
        allowed[max(0,int(height-top)-10):min(height,int(np.ceil(height-bottom))+10),
                max(0,int(left)-10):min(old_width,int(np.ceil(right))+10)] = True

    cmap = ListedColormap(draft.MAP_RISK_COLORS)
    norm = BoundaryNorm(draft.BOUNDS, cmap.N, extend="max")
    geometry = draft.world_paths(geometry_path)
    annotations = []
    for index, case in enumerate(CASES):
        row_top = 20 + index*475
        fig.text((sidebar_x+42)/width, 1-(row_top+18)/height,
                 f"{case['letter']}  {case['name']}", ha="left", va="top", fontsize=6.4, fontweight="bold")
        stage_name = {"iron-mine":"Mining", "iron-factory":"Ironmaking", "steel-factory":"Steelmaking"}[case["stage"]]
        fig.text((sidebar_x+42)/width, 1-(row_top+79)/height,
                 f"{stage_name} · {draft.HAZARD_LABELS[case['hazard']]}",
                 ha="left", va="top", fontsize=5.1, color="#555555")
        plot_top = row_top+127
        # Make room below the wide Indonesian inset for two complete names.
        # The extent, points, colours and marker sizes remain unchanged.
        plot_height = 260 if case['letter'] == 'm' else 320
        ax = fig.add_axes([(sidebar_x+45)/width, (height-plot_top-plot_height)/height,
                           (sidebar_width-90)/width, plot_height/height], zorder=3)
        draft.add_countries(ax, geometry, cmap, norm, extent=case["extent"], focus=case["country"])
        ax.set_facecolor("#F1F1F1")
        ax.set_anchor({'j':(.84, .5), 'k':(.96, .5)}.get(case['letter'], "C"))
        chosen = sites[sites.panel.eq(case['letter'])].copy()
        a,b,c,d = case["extent"]
        assert chosen.longitude.between(a,b).all() and chosen.latitude.between(c,d).all()
        assert chosen.point_id.is_unique and chosen.facility_to_global_ratio.notna().all()
        assert chosen.available_years.eq(11).all()
        ax.scatter(chosen.longitude, chosen.latitude, c=chosen.facility_to_global_ratio,
                   cmap=cmap, norm=norm, s=6.5, linewidths=.22, edgecolors="white", zorder=4)
        label_specs = labels[labels.point_id.isin(chosen.point_id)]
        for spec in label_specs.itertuples(index=False):
            label, xoffset, yoffset, ha = spec.display_label, spec.x_offset, spec.y_offset, spec.align
            site = chosen.loc[chosen.point_id.eq(spec.point_id)].squeeze()
            assert isinstance(site, pd.Series)
            annotation = ax.annotate(label, xy=(site.longitude, site.latitude), xycoords="data",
                xytext=((sidebar_x+xoffset)/width, 1-(row_top+yoffset)/height), textcoords=fig.transFigure,
                ha=ha, va="center", fontsize=5.25, linespacing=1.1, color="#222222",
                arrowprops={"arrowstyle":"-", "color":"#444444", "linewidth":.45,
                            "shrinkA":2.5, "shrinkB":2.2, "connectionstyle":"arc3,rad=0"},
                annotation_clip=False, clip_on=False, zorder=6)
            annotations.append((case["letter"], annotation))

    cax = fig.add_axes([(sidebar_x+58)/width, (height-1985-44)/height, (sidebar_width-116)/width, 44/height], zorder=3)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap), cax=cax, orientation="horizontal", extend="max")
    cb.set_ticks(draft.BOUNDS)
    cb.set_ticklabels(["0","0.5","0.8","0.95","1.05","1.25","2","4+"])
    cb.ax.tick_params(labelsize=5, length=1.5, width=.4, pad=1.5)
    cb.outline.set_linewidth(.4)
    cb.set_label("Relative hazard impact\n(facility / global)", fontsize=5.6, labelpad=3)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = []
    for letter, annotation in annotations:
        # get_window_extent includes the leader and text, so validate page containment.
        box = annotation.get_window_extent(renderer)
        assert box.x0 >= sidebar_x and box.x1 <= width and box.y0 >= 0 and box.y1 <= height
        bounds.append({"panel":letter, "label":annotation.get_text(), "bounds":list(box.bounds)})

    with plt.rc_context({"savefig.bbox":None, "savefig.pad_inches":0}):
        for suffix in ["png", "pdf", "svg"]:
            fig.savefig(stem.with_suffix('.'+suffix), dpi=DPI, bbox_inches=None, facecolor="white")
    updated = np.asarray(Image.open(stem.with_suffix('.png')).convert("RGBA"))
    assert updated.shape == (height, width, 4)
    changed = np.any(updated[:,:old_width] != original, axis=2)
    assert not changed[~allowed].any(), "Original map artwork changed away from locator annotations"
    plt.close(fig)
    return {'size_pixels':[width,height], 'label_bounds':bounds,
            'main_grid_changed_pixels_outside_locators':int(changed[~allowed].sum())}
