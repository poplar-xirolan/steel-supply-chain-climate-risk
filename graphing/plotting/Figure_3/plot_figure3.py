"""Recompute and draw all thirteen Figure 3 panels from the supplied source data."""
from pathlib import Path
import argparse
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import geopandas as gpd
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from common.figure_style import apply_figure_style, check_reference_font, panel_label
from map_geometry import BOUNDS, CASES, STAGES, HAZARDS, STAGE_LABELS, HAZARD_LABELS, MAP_RISK_COLORS, MUTED_INK
from render_composite import draw

DEFAULT_DATA = HERE.parents[1] / 'data/Figure_3'
YEARS = set(range(2055, 2066))


def unique(frame, key):
    if frame[key].isna().any().any() or frame.duplicated(key).any():
        raise ValueError(f'Missing or duplicate keys: {key}')


def physical_totals(frame):
    values = frame[['production_kg','loss_kg']]
    missing = values.isna()
    if (missing.production_kg != missing.loss_kg).any():
        raise ValueError('Production and loss must be missing together')
    present = values[~missing.production_kg]
    if not np.isfinite(present).all().all() or (present < 0).any().any():
        raise ValueError('Production/loss must be finite and nonnegative')
    if (present.loss_kg > present.production_kg + 1e-10*present.production_kg.clip(lower=1)).any():
        raise ValueError('Loss exceeds production')


def country_values(annual):
    """Annual loss/production -> period mean -> country/global period-mean ratio."""
    unique(annual, ['stage','hazard','country','year'])
    physical_totals(annual)
    rows = []
    for (stage,hazard), group in annual.groupby(['stage','hazard'], sort=False):
        global_annual = group[group.country.eq('GLOBAL')].sort_values('year')
        if global_annual.empty or global_annual.production_kg.isna().any() or global_annual.production_kg.le(0).any():
            raise ValueError('Missing or nonpositive global production')
        denominator = float(np.mean((global_annual.loss_kg/global_annual.production_kg).to_numpy()))
        for country, values in group[~group.country.eq('GLOBAL')].groupby('country', sort=False):
            values = values.sort_values('year')
            valid = values.production_kg.gt(0) & values.loss_kg.notna()
            fractions = (values.loc[valid,'loss_kg']/values.loc[valid,'production_kg']).to_numpy()
            mean = float(np.mean(fractions)) if len(fractions) else np.nan
            ratio = mean/denominator if denominator > 0 else np.nan
            rows.append((stage,hazard,country,mean,denominator,ratio,len(fractions)))
    return pd.DataFrame(rows,columns=['stage','hazard','country','mean_loss_fraction','global_mean_loss_fraction','country_to_global_ratio','available_years'])


def facility_values(points, annual, global_annual):
    """Keep the all-facility global denominator; never normalize to the inset."""
    unique(points,['point_id'])
    unique(annual,['point_id','year'])
    unique(global_annual,['panel','year'])
    if set(points.point_id) != set(annual.point_id) or set(points.panel) != set(global_annual.panel):
        raise ValueError('Facility and global input identities do not match')
    for frame in [annual,global_annual]:
        if not np.isfinite(frame.loss_fraction).all() or not frame.loss_fraction.between(0,1+1e-10).all():
            raise ValueError('Invalid facility/global annual loss fraction')
    means = annual.groupby('point_id',sort=False).agg(mean_loss_fraction=('loss_fraction','mean'),available_years=('loss_fraction','count')).reset_index()
    result = points.merge(means,on='point_id',validate='one_to_one',sort=False)
    denominators = {}
    for panel, group in global_annual.groupby('panel',sort=False):
        denominator = float(group.sort_values('year').loss_fraction.mean())
        if denominator <= 0:
            raise ValueError('Facility ratio undefined for zero global loss')
        denominators[panel] = denominator
        expected = set(group.year)
        ids = points.loc[points.panel.eq(panel),'point_id']
        if any(set(values.year) != expected for _,values in annual[annual.point_id.isin(ids)].groupby('point_id')):
            raise ValueError('Incomplete facility year coverage')
    result['global_mean_loss_fraction'] = result.panel.map(denominators)
    result['facility_to_global_ratio'] = result.mean_loss_fraction/result.global_mean_loss_fraction
    return result


def read_data(data_dir):
    def table(name, fields):
        frame = pd.read_csv(data_dir/name, float_precision='round_trip', keep_default_na=True)
        if set(frame.columns) != set(fields):
            raise ValueError(f'Unexpected fields in {name}; expected {fields}')
        return frame
    annual = table('country_annual.csv',['stage','hazard','country','year','production_kg','loss_kg'])
    points = table('facility_points.csv',['point_id','panel','longitude','latitude'])
    facility = table('facility_annual.csv',['point_id','year','loss_fraction'])
    glob = table('facility_global_annual.csv',['panel','year','loss_fraction'])
    labels = table('facility_labels.csv',['point_id','display_label','x_offset','y_offset','align'])
    geometry = gpd.read_parquet(data_dir/'world_boundaries.parquet')
    if set(geometry.columns) != {'ISO_A3','geometry'} or geometry.crs is None or geometry.crs.to_epsg() != 4326:
        raise ValueError('Expected country geometry in EPSG:4326')
    # The accepted 0.12-degree simplification collapses some tiny uncoded parts.
    # Preserve valid empty geometries and skip them when drawing, as originally.
    if geometry.geometry.isna().any() or not geometry.geometry.is_valid.all() or geometry.geometry.is_empty.all():
        raise ValueError('Invalid, missing or entirely empty boundary geometry')
    codes = set(geometry.ISO_A3.dropna()) | {'GLOBAL'}
    unique(annual,['stage','hazard','country','year'])
    expected = pd.MultiIndex.from_product([STAGES,HAZARDS,sorted(codes),sorted(YEARS)])
    actual = pd.MultiIndex.from_frame(annual[['stage','hazard','country','year']])
    if len(expected.difference(actual)) or len(actual.difference(expected)):
        raise ValueError('Incomplete or unexpected country/stage/hazard/year coverage')
    physical_totals(annual)
    unique(points,['point_id'])
    if not np.isfinite(points[['longitude','latitude']]).all().all() or set(points.panel) != set('jklm'):
        raise ValueError('Invalid point coordinates/panels')
    for case in CASES:
        selected = points[points.panel.eq(case['letter'])]
        x0,x1,y0,y1 = case['extent']
        if not selected.longitude.between(x0,x1).all() or not selected.latitude.between(y0,y1).all():
            raise ValueError(f"Points outside panel {case['letter']}")
    for frame,key in [(facility,'point_id'),(glob,'panel')]:
        unique(frame,[key,'year'])
        if any(set(group.year) != YEARS for _,group in frame.groupby(key)):
            raise ValueError('Expected complete 2055–2065 annual input')
    unique(labels,['point_id'])
    if not set(labels.point_id).issubset(set(points.point_id)) or labels.isna().any().any():
        raise ValueError('Missing label or point-label association')
    if not labels['align'].isin(['left','right']).all() or not np.isfinite(labels[['x_offset','y_offset']]).all().all():
        raise ValueError('Invalid annotation placement')
    facility_values(points,facility,glob)
    return annual,points,facility,glob,labels,geometry


def render_country_grid(values, world, output):
    apply_figure_style()
    cmap = ListedColormap(MAP_RISK_COLORS)
    norm = BoundaryNorm(BOUNDS,cmap.N,extend='max')
    fig, axes = plt.subplots(3,3,figsize=(6.15,3.95))
    fig.subplots_adjust(wspace=.025,hspace=.08,left=.06,right=.995,top=.92,bottom=.16)
    for i,stage in enumerate(STAGES):
        for j,hazard in enumerate(HAZARDS):
            ax = axes[i,j]
            subset = values[values.stage.eq(stage)&values.hazard.eq(hazard)]
            data_map = subset.set_index('country').country_to_global_ratio.to_dict()
            base = world.assign(rel_val=world.ISO_A3.map(data_map))
            no_data = base[base.rel_val.isna()]
            if len(no_data):
                no_data.plot(ax=ax,color='#F4F4F4',edgecolor='#B8BDC1',linewidth=.10,hatch='////')
            present = base.dropna(subset=['rel_val'])
            if len(present):
                present.plot(column='rel_val',ax=ax,cmap=cmap,norm=norm,edgecolor='#AAAAAA',linewidth=.045)
            ax.set_axis_off()
            ax.set_aspect('equal')
            panel_label(ax,i*3+j,x=.014,y=.88)
            if j==0:
                ax.text(-.035,.5,STAGE_LABELS[stage],transform=ax.transAxes,va='center',ha='right',fontsize=6.5,fontweight='normal',rotation=90)
            if i==0:
                ax.set_title(HAZARD_LABELS[hazard],pad=2,fontweight='normal')
    cax = fig.add_axes([.20,.075,.60,.024])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cmap),cax=cax,orientation='horizontal',extend='max')
    cb.outline.set_linewidth(.4)
    cb.set_label('Relative hazard impact (country / global)',labelpad=3)
    cb.set_ticks(BOUNDS)
    cb.ax.set_xticklabels(['0','0.5','0.8','0.95','1.05','1.25','2','4+'])
    cb.ax.tick_params(length=1.8,width=.4,pad=1.5)
    fig.text(.20,.115,'Lower relative hazard impact',fontsize=6.2,color=MUTED_INK,ha='left',va='bottom')
    fig.text(.80,.115,'Higher relative hazard impact',fontsize=6.2,color=MUTED_INK,ha='right',va='bottom')
    apply_figure_style()
    fig.savefig(output,dpi=600,bbox_inches='tight',facecolor='white')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,default=DEFAULT_DATA)
    parser.add_argument('--output-dir',type=Path,default=HERE/'output')
    args = parser.parse_args()
    check_reference_font()
    annual,points,facility,glob,labels,world = read_data(args.data_dir)
    country = country_values(annual)
    sites = facility_values(points,facility,glob)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    base_png = args.output_dir/'Figure_3_country_grid.png'
    render_country_grid(country,world,base_png)
    details = draw(sites,labels,args.data_dir/'world_boundaries.parquet',base_png,args.output_dir)
    country.to_csv(args.output_dir/'Figure_3_country_values.csv',index=False)
    sites.to_csv(args.output_dir/'Figure_3_facility_values.csv',index=False)
    print(json.dumps({'output':str(args.output_dir.resolve()),'facility_points':len(sites),**details},indent=2))


if __name__=='__main__':
    main()
