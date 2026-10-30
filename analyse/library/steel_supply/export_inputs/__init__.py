"""Numerical interfaces between the scientific pipeline and the separate figures.

No renderer, notebook, private preparation script or reference image is imported.
"""
import calendar
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from ..pipeline_paths import data_path
from ..reproduction.execution import sha256,write_json

STAGES=['iron-mine','iron-factory','steel-factory']
HAZARDS=['heatwave','tropical-cyclone','wildfire']
SCENARIOS=['picontrol','ssp126','ssp370','ssp585']
CASES=[('j','iron-mine','heatwave','BRA'),('k','iron-mine','tropical-cyclone','AUS'),
       ('l','iron-factory','heatwave','IND'),('m','steel-factory','wildfire','IDN')]
FACILITIES={'iron-mine':'mine_scaled.parquet','iron-factory':'iron_factory_scaled.parquet',
            'steel-factory':'steel_factory_scaled.parquet'}
LABELS={
    'j':[('Northern System','Vale\nNorthern System\n(Carajás)',30,245,'left')],
    'k':[('Hamersley Consolidated','Rio Tinto\nHamersley mines',26,200,'left'),
         ('Area C','BHP\nMining Area C',50,365,'left')],
    'l':[('JSW Steel Vijayanagar steel plant','JSW Steel\nVijayanagar',28,350,'left'),
         ('Tata Steel Jamshedpur steel plant','Tata Steel\nJamshedpur',842,215,'right')],
    'm':[('Dexin Steel Morowali plant','Dexin Steel\nMorowali',825,465,'right'),
         ('Krakatau Steel Cilegon plant','Krakatau Steel\nCilegon',50,465,'left')]}


def annual_totals(frame):
    numeric=frame[['daily_fe_eq_weight_kg_scaled','production_loss_kg_scaled']]
    if not np.isfinite(numeric).all().all() or (numeric<0).any().any():
        raise ValueError('Daily production/loss must be finite, nonnegative; missing is not zero')
    production=frame.daily_fe_eq_weight_kg_scaled.sum()
    loss=frame.production_loss_kg_scaled.sum()
    if production<=0 or loss>production*(1+1e-10):
        raise ValueError('Annual loss outside physical bounds')
    return production,loss,loss/production


def seasonal_totals(frame,year):
    keys=['country_a3','day_of_year']
    days=366 if calendar.isleap(year) else 365
    if frame[keys].isna().any().any() or frame.duplicated(keys).any():
        raise ValueError('Duplicate/missing country-day identity')
    if not frame.day_of_year.between(1,days).all() or not frame.groupby('country_a3').day_of_year.nunique().eq(days).all():
        raise ValueError('Incomplete country-day calendar')
    if not np.isfinite(frame.production_loss_kg_scaled).all() or frame.production_loss_kg_scaled.lt(0).any():
        raise ValueError('Invalid daily loss')
    dates=pd.Timestamp(f'{year}-01-01')+pd.to_timedelta(frame.day_of_year-1,unit='D')
    temp=frame[['country_a3','production_loss_kg_scaled']].copy()
    temp['season']=np.where(dates.dt.month.between(4,9),'Apr-Sep','Oct-Mar')
    return temp.groupby(['country_a3','season']).agg(
        loss_kg=('production_loss_kg_scaled','sum'),days_covered=('production_loss_kg_scaled','size')).reset_index()


def save(frame,path,**kwargs):
    path.parent.mkdir(parents=True,exist_ok=True)
    frame.to_csv(path,index=False,**kwargs)


def period_plot_values(annual):
    def period(frame,year):
        values=frame[(frame.availability=='available')&frame.year.between(year-5,year+5)].sort_values('year').annual_loss_fraction.to_numpy()
        if not len(values):return np.nan,np.nan,0
        return np.mean(values),stats.sem(values)*stats.t.ppf(.975,len(values)-1) if len(values)>1 else 0.,len(values)
    records=[]
    for stage,hazard in itertools.product(STAGES,HAZARDS):
        group=annual[(annual.industry==stage)&(annual.hazard==hazard)]
        baseline,_,n=period(group[group.scenario=='ssp370'],2023)
        if n!=11 or baseline<=0:raise ValueError('Invalid eleven-year baseline')
        for year,scenario in itertools.product([2035,2060,2100],SCENARIOS):
            mean,ci,n=period(group[group.scenario==scenario],year)
            records.append(dict(industry=stage,hazard=hazard,target_year=year,scenario=scenario,
                                mean_risk_ratio=mean/baseline,temporal_95ci_half_width=ci/baseline,
                                baseline_scenario='ssp370',baseline_center_year=2023,
                                requested_window_years=11,available_years=n))
    return pd.DataFrame(records)


def world_geometry(destination):
    import geopandas as gpd
    raw=gpd.read_file(data_path('input/spatialization/world_administration_esri.gpkg'))
    mapping=pd.read_excel(data_path('input/mfa/mfa_inputs.xlsx'),sheet_name='国家或地区代号')
    raw['ISO_A3']=None
    for _,row in mapping.iterrows():
        raw.loc[raw.ISO_CC.eq(row['Alpha-2 code']),'ISO_A3']=row['Alpha-3 code']
    coded=raw.dropna(subset=['ISO_A3']).copy()
    uncoded=raw[raw.ISO_A3.isna()].copy()
    countries=coded.dissolve(by='ISO_A3',as_index=False)
    countries['geometry']=countries.simplify(.12,preserve_topology=False)
    uncoded['geometry']=uncoded.simplify(.12,preserve_topology=False)
    world=gpd.GeoDataFrame(pd.concat([countries,uncoded],ignore_index=True),crs=coded.crs)[['ISO_A3','geometry']]
    world.to_parquet(destination,index=False)
    return world


def facility_inputs(destination):
    names=['facility_points.csv','facility_labels.csv','facility_annual.csv','facility_global_annual.csv']
    impacts=pd.read_csv(data_path('results/postprocessing/facility_risk/facility_relative_impacts.csv'))
    points,labels,annual,globals_=[] ,[],[],[]
    for panel,stage,hazard,country in CASES:
        scaled=pd.read_parquet(data_path('results/facilities')/FACILITIES[stage])
        selected=impacts[(impacts.industry==stage)&(impacts.hazard==hazard)&(impacts.country_a3==country)].copy()
        selected=selected.merge(scaled[['ID_internal','Name']],on='ID_internal',how='left',validate='one_to_one')
        if selected.Name.isna().any():raise ValueError('Missing facility identity')
        ids={site.ID_internal:f'{panel}{i:03d}' for i,site in enumerate(selected.itertuples(index=False),1)}
        for site in selected.itertuples(index=False):points.append((ids[site.ID_internal],panel,site.Geo_lon,site.Geo_lat))
        for name,label,x,y,align in LABELS[panel]:
            matched=selected.loc[selected.Name.eq(name)]
            if len(matched)!=1:raise ValueError('Label source not unique')
            labels.append((ids[matched.iloc[0].ID_internal],label,x,y,align))
        for year in range(2055,2066):
            daily=pd.read_parquet(data_path('results/hazard_facility_loss')/f'{stage}_{hazard}_{year}_ssp370.parquet')
            yearly=daily.groupby('ID_internal').agg(production=('daily_fe_eq_weight_kg_scaled','sum'),loss=('production_loss_kg_scaled','sum')).reset_index()
            totals=yearly[['production','loss']].sum()
            globals_.append((panel,year,totals.loss/totals.production))
            for row in yearly[yearly.ID_internal.isin(ids)].itertuples(index=False):annual.append((ids[row.ID_internal],year,row.loss/row.production))
    for rows,columns,name in [(points,['point_id','panel','longitude','latitude'],names[0]),
                              (labels,['point_id','display_label','x_offset','y_offset','align'],names[1]),
                              (annual,['point_id','year','loss_fraction'],names[2]),
                              (globals_,['panel','year','loss_fraction'],names[3])]:
        save(pd.DataFrame(rows,columns=columns),destination/name)
    return 'recomputed from facility results and source facility identities'


def export_hazards(output):
    fig2,fig3,fig4=[output/f'Figure_{i}' for i in [2,3,4]]
    for folder in [fig2,fig3,fig4]:folder.mkdir(parents=True,exist_ok=True)
    world=world_geometry(fig3/'world_boundaries.parquet')
    codes=['GLOBAL',*world.ISO_A3.dropna().unique().tolist()]
    global_rows,country_rows,season_rows,source_rows=[],[],[],[]
    for stage,hazard,scenario,year in itertools.product(STAGES,HAZARDS,SCENARIOS,range(2018,2101)):
        name=f'{stage}_{hazard}_{year}_{scenario}.parquet'
        path=data_path('results/hazard_country_level_loss')/name
        row=dict(industry=stage,hazard=hazard,scenario=scenario,year=year,
                 baseline_production_fe_kg=np.nan,production_loss_fe_kg=np.nan,
                 annual_loss_fraction=np.nan,availability='unavailable',source_file='',source_sha256='')
        if not path.is_file():
            if not(hazard=='tropical-cyclone' and scenario=='ssp126' and year<2061):raise FileNotFoundError(path)
            global_rows.append(row);continue
        frame=pd.read_parquet(path)
        production,loss,fraction=annual_totals(frame)
        digest=sha256(path)
        row.update(baseline_production_fe_kg=production,production_loss_fe_kg=loss,
                   annual_loss_fraction=fraction,availability='available',source_file=name,source_sha256=digest)
        global_rows.append(row)
        if scenario=='ssp370' and 2055<=year<=2065:
            for code in codes:
                subset=frame if code=='GLOBAL' else frame[frame.country_a3.eq(code)]
                p=subset.daily_fe_eq_weight_kg_scaled.sum() if len(subset) else np.nan
                l=subset.production_loss_kg_scaled.sum() if len(subset) else np.nan
                country_rows.append((stage,hazard,code,year,p,l))
            seasonal=seasonal_totals(frame,year)
            season_rows.extend((stage,hazard,scenario,year,r.country_a3,r.season,r.loss_kg,r.days_covered) for r in seasonal.itertuples(index=False))
            source_rows.append(dict(source_file=name,sha256=digest,stage=stage,hazard=hazard,
                                    scenario=scenario,year=year,countries=frame.country_a3.nunique(),daily_rows=len(frame)))
    annual=pd.DataFrame(global_rows)
    save(annual,fig2/'annual_global_losses.csv',float_format='%.17g',na_rep='')
    save(period_plot_values(annual),fig2/'reference_plot_values.csv',float_format='%.17g',na_rep='')
    save(pd.DataFrame(country_rows,columns=['stage','hazard','country','year','production_kg','loss_kg']),fig3/'country_annual.csv')
    save(pd.DataFrame(season_rows,columns=['stage','hazard','scenario','year','country','season','loss_kg','days_covered']),fig4/'country_seasonal_losses.csv')
    save(pd.DataFrame(source_rows),fig4/'source_files.csv')
    return facility_inputs(fig3)


def select_named_endpoints(endpoints):
    trades=['free-trade','us-cn-decoupling','bloc-confrontation','global-decoupling']
    selected=endpoints[endpoints.climate.eq('ssp370')&endpoints.inventory.eq('4.0')&endpoints.capacity.eq('1.1')&endpoints.trade.isin(trades)]
    expected={(trade,str(year)) for trade in trades for year in [2023,2100]}
    observed=set(zip(selected.trade,selected.year.astype(str)))
    if len(selected)!=8 or observed!=expected:raise ValueError('Expected baseline and 2100 endpoints for each named trade scenario')
    return selected


def export_dynamic(output):
    def read(path):return pd.read_csv(path,dtype=str,keep_default_na=False)
    metrics=data_path('results/postprocessing/formal_abm_metrics')
    dscci=data_path('results/postprocessing/e2eci')
    f5,f6,si=[output/name for name in ['Figure_5','Figure_6','Supplementary_Figure_2']]
    fields=['year','annual_mass','coverage_days','loss_percent','climate','trade','inventory','capacity']
    annual=read(metrics/'full_annual_climate_inventory.csv')
    steps=read(metrics/'ssp370_inventory_step_series.csv')
    steps=steps[(pd.to_datetime(steps.start_date)<'2101-01-01')&(pd.to_datetime(steps.end_date_exclusive)>'2100-01-01')]
    save(annual[fields],f5/'annual_summary.csv')
    save(steps[['step','mass','start_date','end_date_exclusive','step_days','loss_percent','climate','trade','inventory','capacity']],f5/'within_year_2100.csv')
    country=read(metrics/'e2eci_with_formal_satisfaction.csv').rename(columns={
        'E2ECI':'dscci','fulfilment_percent':'final_demand_satisfaction','loss_percent':'final_demand_loss'})
    scores=read(dscci/'seven_stage_e2eci_country_scores.csv').set_index('Country')
    country['mfa_final_demand_2023']=country.country.map(scores.final_demand)
    if country.mfa_final_demand_2023.isna().any():raise ValueError('Missing DSCCI country demand')
    save(country,f6/'country_capability.csv')
    save(read(metrics/'random_trade_endpoints.csv')[fields+['random_ratio','random_seed']],f6/'random_trade_endpoints.csv')
    endpoints=read(metrics/'capacity_and_named_trade_endpoints.csv')
    selected=select_named_endpoints(endpoints)
    save(selected[fields],f6/'global_trade_endpoints.csv')
    save(read(metrics/'named_trade_country_final_demand.csv')[['country','2023','2100','fulfilment_percent','loss_percent','trade']],f6/'country_trade_endpoints.csv')
    figure=data_path('results/postprocessing/figure_metrics')
    save(read(figure/'production_components_annual.csv')[['country','sector','actual_output','year']],f6/'production_components.csv')
    components=read(dscci/'positive_demand_country_stage_adequacy.csv')
    components=components[components.Country.isin(['CHN','USA','IND','BRA','AUS','CAN','DEU','GBR'])&components.Stage.isin([f'P{i}' for i in range(1,7)])]
    components=components[['Country','Stage','total_inflow','total_outflow','baseline_throughput','capacity_factor','modeled_capacity','final_demand']].rename(columns={'Country':'country','Stage':'sector'})
    save(components,f6/'baseline_capacity_components.csv')
    save(read(figure/'capacity_sensitivity_annual.csv'),si/'capacity_sensitivity_annual.csv')


def export_all():
    output=data_path('results/graphing_inputs')
    output.mkdir(parents=True,exist_ok=True)
    facility_origin=export_hazards(output)
    export_dynamic(output)
    files={p.relative_to(output).as_posix():sha256(p) for p in output.rglob('*') if p.is_file()}
    write_json(output/'export_manifest.json',dict(mode='full',facility_data_origin=facility_origin,
               file_hashes=files,note='Numerical inputs only. No graphing sources or accepted figures overwritten.'))
    print(json.dumps(dict(output=str(output),files=len(files),mode='full'),indent=2))
