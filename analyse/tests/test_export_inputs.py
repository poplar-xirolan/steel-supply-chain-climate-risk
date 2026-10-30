import numpy as np
import pandas as pd
import pytest


def test_facility_reference_subset_cannot_replace_missing_computed_results(tmp_path,monkeypatch):
    from steel_supply.export_inputs import facility_inputs
    monkeypatch.setenv('STEEL_ANALYSE_ROOT',str(tmp_path))
    monkeypatch.setenv('STEEL_INPUT_MODE','provided-summaries')
    reference=tmp_path/'data/provided/plot_auxiliary/Figure_3'
    reference.mkdir(parents=True)
    for name in ['facility_points.csv','facility_labels.csv','facility_annual.csv','facility_global_annual.csv']:
        (reference/name).write_text('reference\n1\n',encoding='utf-8')
    output=tmp_path/'result/graphing_inputs/Figure_3'
    output.mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        facility_inputs(output)
    assert not list(output.iterdir())


def test_world_geometry_is_rebuilt_from_sources_not_reference(tmp_path,monkeypatch):
    import geopandas as gpd
    from shapely.geometry import box
    from steel_supply.export_inputs import world_geometry
    monkeypatch.setenv('STEEL_ANALYSE_ROOT',str(tmp_path))
    monkeypatch.setenv('STEEL_INPUT_MODE','provided-summaries')
    source=tmp_path/'data/external/spatialization/world_administration_esri.gpkg'
    source.parent.mkdir(parents=True)
    gpd.GeoDataFrame({'ISO_CC':['AA']},geometry=[box(0,0,1,1)],crs='EPSG:4326').to_file(source,driver='GPKG')
    workbook=tmp_path/'data/external/mfa/mfa_inputs.xlsx'
    workbook.parent.mkdir(parents=True)
    pd.DataFrame({'Alpha-2 code':['AA'],'Alpha-3 code':['AAA']}).to_excel(workbook,sheet_name='国家或地区代号',index=False)
    reference=tmp_path/'data/provided/plot_auxiliary/Figure_3/world_boundaries.parquet'
    reference.parent.mkdir(parents=True)
    gpd.GeoDataFrame({'ISO_A3':['REF']},geometry=[box(5,5,6,6)],crs='EPSG:4326').to_parquet(reference,index=False)
    output=tmp_path/'result/world_boundaries.parquet'
    output.parent.mkdir(parents=True)
    rebuilt=world_geometry(output)
    assert rebuilt.ISO_A3.dropna().tolist()==['AAA']
    assert gpd.read_parquet(reference).ISO_A3.tolist()==['REF']


def test_annual_totals_preserve_weights_and_missingness():
    from steel_supply.export_inputs import annual_totals
    frame=pd.DataFrame({'daily_fe_eq_weight_kg_scaled':[10.,90.],
                        'production_loss_kg_scaled':[5.,9.]})
    production,loss,fraction=annual_totals(frame)
    assert (production,loss,fraction)==(100.,14.,.14)
    frame.loc[0,'production_loss_kg_scaled']=np.nan
    with pytest.raises(ValueError,match='finite|missing'):
        annual_totals(frame)


def test_seasonal_calendar_separates_april_september():
    from steel_supply.export_inputs import seasonal_totals
    frame=pd.DataFrame({'country_a3':['AAA']*366,'day_of_year':range(1,367),
                        'production_loss_kg_scaled':[1.]*366})
    result=seasonal_totals(frame,2024).set_index(['country_a3','season'])
    assert result.loc[('AAA','Apr-Sep'),'loss_kg']==183
    assert result.loc[('AAA','Oct-Mar'),'days_covered']==183
    with pytest.raises(ValueError,match='Incomplete|Duplicate'):
        seasonal_totals(frame.iloc[:-1],2024)


def test_trade_endpoints_keep_baseline_and_2100_for_each_named_scenario():
    from steel_supply.export_inputs import select_named_endpoints
    trades=['free-trade','us-cn-decoupling','bloc-confrontation','global-decoupling']
    frame=pd.DataFrame([dict(year=str(y),climate='ssp370',inventory='4.0',capacity='1.1',trade=t)
                        for t in trades for y in [2023,2100]])
    assert len(select_named_endpoints(frame))==8
    with pytest.raises(ValueError,match='endpoint'):
        select_named_endpoints(frame.iloc[:-1])
