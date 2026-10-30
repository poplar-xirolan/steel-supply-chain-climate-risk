import importlib
import json
import calendar
import pandas as pd
import pytest


def mod():
    return importlib.import_module('steel_supply.postprocessing.compact_figure_metrics')


def table():
    return pd.DataFrame([dict(capacity=cap,year=y,annual_mass=float(calendar.isleap(y)+365),
        coverage_days=calendar.isleap(y)+365,mean_daily_mass=1.,loss_percent=0.)
        for cap in [1.,1.1,1.2,1.3,1.4,1.5] for y in range(2023,2101)])


def test_capacity_loader_uses_verified_compact_table_without_trajectories(tmp_path):
    m=mod()
    out=tmp_path/'result/postprocessing/figure_metrics'
    out.mkdir(parents=True)
    path=out/'capacity_sensitivity_annual.csv'
    table().to_csv(path,index=False)
    (out/'capacity_provenance.json').write_text(json.dumps({'table_sha256':m.sha256(path)}))
    result=m.load_capacity_table(tmp_path)
    assert len(result)==468
    assert result.loss_percent.eq(0).all()
    assert not (tmp_path/'result/simulation').exists()
    path.write_text(path.read_text()+'\n')
    with pytest.raises(ValueError,match='hash'):
        m.load_capacity_table(tmp_path)


def test_capacity_rejects_missing_year_and_incomplete_calendar():
    m=mod()
    with pytest.raises(ValueError,match='year|complete'):
        m.validate_capacity_table(table().iloc[1:])
    df=table()
    df.loc[1,'coverage_days']=365
    with pytest.raises(ValueError,match='coverage|calendar'):
        m.validate_capacity_table(df)


def test_capacity_rejects_invalid_loss_despite_full_grid():
    df=table()
    df.loc[2,'loss_percent']=10
    with pytest.raises(ValueError,match='loss|ratio'):
        mod().validate_capacity_table(df)
