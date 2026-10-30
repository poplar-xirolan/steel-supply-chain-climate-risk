"""Compact scientific figure products, regenerated from canonical ABM outputs."""
from __future__ import annotations
from steel_supply.pipeline_paths import science_path, bundle_root, data_path
import json
from pathlib import Path
import numpy as np
import pandas as pd
from ..reproduction.execution import sha256, write_json

COUNTRIES = ["CHN", "USA", "IND", "BRA", "AUS", "CAN", "DEU", "GBR"]


def validate_capacity_table(frame):
    expected = pd.MultiIndex.from_product([[1.,1.1,1.2,1.3,1.4,1.5], range(2023,2101)], names=['capacity','year'])
    index = pd.MultiIndex.from_frame(frame[['capacity','year']])
    if index.has_duplicates or set(index) != set(expected):
        raise ValueError('Incomplete capacity/year grid')
    columns = ['annual_mass','coverage_days','mean_daily_mass','loss_percent']
    if not np.isfinite(frame[columns].to_numpy(float)).all() or frame.annual_mass.lt(0).any():
        raise ValueError('Invalid capacity annual values')
    days = pd.to_datetime(frame.year.astype(str)+'-12-31').dt.dayofyear
    if not np.array_equal(frame.coverage_days, days):
        raise ValueError('Incomplete calendar coverage')
    if not np.allclose(frame.mean_daily_mass, frame.annual_mass/days, rtol=1e-12, atol=1e-12):
        raise ValueError('Annual mass/daily rate mismatch')
    baseline = frame.loc[frame.year.eq(2023)].set_index('capacity').mean_daily_mass
    if baseline.le(0).any():
        raise ValueError('Nonpositive baseline')
    expected_loss = 100*(1-frame.mean_daily_mass/frame.capacity.map(baseline))
    if not np.allclose(frame.loss_percent, expected_loss, rtol=1e-10, atol=1e-10):
        raise ValueError('Capacity loss ratio mismatch')


def _load_verified(root, table, receipt):
    folder = science_path(Path(root), 'results/postprocessing/figure_metrics')
    path = folder/table
    provenance = json.loads((folder/receipt).read_text(encoding='utf-8'))
    if sha256(path) != provenance.get('table_sha256'):
        raise ValueError(f'Compact table hash mismatch: {path}')
    return pd.read_csv(path), provenance


def load_capacity_table(root):
    frame, _ = _load_verified(root, 'capacity_sensitivity_annual.csv', 'capacity_provenance.json')
    validate_capacity_table(frame)
    return frame


def load_production_table(root):
    frame, provenance = _load_verified(root, 'production_components_annual.csv', 'production_provenance.json')
    if frame.duplicated(['country','sector','year']).any():
        raise ValueError('Duplicate production components')
    if set(frame.year) != {2023,2100} or not np.isfinite(frame.actual_output).all() or frame.actual_output.lt(0).any():
        raise ValueError('Invalid production components')
    return frame, provenance


def capacity_annual(run):
    """Read each declared step once; missing file is not zero final demand."""
    import pyarrow.dataset as ds
    from ..abm_annual_metrics import annualize, add_loss
    meta = json.loads((run/'run_metadata.json').read_text(encoding='utf-8'))
    calendar = pd.DataFrame(meta['steps']).rename(columns={'index':'step'})
    start, end = pd.to_datetime(calendar.start_date), pd.to_datetime(calendar.end_date_exclusive)
    if (start.iloc[0] != pd.Timestamp('2023-01-01') or end.iloc[-1] != pd.Timestamp('2101-01-01')
            or not np.array_equal(end.iloc[:-1].to_numpy(), start.iloc[1:].to_numpy())
            or not np.array_equal((end-start).dt.days, calendar.step_days)):
        raise ValueError('Incomplete or overlapping capacity-run calendar')
    paths = [run/'flows'/f'flows_step_{s}.parquet' for s in calendar.step]
    if not all(p.is_file() for p in paths):
        raise FileNotFoundError('Missing capacity-run flow files')
    frame = ds.dataset([str(p) for p in paths],format='parquet').to_table(
        columns=['step','mass'],filter=ds.field('sector_to')=='S1').to_pandas()
    if not np.isfinite(frame.mass).all() or frame.mass.lt(0).any():
        raise ValueError('Invalid final-demand flow mass')
    totals = frame.groupby('step').mass.sum()
    steps = calendar.copy()
    steps['mass'] = steps.step.map(totals).fillna(0.)
    steps.attrs['step_calendar'] = meta['steps']
    annual = add_loss(annualize(steps))
    hashes = {str(p.relative_to(run)):sha256(p) for p in paths}
    hashes['run_metadata.json'] = sha256(run/'run_metadata.json')
    return annual, hashes


def build(root, output=None):
    from ..hazard_dynamic.reproduction_grid import load_catalogue, resolve_run_path
    root = Path(root).resolve()
    output = Path(output).resolve() if output is not None else science_path(root, 'results/postprocessing/figure_metrics')
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Use an empty compact-results destination: {output}')
    catalog = load_catalogue(root)
    production, provenance = production_components(root)
    rows, capacity_sources = [], []
    for cap in [1.,1.1,1.2,1.3,1.4,1.5]:
        selected = catalog.loc[catalog.climate.eq('ssp370') & catalog.trade.eq('free-trade')
                               & catalog.inventory.eq(4.) & catalog.capacity.eq(cap)]
        if len(selected) != 1:
            raise ValueError(f'Expected exactly one capacity scenario: {cap}')
        run = resolve_run_path(selected.iloc[0].run_folder, root)
        print(f'Building compact capacity series: {cap}', flush=True)
        annual, hashes = capacity_annual(run)
        annual['capacity'] = cap
        rows.append(annual)
        capacity_sources.append({'capacity':cap, 'run':run.relative_to(root).as_posix(), 'source_hashes':hashes})
    capacity = pd.concat(rows,ignore_index=True)
    validate_capacity_table(capacity)
    output.mkdir(parents=True,exist_ok=True)
    production_path, capacity_path = output/'production_components_annual.csv', output/'capacity_sensitivity_annual.csv'
    production.to_csv(production_path,index=False)
    capacity.to_csv(capacity_path,index=False)
    write_json(output/'production_provenance.json', {**provenance,'table_sha256':sha256(production_path)})
    write_json(output/'capacity_provenance.json', {'table_sha256':sha256(capacity_path), 'sources':capacity_sources,
        'definition':'Calendar-overlap allocated annual S1 mass; loss relative to the 2023 mean daily mass. SSP3-7.0, free trade, four-week inventory.'})
    return output

def production_components(root: Path, countries=COUNTRIES):
    root = Path(root).resolve()
    METRICS = science_path(root, 'results/postprocessing/formal_abm_metrics')
    MFA_SOURCE = science_path(root, 'results/mfa/2023.parquet')
    """Read existing engine states and independently reconcile production flows.

    No simulation is run and no formal summary table is overwritten.
    """
    import pyarrow.dataset as ds

    catalog = pd.read_csv(METRICS / "scenario_catalog.csv")
    selected = catalog.loc[catalog.climate.eq("ssp370") & catalog.trade.eq("free-trade")
                           & catalog.inventory.eq(4.) & catalog.capacity.eq(1.1)]
    if len(selected) != 1:
        raise ValueError("Expected one baseline SSP370 production run")
    from steel_supply.hazard_dynamic.reproduction_grid import resolve_run_path
    run = resolve_run_path(selected.iloc[0].run_folder, root)
    meta_path = run / "run_metadata.json"
    meta = pd.DataFrame(json.loads(meta_path.read_text(encoding="utf-8"))["steps"]).rename(
        columns={"index": "step"})
    start, end = pd.to_datetime(meta.start_date), pd.to_datetime(meta.end_date_exclusive)
    if not np.array_equal((end - start).dt.days, meta.step_days):
        raise ValueError("Engine step durations disagree with calendar")
    weights = {
        y: ((end.clip(upper=pd.Timestamp(y+1, 1, 1))
             - start.clip(lower=pd.Timestamp(y, 1, 1))).dt.total_seconds()/86400
            ).clip(lower=0) / meta.step_days
        for y in (2023, 2100)
    }
    for year, w in weights.items():
        if not np.isclose((w * meta.step_days).sum(), 365):
            raise ValueError(f"Incomplete annual coverage: {year}")
    chosen = meta.loc[(weights[2023] > 0) | (weights[2100] > 0)]
    sectors = [f"P{i}" for i in range(1, 7)]
    agent_paths = [run / "agents" / f"agents_step_{s}.parquet" for s in chosen.step]
    flow_paths = [run / "flows" / f"flows_step_{s}.parquet" for s in chosen.step]
    if not all(p.is_file() for p in agent_paths + flow_paths):
        raise FileNotFoundError("Missing endpoint engine files")
    agents = ds.dataset([str(p) for p in agent_paths], format="parquet").to_table(
        columns=["step", "country", "sector", "stage", "capacity_left"],
        filter=ds.field("country").isin(countries) & ds.field("sector").isin(sectors)
    ).to_pandas()
    if agents.duplicated(["step", "country", "sector", "stage"]).any():
        raise ValueError("Duplicate production agent states")
    states = agents.pivot(index=["step", "country", "sector"], columns="stage",
                          values="capacity_left")
    if not {"START", "END"}.issubset(states.columns) or not np.isfinite(states[["START", "END"]]).all().all():
        raise ValueError("Incomplete production agent START/END states")
    if states.reset_index().groupby(["country", "sector"]).size().ne(len(chosen)).any():
        raise ValueError("Missing production agent intervals")
    production = (states.START - states.END).rename("realized_throughput")
    if (production < -1e-5).any():
        raise ValueError("Negative realized production")
    production = production.clip(lower=0)
    flows = ds.dataset([str(p) for p in flow_paths], format="parquet").to_table(
        columns=["step", "entity_from", "sector_from", "sector_to", "mass"],
        filter=ds.field("entity_from").isin(countries) & ds.field("sector_from").isin(sectors)
    ).to_pandas()
    outgoing = flows.groupby(["step", "entity_from", "sector_from"]).mass.sum()
    outgoing.index.names = production.index.names
    outgoing = outgoing.reindex(production.index, fill_value=0)
    main_products = {"P1": "M1", "P2": "M2", "P3": "M3", "P4": "M4", "P5": "M5", "P6": "M7"}
    primary_flows = flows.loc[flows.sector_to.eq(flows.sector_from.map(main_products))]
    output = primary_flows.groupby(["step", "entity_from", "sector_from"]).mass.sum()
    output.index.names = production.index.names
    output = output.reindex(production.index, fill_value=0).rename("actual_output")
    # Used capacity is throughput. Apply the fixed product yield before comparing
    # it with produced steel/cast iron, so process losses are not counted as output.
    mfa = pd.read_parquet(MFA_SOURCE)
    incoming = mfa.groupby(["entity_to", "sector_to"]).mass.sum()
    outgoing_mfa = mfa.groupby(["entity_from", "sector_from"]).mass.sum()
    primary_mfa = mfa.loc[mfa.sector_to.eq(mfa.sector_from.map(main_products))].groupby(
        ["entity_from", "sector_from"]).mass.sum()
    yields = np.array([
        primary_mfa.get((c, p), 0.) / max(incoming.get((c, p), 0.), outgoing_mfa.get((c, p), 0.))
        for _, c, p in production.index
    ])
    if not np.allclose(production * yields, output, rtol=1e-10, atol=1e-4):
        raise ValueError("Actual main-product output disagrees with used capacity times product yield")
    step_values = pd.concat([production, output, outgoing.rename("outbound_delivery")], axis=1).reset_index()
    annual = []
    for year, w in weights.items():
        frame = step_values.copy()
        factor = frame.step.map(dict(zip(meta.step, w)))
        frame[["realized_throughput", "actual_output", "outbound_delivery"]] = frame[
            ["realized_throughput", "actual_output", "outbound_delivery"]].mul(factor, axis=0)
        result = frame.groupby(["country", "sector"], as_index=False)[
            ["realized_throughput", "actual_output", "outbound_delivery"]].sum()
        result["year"] = year
        annual.append(result)
    provenance = {
        "run_folder": str(run.relative_to(root)),
        "run_metadata_sha256": sha256(meta_path),
        "source_file_count": len(agent_paths) + len(flow_paths),
        "source_hashes": {str(p.relative_to(root)): sha256(p) for p in agent_paths + flow_paths},
        "definition": "Actual output = outbound flow to the stage's main-product market (P1:M1, P2:M2, P3:M3, P4:M4, P5:M5, P6:M7), reconciled step by step with used capacity times the fixed MFA product yield. Process losses and replenishment receipts are not product output.",
        "annual_allocation": "Each step's mass times its overlap days with the calendar year divided by actual step days; 365 days in each endpoint year.",
    }
    return pd.concat(annual, ignore_index=True), provenance


