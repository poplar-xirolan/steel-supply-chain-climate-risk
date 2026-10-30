import json
from pathlib import Path

import pandas as pd
import pytest


def test_real_batch_runner_resume_and_tamper(tmp_path,monkeypatch):
    from steel_supply.smoke import toy_network
    from steel_supply.hazard_dynamic import reproduction_grid as module
    root=tmp_path/'bundle'
    mfa=root/'result/mfa/2023.parquet'
    mfa.parent.mkdir(parents=True)
    toy_network().to_parquet(mfa,index=False)
    command=module.commands(root)[0]
    for flag,value in {'--hazard-type':'none','--end-year':'2023'}.items():
        command[command.index(flag)+1]=value
    command+=['--step-days','366']
    monkeypatch.setattr(module,'commands',lambda *args:[command])
    catalog=module.run_grid(root)
    before=catalog.read_bytes()
    assert module.run_grid(root,resume=True).read_bytes()==before
    run=module.resolve_run_path(pd.read_csv(catalog).iloc[0].run_folder,root)
    flow=run/'flows/flows_step_0.parquet'
    flow.write_bytes(flow.read_bytes()+b'tamper')
    with pytest.raises(ValueError,match='hash|changed|manifest'):
        module.run_grid(root,resume=True)


def test_batch_uses_new_mfa_even_when_reference_mfa_is_present(tmp_path,monkeypatch):
    from steel_supply.smoke import toy_network
    from steel_supply.hazard_dynamic import reproduction_grid as module
    monkeypatch.setenv('STEEL_INPUT_MODE','provided-inputs')
    mfa=tmp_path/'result/mfa/2023.parquet'
    mfa.parent.mkdir(parents=True)
    toy_network().to_parquet(mfa,index=False)
    reference=tmp_path/'data/provided/mfa/2023.parquet'
    reference.parent.mkdir(parents=True)
    reference.write_bytes(b'reference only, not a computational input')
    command=module.commands(tmp_path)[0]
    for flag,value in {'--hazard-type':'none','--end-year':'2023'}.items():
        command[command.index(flag)+1]=value
    command+=['--step-days','366']
    monkeypatch.setattr(module,'commands',lambda *args:[command])
    module.run_grid(tmp_path)
    receipt=json.loads((tmp_path/'result/simulation/reproduction/execution_contract.json').read_text())
    assert set(receipt['inputs'])=={'result/mfa/2023.parquet'}
    assert reference.read_bytes()==b'reference only, not a computational input'
