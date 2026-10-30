import json
from pathlib import Path

import pytest


def setup_stage(tmp_path,monkeypatch,body=None):
    from steel_supply.reproduction import execution
    code=tmp_path/'code'
    code.mkdir()
    (code/'run_analyse.py').write_text('# test fixture',encoding='utf-8')
    script=code/'worker.py'
    script.write_text(body or "import os\nfrom pathlib import Path\nr=Path(os.environ['STEEL_ANALYSE_ROOT'])\np=r/'result/value.txt'\np.parent.mkdir(parents=True,exist_ok=True)\np.write_text((r/'data/external/input.txt').read_text())\n",encoding='utf-8')
    root=tmp_path/'case'
    (root/'data/external').mkdir(parents=True)
    (root/'data/external/input.txt').write_text('one',encoding='utf-8')
    monkeypatch.setattr(execution,'bundle_root',lambda:code)
    return execution,root,dict(id='small',script='worker.py',inputs=['data/external/input.txt'],outputs=['result/value.txt'])


def test_resume_binds_inputs_and_outputs_not_just_existing_files(tmp_path,monkeypatch):
    module,root,stage=setup_stage(tmp_path,monkeypatch)
    assert module.run_stage(stage,root)=='ran'
    assert module.run_stage(stage,root,resume=True)=='skipped'
    (root/'data/external/input.txt').write_text('two')
    with pytest.raises(ValueError,match='contract changed'):
        module.run_stage(stage,root,resume=True)
    (root/'data/external/input.txt').write_text('one')
    (root/'result/value.txt').write_text('tamper')
    with pytest.raises(ValueError,match='content changed'):
        module.run_stage(stage,root,resume=True)


def test_failed_child_never_publishes_pass_and_preserves_input(tmp_path,monkeypatch):
    module,root,stage=setup_stage(tmp_path,monkeypatch,"raise RuntimeError('intentional fixture failure')")
    import subprocess
    with pytest.raises(subprocess.CalledProcessError):module.run_stage(stage,root)
    assert not(root/'result/reproduction/small.json').exists()
    assert (root/'result/reproduction/small.pending.json').is_file()
    assert (root/'data/external/input.txt').read_text()=='one'


def test_preflight_missing_data_is_read_only(tmp_path):
    from steel_supply.reproduction.execution import preflight
    from steel_supply.reproduction.plan import build_plan
    result=preflight(tmp_path,build_plan())
    assert result['status']=='FAIL' and result['missing_inputs']
    assert not list(tmp_path.iterdir())


def test_child_environment_cannot_enable_a_retired_starting_point(tmp_path,monkeypatch):
    from steel_supply.reproduction.execution import execution_environment
    monkeypatch.setenv('STEEL_INPUT_MODE','provided-summaries')
    environment=execution_environment(tmp_path)
    assert 'STEEL_INPUT_MODE' not in environment
    assert environment['STEEL_ANALYSE_ROOT']==str(tmp_path.resolve())


def test_reference_mfa_is_not_a_substitute_for_initial_sources(tmp_path):
    from steel_supply.reproduction.execution import preflight
    from steel_supply.reproduction.plan import build_plan
    reference=tmp_path/'data/provided/mfa/2023.parquet'
    reference.parent.mkdir(parents=True)
    reference.write_bytes(b'published reference')
    result=preflight(tmp_path,build_plan(last='mfa'))
    assert result['status']=='FAIL'
    assert 'data/external/mfa/mfa_inputs.xlsx' in result['missing_inputs']
    assert not (tmp_path/'result').exists()


def test_catalogue_requires_a_new_run_not_historical_control(tmp_path):
    from steel_supply.hazard_dynamic.reproduction_grid import load_catalogue
    legacy=tmp_path/'result/simulation/control'
    legacy.mkdir(parents=True)
    (legacy/'scenario_grid.json').write_text('not a current run')
    with pytest.raises(FileNotFoundError,match='reproduction'):
        load_catalogue(tmp_path)


def test_nonempty_raw_directory_is_not_a_complete_collection(tmp_path,monkeypatch):
    from steel_supply.reproduction import execution
    metadata=tmp_path/'code/data/external'
    metadata.mkdir(parents=True)
    (metadata/'required_sources.csv').write_text('file,kind\nisimip/tas/a.nc,climate\nisimip/tas/b.nc,climate\n',encoding='utf-8')
    monkeypatch.setattr(execution,'bundle_root',lambda:tmp_path/'code')
    actual=tmp_path/'case/data/external/isimip/tas'
    actual.mkdir(parents=True)
    (actual/'a.nc').write_bytes(b'source placeholder')
    assert execution.missing_source_files(tmp_path/'case',['data/external/isimip/tas'])==['data/external/isimip/tas/b.nc']
    assert execution.missing_source_files(tmp_path/'case',['data/external/mfa/mfa_inputs.xlsx'])==[]
