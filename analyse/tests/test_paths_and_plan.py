"""Behavioural contracts for relocatable scientific inputs and generated artifacts."""
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ANALYSE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ANALYSE / 'library'))


def paths_module():
    entry = ANALYSE / 'library/steel_supply/pipeline_paths.py'
    assert entry.is_file(), 'The exported three-root path implementation does not exist yet'
    return importlib.import_module('steel_supply.pipeline_paths')


def test_generated_and_source_roots_are_separate(tmp_path, monkeypatch):
    monkeypatch.setenv('STEEL_ANALYSE_ROOT', str(tmp_path))
    module = paths_module()
    assert module.data_path('input/mfa/raw.parquet') == tmp_path / 'data/external/mfa/raw.parquet'
    assert module.data_path('work/exposure/a.parquet') == tmp_path / 'work/exposure/a.parquet'
    assert module.data_path('results/mfa/2023.parquet') == tmp_path / 'result/mfa/2023.parquet'
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('legacy_mode', ['provided-inputs', 'provided-summaries'])
def test_reference_results_never_replace_computed_inputs(tmp_path, monkeypatch, legacy_mode):
    monkeypatch.setenv('STEEL_ANALYSE_ROOT', str(tmp_path))
    monkeypatch.setenv('STEEL_INPUT_MODE', legacy_mode)
    module = paths_module()
    reference = tmp_path / 'data/provided/mfa/2023.parquet'
    reference.parent.mkdir(parents=True)
    reference.write_bytes(b'reference only')
    assert module.data_path('results/mfa/2023.parquet') == tmp_path / 'result/mfa/2023.parquet'
    assert module.data_path('results/hazard_country_level_loss/a.parquet') == tmp_path / 'result/hazard_country_level_loss/a.parquet'
    assert module.data_path('results/postprocessing/formal_abm_metrics/x.csv') == tmp_path / 'result/postprocessing/formal_abm_metrics/x.csv'
    assert module.data_path('results/postprocessing/e2eci/x.csv') == tmp_path / 'result/postprocessing/e2eci/x.csv'
    assert reference.read_bytes() == b'reference only'


@pytest.mark.parametrize('legacy_mode', ['provided-inputs', 'provided-summaries'])
def test_retired_starting_points_cannot_bypass_upstream_steps(tmp_path, legacy_mode):
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    completed = subprocess.run(
        [sys.executable, '-B', str(ANALYSE / 'run_analyse.py'), 'plan',
         '--workspace', str(tmp_path), '--mode', legacy_mode],
        capture_output=True, text=True, encoding='utf-8', env=environment,
    )
    assert completed.returncode == 2
    assert not list(tmp_path.iterdir())


def test_default_cli_plan_is_source_only_despite_legacy_environment(tmp_path):
    environment = dict(os.environ, STEEL_INPUT_MODE='provided-summaries', PYTHONDONTWRITEBYTECODE='1')
    completed = subprocess.run(
        [sys.executable, '-B', str(ANALYSE / 'run_analyse.py'), 'plan', '--workspace', str(tmp_path)],
        capture_output=True, text=True, encoding='utf-8', env=environment, check=True,
    )
    plan = json.loads(completed.stdout)
    assert plan['mode'] == 'full'
    assert plan['stages'][0]['id'] == 'trade-clean'
    assert plan['stages'][-1]['id'] == 'export-graphing'
    assert all(path.startswith('data/external/') for path in plan['external_inputs'])
    assert not list(tmp_path.iterdir())


def test_path_cannot_escape_selected_workspace(tmp_path, monkeypatch):
    monkeypatch.setenv('STEEL_ANALYSE_ROOT', str(tmp_path))
    module = paths_module()
    with pytest.raises(ValueError):
        module.data_path('results/../../outside')


def test_full_plan_has_no_required_prebuilt_results():
    entry = ANALYSE / 'library/steel_supply/reproduction/plan.py'
    assert entry.is_file(), 'The exported dependency plan does not exist yet'
    module = importlib.import_module('steel_supply.reproduction.plan')
    stages = module.build_plan()
    external = module.external_inputs(stages)
    assert external and all(item.startswith('data/external/') for item in external)
    assert not any(item.startswith(('result/', 'work/', 'data/provided/')) for item in external)
    assert 'abm' in [s['id'] for s in stages]
    assert 'fig1' not in [s['id'] for s in stages]


def test_climate_locations_are_built_from_original_spatial_sources():
    from steel_supply.reproduction.plan import build_plan,external_inputs
    stages=build_plan()
    assert not any('climate_facilities' in p for p in external_inputs(stages))
    assert 'prepare-facilities' in [s['id'] for s in stages]
