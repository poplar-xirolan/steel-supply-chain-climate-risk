"""Declared inputs, immutable sources, and content-verified continuation."""
import fnmatch
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

from ..pipeline_paths import bundle_root, canonical_selector


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    temporary.replace(path)


def artifact_files(root, selector):
    root = Path(root).resolve()
    selector = canonical_selector(selector)
    matches = list(root.glob(selector)) if any(c in selector for c in '*?[') else [root / selector]
    files = []
    for path in matches:
        candidates = path.rglob('*') if path.is_dir() else [path]
        for candidate in candidates:
            if candidate.is_file() and candidate.suffix.lower() not in {'.md', '.pyc'} and candidate.name != '.gitkeep':
                if not candidate.resolve().is_relative_to(root):
                    raise ValueError('Artifact resolves outside the selected workspace')
                files.append(candidate)
    return sorted(set(files))


def fingerprint(root, selectors):
    result = {}
    for selector in selectors:
        paths = artifact_files(root, selector)
        if not paths:
            raise FileNotFoundError(f'Missing or empty artifact: {selector}')
        result.update((p.relative_to(root).as_posix(), sha256(p)) for p in paths)
    return result


def code_fingerprint():
    base = bundle_root()
    paths = [p for folder in ['library', 'config', *[p.name for p in base.iterdir() if p.is_dir() and p.name[:2].isdigit()]]
             for p in (base / folder).rglob('*') if p.is_file() and p.suffix in {'.py', '.json', '.csv'}]
    paths.append(base / 'run_analyse.py')
    return {p.relative_to(base).as_posix(): sha256(p) for p in sorted(set(paths))}


def runtime_signature():
    packages = ['numpy','pandas','scipy','pyarrow','ortools','xarray','dask','netCDF4','geopandas','shapely','openpyxl','pandera']
    result = {}
    for name in packages:
        try: result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: pass
    return {'python': sys.version, 'packages': result}


def execution_environment(root):
    environment = dict(os.environ)
    environment.update(STEEL_ANALYSE_ROOT=str(Path(root).resolve()),
                       PYTHONIOENCODING='utf-8', PYTHONDONTWRITEBYTECODE='1', MPLBACKEND='Agg')
    environment['PYTHONPATH'] = str(bundle_root() / 'library')
    environment.pop('STEEL_DATA_ROOT', None)
    environment.pop('STEEL_INPUT_MODE', None)
    return environment


def missing_source_files(root, selectors):
    """Presence of the declared study collection, not just a nonempty directory."""
    manifest=bundle_root()/'data/external/required_sources.csv'
    if not manifest.is_file():raise FileNotFoundError(manifest)
    missing=[]
    with manifest.open(encoding='utf-8-sig',newline='') as stream:
        for row in csv.DictReader(stream):
            name=canonical_selector('data/external/'+row['file'])
            if any(name==s or name.startswith(s.rstrip('/')+'/') or fnmatch.fnmatch(name,s) for s in selectors):
                if not (Path(root)/name).is_file():missing.append(name)
    return missing


def preflight(root, stages):
    from .plan import external_inputs
    required=external_inputs(stages)
    missing = [s for s in required if not artifact_files(root, s)]
    missing += [s for s in missing_source_files(root,required) if s not in missing]
    imports = ['numpy','pandas','scipy','pyarrow','ortools','geopandas','shapely',
               'openpyxl','loguru','tqdm','pandera','xarray','dask','distributed','netCDF4']
    errors = [f'Missing dependency: {name}' for name in imports if importlib.util.find_spec(name) is None]
    errors += [f'Missing script: {s["script"]}' for s in stages if not s['script'].startswith('@') and not (bundle_root()/s['script']).is_file()]
    if sys.version_info < (3, 12): errors.append('Python >= 3.12 is required')
    return {'status':'PASS' if not missing and not errors else 'FAIL', 'mode':'full',
            'external_inputs':external_inputs(stages), 'missing_inputs':missing, 'environment_errors':errors,
            'note':'Read-only presence/environment check; producing stages enforce scientific schemas and coverage.'}


def run_stage(stage, root, *, resume=False, retry_failed=False):
    root = Path(root).resolve()
    record = root / 'result/reproduction' / (stage['id'] + '.json')
    pending = record.with_suffix('.pending.json')
    before = fingerprint(root, stage['inputs'])
    contract = {'stage':stage['id'], 'mode':'full', 'inputs':before, 'code':code_fingerprint(),
                'environment':runtime_signature(), 'outputs_declared':stage['outputs']}
    if record.exists():
        previous = json.loads(record.read_text(encoding='utf-8'))
        if not resume: raise FileExistsError(f'Existing completed stage: {stage["id"]}; use --resume or another workspace')
        if any(previous.get(k) != v for k,v in contract.items()): raise ValueError('Input/code/runtime contract changed; existing results retained')
        if previous.get('status') != 'PASS' or previous['outputs'] != fingerprint(root, stage['outputs']):
            raise ValueError('Completed output content changed; existing results retained')
        print('Verified; skipping ' + stage['id'], flush=True)
        return 'skipped'
    occupied = [s for s in stage['outputs'] if artifact_files(root,s)]
    if occupied and not stage.get('resumable_attempts'):
        if not (resume and retry_failed and pending.is_file()): raise FileExistsError('Unverified/partial outputs retained; use an explicit managed retry')
        if json.loads(pending.read_text(encoding='utf-8')) != contract: raise ValueError('Pending-stage contract changed')
        destination = root/'work/failed'/stage['id']/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
        for path in sorted(set(p for s in occupied for p in artifact_files(root,s))):
            target = destination/path.relative_to(root)
            if not target.resolve().is_relative_to(root/'work/failed'): raise ValueError('Unsafe failed-output destination')
            target.parent.mkdir(parents=True,exist_ok=True);path.rename(target)
    write_json(pending, contract)
    if stage['script']=='@abm':
        command=[sys.executable,'-X','utf8','-B',str(bundle_root()/'run_analyse.py'),'_abm','--workspace',str(root)]
        if resume:command+=['--resume']
    else:
        command=[sys.executable,'-X','utf8','-B',str(bundle_root()/stage['script']),*stage.get('args',[])]
    if stage['id']=='hazard-combine':command+=['--receipt-path',str(root/'result/hazard_coverage/multi_hazard_max_loss_receipt.json')]
    print('Running '+stage['id'],flush=True)
    subprocess.run(command,cwd=bundle_root(),env=execution_environment(root),check=True)
    if before != fingerprint(root,stage['inputs']):raise ValueError('Stage modified a declared input')
    write_json(record,{**contract,'status':'PASS','outputs':fingerprint(root,stage['outputs'])})
    return 'ran'
