"""Small, explicitly synthetic exercise of the unchanged scientific engine."""
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd

from .pipeline_paths import bundle_root
from .reproduction.execution import execution_environment, write_json


def toy_network():
    chain = [('P1','M1'),('M1','P2'),('P2','M2'),('M2','P3'),('P3','M3'),
             ('M3','P5'),('P5','M5'),('M5','M6'),('M6','P6'),('P6','M7'),
             ('M7','M8'),('M8','S1')]
    return pd.DataFrame([dict(entity_from='AAA',entity_to='AAA',sector_from=a,
                              sector_to=b,mass=100.,unit='kg',extra=None)
                         for a,b in chain])


def smoke_command(engine, mfa, output, *, inventory=4, capacity=1.1):
    return [sys.executable, '-X', 'utf8', '-B', str(engine),
            '--mfa-path',str(mfa), '--hazard-data-dir',str(mfa.parent/'no-hazards-needed'),
            '--output-dir',str(output), '--start-year','2023','--end-year','2023',
            '--step-days','7','--terminal-step-policy','complete','--fair-match-chunks','100',
            '--hazard-type','none','--climate-scenario','ssp370','--trade-scenario','free-trade',
            '--inventory-to-capacity-ratio',str(inventory),'--capacity-to-initial-flow',str(capacity)]


def run_smoke(destination):
    destination = Path(destination).resolve()
    if destination.exists() and any(destination.iterdir()):
        raise FileExistsError('Smoke output already exists; choose an empty directory')
    destination.mkdir(parents=True,exist_ok=True)
    mfa = destination/'synthetic_mfa.parquet'
    toy_network().to_parquet(mfa,index=False)
    command = smoke_command(bundle_root()/'06_dynamic_simulation/11_run_simulation.py',
                            mfa,destination/'trajectory')
    completed = subprocess.run(command,env=execution_environment(destination),
                               capture_output=True,text=True,encoding='utf-8')
    (destination/'engine.log').write_text(completed.stdout+completed.stderr,encoding='utf-8')
    if completed.returncode:
        raise RuntimeError(f'Smoke engine failed: see {destination / "engine.log"}')
    runs = list((destination/'trajectory').rglob('run_metadata.json'))
    if len(runs) != 1:
        raise ValueError('Expected exactly one synthetic trajectory')
    metadata=json.loads(runs[0].read_text(encoding='utf-8'))
    steps=metadata['steps']
    if len(steps)!=53 or sum(s['step_days'] for s in steps)!=365 or steps[-1]['step_days']!=1:
        raise ValueError('Incomplete weekly calendar in smoke output')
    report={'status':'PASS','kind':'synthetic test, not a manuscript scenario',
            'steps':len(steps),'days':365,'run_directory':str(runs[0].parent)}
    write_json(destination/'smoke_report.json',report)
    return report
