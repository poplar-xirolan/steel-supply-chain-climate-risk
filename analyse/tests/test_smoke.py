import json
from pathlib import Path

import pandas as pd
import pytest


def test_weekly_smoke_uses_real_engine_and_closes_calendar(tmp_path):
    from steel_supply.smoke import run_smoke
    report = run_smoke(tmp_path / 'case')
    assert report['status'] == 'PASS'
    metadata = json.loads(Path(report['run_directory'], 'run_metadata.json').read_text())
    assert len(metadata['steps']) == 53
    assert sum(row['step_days'] for row in metadata['steps']) == 365
    assert metadata['steps'][-1]['step_days'] == 1
    assert list(Path(report['run_directory'], 'flows').glob('*.parquet'))
    with pytest.raises(FileExistsError):
        run_smoke(tmp_path / 'case')
