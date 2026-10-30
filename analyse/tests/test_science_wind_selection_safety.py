"""Selection queries must leave extraction outputs untouched."""

from pathlib import Path
import subprocess
import sys

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "03_climate_extraction/05_extract_wind.py"


@pytest.mark.parametrize("existing_output", [True, False])
def test_selection_only_does_not_delete_or_create_output(tmp_path, existing_output):
    output = tmp_path / "wind"
    marker = output / "keep.parquet"
    if existing_output:
        output.mkdir()
        marker.write_bytes(b"existing extraction")
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--selection-only",
         "--facilities-dir", str(tmp_path / "facilities"),
         "--netcdf-dir", str(tmp_path / "netcdf"), "--output-dir", str(output)],
        capture_output=True, text=True, timeout=60,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    assert result.returncode == 0, result.stderr
    if existing_output:
        assert marker.is_file(), "Selection query deleted an existing extraction"
        assert marker.read_bytes() == b"existing extraction"
    else:
        assert not output.exists(), "Selection query created an output directory"
