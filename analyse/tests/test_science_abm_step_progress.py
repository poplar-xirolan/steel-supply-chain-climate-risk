from __future__ import annotations

from datetime import date
import json
from pathlib import Path
import sys

import pandas as pd
import pytest


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from steel_supply.hazard_dynamic import abm_step_progress as progress  # noqa: E402
from steel_supply.hazard_dynamic.simulation_runtime import SimulationStep  # noqa: E402


def first_step() -> SimulationStep:
    return SimulationStep(0, date(2023, 1, 1), date(2023, 1, 8), 7, 7 / 365.25)


def test_marker_is_published_only_after_both_parquets_are_durable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A premature marker would falsely advertise a fully published step."""

    observed: list[str] = []
    real_atomic_write_parquet = progress.atomic_write_parquet
    real_atomic_write_json = progress.atomic_write_json

    def record_parquet(frame: pd.DataFrame, destination: Path, **kwargs) -> dict[str, object]:
        result = real_atomic_write_parquet(frame, destination, **kwargs)
        observed.append(destination.name)
        return result

    def record_marker(destination: Path, payload: dict[str, object]) -> None:
        real_atomic_write_json(destination, payload)
        observed.append(destination.name)

    monkeypatch.setattr(progress, "atomic_write_parquet", record_parquet)
    monkeypatch.setattr(progress, "atomic_write_json", record_marker)

    progress.publish_step_outputs(
        run_root=tmp_path,
        current_step=first_step(),
        flow_frame=pd.DataFrame({"mass": [1.0]}),
        agent_frame=pd.DataFrame({"capacity": [2.0]}),
    )

    assert observed == ["flows_step_0.parquet", "agents_step_0.parquet", "step_progress.json"]
    assert not list(tmp_path.rglob("*.tmp"))


def test_failed_second_parquet_does_not_advance_progress(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An agent-write interruption must not create a completed-step marker."""

    calls = 0
    real_atomic_write_parquet = progress.atomic_write_parquet

    def fail_on_agents(frame: pd.DataFrame, destination: Path, **kwargs) -> dict[str, object]:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected agent write failure")
        return real_atomic_write_parquet(frame, destination, **kwargs)

    monkeypatch.setattr(progress, "atomic_write_parquet", fail_on_agents)

    with pytest.raises(OSError, match="injected agent write failure"):
        progress.publish_step_outputs(
            run_root=tmp_path,
            current_step=first_step(),
            flow_frame=pd.DataFrame({"mass": [1.0]}),
            agent_frame=pd.DataFrame({"capacity": [2.0]}),
        )

    assert not (tmp_path / "step_progress.json").exists()
    assert not list(tmp_path.rglob("*.tmp"))


def test_empty_flow_buffer_marks_flows_absent_without_creating_a_zero_row_file(
    tmp_path: Path,
) -> None:
    """A synthetic empty flow parquet would change the historical telemetry surface."""

    progress.publish_step_outputs(
        run_root=tmp_path,
        current_step=first_step(),
        flow_frame=None,
        agent_frame=pd.DataFrame({"capacity": [2.0]}),
    )

    marker = json.loads((tmp_path / "step_progress.json").read_text(encoding="utf-8"))
    assert marker["flows"] is None
    assert marker["agents"]["rows"] == 1
    assert not (tmp_path / "flows" / "flows_step_0.parquet").exists()


def test_atomic_parquet_removes_temporary_file_when_serialization_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failed serialization must not leave a stale file that resembles work in progress."""

    def fail_to_parquet(self, *args, **kwargs) -> None:
        raise OSError("injected parquet failure")

    monkeypatch.setattr(pd.DataFrame, "to_parquet", fail_to_parquet)

    with pytest.raises(OSError, match="injected parquet failure"):
        progress.atomic_write_parquet(pd.DataFrame({"mass": [1.0]}), tmp_path / "output.parquet")

    assert not list(tmp_path.rglob("*.tmp"))
