"""Shared crash-safe ABM telemetry publication and display-only step marker.

``step_progress.json`` is deliberately not a scheduler input: it records only
the latest successfully published telemetry pair for status displays.
"""

from __future__ import annotations
from steel_supply.pipeline_paths import science_path, bundle_root, data_path

from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import tempfile
from typing import Callable

import pandas as pd

from .run_contract import atomic_write_json
from .simulation_runtime import SimulationStep


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def atomic_write_parquet(
    frame: pd.DataFrame,
    destination: Path,
    *,
    replace: Callable[[str | Path, str | Path], None] = os.replace,
) -> dict[str, object]:
    """Durably publish one existing telemetry parquet via a sibling temporary."""

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=destination.parent,
        prefix=f".{destination.name}.{os.getpid()}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    os.close(descriptor)
    try:
        frame.to_parquet(temporary, index=False)
        with temporary.open("rb+") as handle:
            os.fsync(handle.fileno())
        replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return {"path": str(destination), "rows": len(frame), "sha256": sha256_file(destination)}


def publish_completed_step(
    run_root: Path,
    step: SimulationStep,
    flows: dict[str, object] | None,
    agents: dict[str, object],
) -> None:
    """Publish the non-authoritative marker after telemetry outputs are durable."""

    atomic_write_json(
        Path(run_root) / "step_progress.json",
        {
            "contract_version": 1,
            "completed_step": step.index,
            "completed_steps": step.index + 1,
            "step_start_date": step.start.isoformat(),
            "step_end_date_exclusive": step.end_exclusive.isoformat(),
            "flows": flows,
            "agents": agents,
            "updated_at_utc": utc_now(),
        },
    )


def publish_step_outputs(
    run_root: Path,
    current_step: SimulationStep,
    flow_frame: pd.DataFrame | None,
    agent_frame: pd.DataFrame,
    *,
    replace: Callable[[str | Path, str | Path], None] = os.replace,
) -> None:
    """Publish one complete telemetry step, with marker last.

    This convenience API is used by focused tests; the engine keeps its
    historical buffer-clearing points while calling the same two primitives.
    """

    run_root = Path(run_root)
    flow_metadata = None
    if flow_frame is not None:
        flow_metadata = atomic_write_parquet(
            flow_frame,
            run_root / "flows" / f"flows_step_{current_step.index}.parquet",
            replace=replace,
        )
    agent_metadata = atomic_write_parquet(
        agent_frame,
        run_root / "agents" / f"agents_step_{current_step.index}.parquet",
        replace=replace,
    )
    publish_completed_step(run_root, current_step, flow_metadata, agent_metadata)

