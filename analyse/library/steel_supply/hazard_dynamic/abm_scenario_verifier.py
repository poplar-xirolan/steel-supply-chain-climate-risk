"""Shared fail-closed output verification for one dispatched ABM attempt."""

from __future__ import annotations
from steel_supply.pipeline_paths import science_path, bundle_root, data_path

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd

from .run_contract import (
    ScenarioSpec,
    atomic_write_json,
    canonical_argv_sha256,
)
from .simulation_runtime import SimulationStep, build_step_schedule


AGENT_COLUMNS = {
    "step", "year", "sector", "country", "capacity", "capacity_left",
    "inventory", "stage", "step_start_date", "step_end_date_exclusive",
    "step_days", "time_scale_ratio", "random_seed", "terminal_step_policy",
}
FLOW_COLUMNS = {
    "entity_from", "entity_to", "sector_from", "sector_to", "mass", "unit",
    "extra", "step", "year", "step_start_date", "step_end_date_exclusive",
    "step_days", "time_scale_ratio", "random_seed", "terminal_step_policy",
}
MANIFEST_NAME = "output_manifest.json"
PASS_NAME = "PASS.json"
_STEP_FILE = re.compile(r"(?P<kind>agents|flows)_step_(?P<index>\d+)\.parquet\Z")
_ATTEMPT_ROOT = re.compile(r"attempt-(?P<number>\d{4})\Z")
_PASS_KEYS = {
    "contract_version", "status", "scenario_key", "scenario_ordinal",
    "scenario_argv_sha256", "attempt", "executed_argv_sha256", "run_directory",
    "manifest_path", "manifest_sha256", "agent_files", "flow_files",
    "completed_steps", "total_rows_agents", "total_rows_flows", "completed_at_utc",
}


@dataclass(frozen=True, slots=True)
class VerificationSummary:
    run_directory: Path
    metadata_path: Path
    manifest_path: Path
    manifest_sha256: str
    agent_files: int
    flow_files: int
    completed_steps: int
    total_rows_agents: int
    total_rows_flows: int


def verify_engine_output(
    spec: ScenarioSpec,
    attempt_root: Path,
    executed_argv: Sequence[str],
) -> VerificationSummary:
    """Scan one completed engine attempt and write its only manifest artifact."""

    attempt = Path(attempt_root).resolve()
    argv = tuple(executed_argv)
    _parse_attempt_number(attempt)
    _require_executed_argv_matches_spec(spec, attempt, argv)
    expected = _expected_schedule(argv)
    _require_engine_output_binding(attempt, argv)
    manifest_path = attempt / MANIFEST_NAME
    if manifest_path.exists():
        raise ValueError("output manifest already exists for this attempt")
    _reject_temporary_files(attempt)
    run_directory = _find_run_directory(attempt)
    metadata_path = run_directory / "run_metadata.json"
    _validate_metadata(metadata_path, expected, argv)

    agent_entries, agent_rows = _validate_step_files(
        run_directory, "agents", AGENT_COLUMNS, expected, argv, required=True
    )
    flow_entries, flow_rows = _validate_step_files(
        run_directory, "flows", FLOW_COLUMNS, expected, argv, required=False
    )
    entries = [_manifest_entry(metadata_path, run_directory, 0), *agent_entries, *flow_entries]
    entries.sort(key=lambda entry: str(entry["relative_path"]))
    counts = {
        "agent_files": len(agent_entries),
        "flow_files": len(flow_entries),
        "completed_steps": len(expected),
        "total_rows_agents": agent_rows,
        "total_rows_flows": flow_rows,
    }
    manifest = {
        "contract_version": 1,
        "scenario_key": spec.key,
        "scenario_ordinal": spec.ordinal,
        "scenario_argv_sha256": spec.argv_sha256,
        "executed_argv_sha256": canonical_argv_sha256(argv),
        "run_directory": str(run_directory.relative_to(attempt).as_posix()),
        "metadata_path": str(metadata_path.relative_to(run_directory).as_posix()),
        "files": entries,
        "counts": counts,
    }
    atomic_write_json(manifest_path, manifest, exclusive=True)
    return VerificationSummary(
        run_directory=run_directory,
        metadata_path=metadata_path,
        manifest_path=manifest_path,
        manifest_sha256=_sha256_file(manifest_path),
        **counts,
    )


def validate_pass_receipt(
    spec: ScenarioSpec,
    attempt_root: Path,
    *,
    rehash_outputs: bool = False,
) -> Mapping[str, object]:
    """Validate the dispatcher-written PASS receipt without granting a PASS."""

    attempt = Path(attempt_root).resolve()
    expected_attempt = _parse_attempt_number(attempt)
    receipt_path = attempt / PASS_NAME
    receipt = _read_json_object(receipt_path, "PASS receipt")
    if set(receipt) != _PASS_KEYS:
        raise ValueError("PASS receipt schema differs from the versioned contract")
    _require_equal(receipt["contract_version"], 1, "PASS contract version")
    _require_equal(receipt["status"], "PASS", "PASS status")
    _require_equal(receipt["scenario_key"], spec.key, "PASS scenario")
    _require_equal(receipt["scenario_ordinal"], spec.ordinal, "PASS scenario")
    _require_equal(receipt["scenario_argv_sha256"], spec.argv_sha256, "PASS scenario")
    if isinstance(receipt["attempt"], bool) or not isinstance(receipt["attempt"], int) or receipt["attempt"] < 1:
        raise ValueError("PASS attempt must be a positive integer")
    _require_equal(receipt["attempt"], expected_attempt, "PASS attempt")
    _validate_completed_at(receipt["completed_at_utc"])
    manifest_path = attempt / MANIFEST_NAME
    _require_equal(receipt["manifest_path"], MANIFEST_NAME, "PASS manifest path")
    _require_equal(receipt["manifest_sha256"], _sha256_file(manifest_path), "PASS manifest hash")
    manifest = _read_json_object(manifest_path, "output manifest")
    _validate_manifest_against_receipt(manifest, receipt, attempt, rehash_outputs)
    return receipt


def _expected_schedule(argv: Sequence[str]) -> list[SimulationStep]:
    values = _argv_values(argv)
    try:
        start_year = int(values.get("--start-year", "2023"))
        end_year = int(values.get("--end-year", "2100"))
        step_days = int(values.get("--step-days", "7"))
    except ValueError as error:
        raise ValueError("executed argv has non-integer schedule values") from error
    terminal_policy = values.get("--terminal-step-policy", "complete")
    if terminal_policy != "complete":
        raise ValueError("executed argv terminal policy must be complete")
    if start_year > end_year:
        raise ValueError("executed argv schedule has start year after end year")
    return build_step_schedule(
        date(start_year, 1, 1), date(end_year, 12, 31), step_days,
        complete_terminal_step=True,
    )


def _argv_values(argv: Sequence[str]) -> dict[str, str]:
    if not argv or any(not isinstance(value, str) or not value for value in argv):
        raise ValueError("executed argv must contain non-empty strings")
    values: dict[str, str] = {}
    flags = {"--output-dir", "--start-year", "--end-year", "--step-days", "--random-seed", "--terminal-step-policy"}
    for index, value in enumerate(argv):
        if value not in flags:
            continue
        if value in values or index + 1 >= len(argv) or argv[index + 1].startswith("--"):
            raise ValueError(f"executed argv has invalid {value}")
        values[value] = argv[index + 1]
    if "--output-dir" not in values:
        raise ValueError("executed argv has no output directory")
    if "--random-seed" in values:
        try:
            int(values["--random-seed"])
        except ValueError as error:
            raise ValueError("executed argv has invalid random seed") from error
    return values


def _require_engine_output_binding(attempt: Path, argv: Sequence[str]) -> None:
    output_directory = Path(_argv_values(argv)["--output-dir"]).resolve()
    if output_directory != attempt / "engine_output":
        raise ValueError("executed argv output directory does not bind this attempt")


def _require_executed_argv_matches_spec(
    spec: ScenarioSpec, attempt: Path, argv: Sequence[str],
) -> None:
    expected = list(spec.argv)
    locations = [index for index, value in enumerate(expected) if value == "--output-dir"]
    if len(locations) != 1 or locations[0] + 1 >= len(expected):
        raise ValueError("ScenarioSpec argv must contain exactly one output directory")
    expected[locations[0] + 1] = str((attempt / "engine_output").resolve())
    if tuple(argv) != tuple(expected):
        raise ValueError("executed argv differs from ScenarioSpec outside the attempt output directory")


def _parse_attempt_number(attempt: Path) -> int:
    match = _ATTEMPT_ROOT.fullmatch(attempt.name)
    if match is None:
        raise ValueError("attempt root must be named attempt-<four-digit-positive-integer>")
    number = int(match["number"])
    if number < 1:
        raise ValueError("attempt root number must be positive")
    return number


def _find_run_directory(attempt: Path) -> Path:
    output = attempt / "engine_output"
    if not output.is_dir():
        raise ValueError("attempt has no engine_output directory")
    children = [path for path in output.iterdir() if path.is_dir()]
    if len(children) != 1:
        raise ValueError("engine_output must contain exactly one engine-created run directory")
    if any(path.is_file() for path in output.iterdir()):
        raise ValueError("engine_output contains an unexpected direct file")
    return children[0].resolve()


def _validate_metadata(path: Path, schedule: Sequence[SimulationStep], argv: Sequence[str]) -> None:
    metadata = _read_json_object(path, "run metadata")
    if set(metadata) != {"random_seed", "terminal_step_policy", "steps"}:
        raise ValueError("metadata schema is malformed")
    values = _argv_values(argv)
    expected_seed: int | None = int(values["--random-seed"]) if "--random-seed" in values else None
    if metadata["random_seed"] != expected_seed:
        raise ValueError("metadata random seed differs from executed argv")
    if metadata["terminal_step_policy"] != "complete":
        raise ValueError("metadata terminal policy differs from executed argv")
    expected_steps = [
        {"index": step.index, "start_date": step.start.isoformat(), "end_date_exclusive": step.end_exclusive.isoformat(), "step_days": step.days, "time_scale_ratio": step.time_scale_ratio}
        for step in schedule
    ]
    if metadata["steps"] != expected_steps:
        raise ValueError("metadata schedule differs from executed argv")


def _validate_step_files(
    run_directory: Path, kind: str, columns: set[str], schedule: Sequence[SimulationStep],
    argv: Sequence[str], *, required: bool,
) -> tuple[list[dict[str, object]], int]:
    directory = run_directory / kind
    if not directory.exists():
        if required:
            raise ValueError("agent inventory directory is missing")
        return [], 0
    if not directory.is_dir():
        raise ValueError(f"{kind} inventory is not a directory")
    files = sorted(directory.iterdir(), key=lambda path: path.name)
    matched: dict[int, Path] = {}
    for path in files:
        match = _STEP_FILE.fullmatch(path.name)
        if not path.is_file() or match is None or match["kind"] != kind:
            raise ValueError(f"{kind} inventory contains an unexpected file")
        index = int(match["index"])
        if index in matched or index >= len(schedule):
            raise ValueError(f"{kind.rstrip('s')} inventory has an invalid step index")
        matched[index] = path
    if required and set(matched) != set(range(len(schedule))):
        raise ValueError("agent inventory does not cover every expected step")
    entries: list[dict[str, object]] = []
    total_rows = 0
    for index, path in sorted(matched.items()):
        frame = _read_parquet(path, kind)
        if set(frame.columns) != columns:
            raise ValueError(f"{kind} schema differs from the exact required columns")
        _validate_frame(frame, kind, schedule[index], argv)
        entries.append(_manifest_entry(path, run_directory, len(frame)))
        total_rows += len(frame)
    return entries, total_rows


def _read_parquet(path: Path, kind: str) -> pd.DataFrame:
    try:
        return pd.read_parquet(path)
    except Exception as error:
        raise ValueError(f"{kind} parquet cannot be read: {path.name}") from error


def _validate_frame(frame: pd.DataFrame, kind: str, step: SimulationStep, argv: Sequence[str]) -> None:
    expected_seed = _argv_values(argv).get("--random-seed")
    expected_seed_value: int | None = int(expected_seed) if expected_seed is not None else None
    context = {
        "step": step.index, "year": step.start.year, "step_start_date": step.start.isoformat(),
        "step_end_date_exclusive": step.end_exclusive.isoformat(), "step_days": step.days,
        "time_scale_ratio": step.time_scale_ratio, "random_seed": expected_seed_value,
        "terminal_step_policy": "complete",
    }
    for name, expected in context.items():
        series = frame[name]
        valid = series.isna().all() if expected is None else series.eq(expected).all()
        if not bool(valid):
            raise ValueError(f"{kind} context differs for {name}")
    numeric = ["mass"] if kind == "flows" else ["capacity", "capacity_left", "inventory"]
    for name in numeric:
        if not pd.api.types.is_numeric_dtype(frame[name]):
            raise ValueError(f"{kind} {name} must be numeric")
        values = frame[name].to_numpy(dtype=float, copy=False)
        if not np.isfinite(values).all():
            raise ValueError(f"{kind} {name} must be finite")
        if (values < 0).any():
            raise ValueError(f"{kind} {name} must be nonnegative")
    if kind == "agents":
        if set(frame["stage"].unique()) != {"START", "END"}:
            raise ValueError("agents stages must be exactly START and END")
        if (frame["capacity_left"].to_numpy(dtype=float) > frame["capacity"].to_numpy(dtype=float) + 1e-9).any():
            raise ValueError("agents capacity_left exceeds capacity")


def _manifest_entry(path: Path, run_directory: Path, rows: int) -> dict[str, object]:
    return {
        "relative_path": path.relative_to(run_directory).as_posix(),
        "size_bytes": path.stat().st_size,
        "row_count": rows,
        "sha256": _sha256_file(path),
    }


def _reject_temporary_files(root: Path) -> None:
    if not root.is_dir():
        raise ValueError("attempt root does not exist")
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() == ".tmp":
            raise ValueError(f"temporary output exists: {path.name}")


def _validate_manifest_against_receipt(
    manifest: Mapping[str, object], receipt: Mapping[str, object], attempt: Path,
    rehash_outputs: bool,
) -> None:
    required = {"contract_version", "scenario_key", "scenario_ordinal", "scenario_argv_sha256", "executed_argv_sha256", "run_directory", "metadata_path", "files", "counts"}
    if set(manifest) != required:
        raise ValueError("output manifest schema differs from the versioned contract")
    for field in ("scenario_key", "scenario_ordinal", "scenario_argv_sha256", "executed_argv_sha256"):
        _require_equal(manifest[field], receipt[field], f"PASS {field}")
    if not isinstance(manifest["run_directory"], str) or Path(manifest["run_directory"]).is_absolute():
        raise ValueError("manifest run directory must be relative")
    _require_equal(receipt["run_directory"], manifest["run_directory"], "PASS run directory")
    run_directory = attempt / str(manifest["run_directory"])
    if not run_directory.is_dir() or run_directory.parent != attempt / "engine_output":
        raise ValueError("PASS run directory does not bind the attempt")
    _require_equal(manifest["metadata_path"], "run_metadata.json", "manifest metadata path")
    counts = manifest["counts"]
    if not isinstance(counts, dict):
        raise ValueError("manifest counts are malformed")
    for field in ("agent_files", "flow_files", "completed_steps", "total_rows_agents", "total_rows_flows"):
        _require_equal(receipt[field], counts.get(field), f"PASS {field}")
        if isinstance(receipt[field], bool) or not isinstance(receipt[field], int) or receipt[field] < 0:
            raise ValueError(f"PASS {field} must be a nonnegative integer")
    _validate_manifest_files(manifest, run_directory, rehash_outputs)


def _validate_manifest_files(
    manifest: Mapping[str, object], run_directory: Path, rehash_outputs: bool,
) -> None:
    files = manifest["files"]
    if not isinstance(files, list) or not files:
        raise ValueError("manifest file inventory is malformed")
    previous = ""
    agent_indices: set[int] = set()
    flow_indices: set[int] = set()
    agent_rows = 0
    flow_rows = 0
    has_metadata = False
    for entry in files:
        if not isinstance(entry, dict) or set(entry) != {"relative_path", "size_bytes", "row_count", "sha256"}:
            raise ValueError("manifest file entry is malformed")
        relative = entry["relative_path"]
        if not isinstance(relative, str) or Path(relative).is_absolute() or ".." in Path(relative).parts or relative <= previous:
            raise ValueError("manifest file ordering or path is invalid")
        previous = relative
        path = run_directory / relative
        if (
            isinstance(entry["size_bytes"], bool)
            or not isinstance(entry["size_bytes"], int)
            or entry["size_bytes"] < 0
            or isinstance(entry["row_count"], bool)
            or not isinstance(entry["row_count"], int)
            or entry["row_count"] < 0
            or not isinstance(entry["sha256"], str)
            or len(entry["sha256"]) != 64
        ):
            raise ValueError("manifest file entry has invalid recorded values")
        if not path.is_file() or path.stat().st_size != entry["size_bytes"]:
            raise ValueError("manifest size does not match scientific output")
        if rehash_outputs and _sha256_file(path) != entry["sha256"]:
            raise ValueError("manifest hash does not match scientific output")
        if relative == "run_metadata.json":
            if entry["row_count"] != 0:
                raise ValueError("manifest metadata row count is invalid")
            has_metadata = True
            continue
        match = _STEP_FILE.fullmatch(Path(relative).name)
        if match is None or Path(relative).parent.name != match["kind"]:
            raise ValueError("manifest contains an unexpected scientific path")
        index = int(match["index"])
        if match["kind"] == "agents":
            agent_indices.add(index)
            agent_rows += int(entry["row_count"])
        else:
            flow_indices.add(index)
            flow_rows += int(entry["row_count"])
    counts = manifest["counts"]
    assert isinstance(counts, dict)
    if not has_metadata:
        raise ValueError("manifest metadata entry is missing")
    completed = counts["completed_steps"]
    if set(agent_indices) != set(range(completed)):
        raise ValueError("manifest agent inventory does not cover every completed step")
    if any(index >= completed for index in flow_indices):
        raise ValueError("manifest flow inventory has an invalid step index")
    if (
        counts["agent_files"] != len(agent_indices)
        or counts["flow_files"] != len(flow_indices)
        or counts["total_rows_agents"] != agent_rows
        or counts["total_rows_flows"] != flow_rows
    ):
        raise ValueError("manifest recorded step or file counts are inconsistent")


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"{label} cannot be read") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object")
    return payload


def _validate_completed_at(value: object) -> None:
    if not isinstance(value, str):
        raise ValueError("PASS completion time is malformed")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise ValueError("PASS completion time is malformed") from error
    if parsed.tzinfo is None:
        raise ValueError("PASS completion time must include a timezone")


def _require_equal(actual: object, expected: object, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label} differs from the expected value")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

