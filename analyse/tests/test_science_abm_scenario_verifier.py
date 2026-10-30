from __future__ import annotations

from datetime import date
import hashlib
import importlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


MODULE_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODULE_DIR))

from steel_supply.hazard_dynamic.run_contract import ScenarioSpec  # noqa: E402

try:
    verifier = importlib.import_module("steel_supply.hazard_dynamic.abm_scenario_verifier")
except ModuleNotFoundError:
    verifier = None


def require_verifier() -> None:
    assert verifier is not None, "abm_scenario_verifier must provide the PASS boundary"


def argv_for(output_directory: Path) -> tuple[str, ...]:
    return (
        "python",
        "11_run_simulation.py",
        "--hazard-type", "none",
        "--climate-scenario", "picontrol",
        "--trade-scenario", "random-disruption",
        "--inventory-to-capacity-ratio", "1",
        "--capacity-to-initial-flow", "1",
        "--output-dir", str(output_directory),
        "--start-year", "2023",
        "--end-year", "2024",
        "--step-days", "366",
        "--random-seed", "17",
        "--terminal-step-policy", "complete",
    )


def agent_rows(index: int, start: str, end: str, days: int) -> pd.DataFrame:
    rows = []
    for stage in ("START", "END"):
        rows.append(
            {
                "step": index,
                "year": int(start[:4]),
                "sector": "P1",
                "country": "AAA",
                "capacity": 10.0,
                "capacity_left": 5.0,
                "inventory": 2.0,
                "stage": stage,
                "step_start_date": start,
                "step_end_date_exclusive": end,
                "step_days": days,
                "time_scale_ratio": days / 365.25,
                "random_seed": 17,
                "terminal_step_policy": "complete",
            }
        )
    return pd.DataFrame(rows)


def flow_rows(index: int, start: str, end: str, days: int) -> pd.DataFrame:
    return pd.DataFrame(
        [{
            "entity_from": "AAA", "entity_to": "BBB", "sector_from": "P1",
            "sector_to": "M1", "mass": 3.0, "unit": "kg", "extra": False,
            "step": index, "year": int(start[:4]), "step_start_date": start,
            "step_end_date_exclusive": end, "step_days": days,
            "time_scale_ratio": days / 365.25, "random_seed": 17,
            "terminal_step_policy": "complete",
        }]
    )


def write_attempt(tmp_path: Path) -> tuple[ScenarioSpec, Path, tuple[str, ...], Path]:
    attempt = tmp_path / "attempt-0001"
    engine_output = attempt / "engine_output"
    run_directory = engine_output / "engine-created-run"
    agents = run_directory / "agents"
    flows = run_directory / "flows"
    agents.mkdir(parents=True)
    flows.mkdir()
    spec = ScenarioSpec.from_argv(1, argv_for(tmp_path / "planned-formal-output"))
    argv = list(spec.argv)
    argv[argv.index("--output-dir") + 1] = str(engine_output.resolve())
    metadata = {
        "random_seed": 17,
        "terminal_step_policy": "complete",
        "steps": [
            {"index": 0, "start_date": "2023-01-01", "end_date_exclusive": "2024-01-02", "step_days": 366, "time_scale_ratio": 366 / 365.25},
            {"index": 1, "start_date": "2024-01-02", "end_date_exclusive": "2025-01-01", "step_days": 365, "time_scale_ratio": 365 / 365.25},
        ],
    }
    (run_directory / "run_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    agent_rows(0, "2023-01-01", "2024-01-02", 366).to_parquet(agents / "agents_step_0.parquet", index=False)
    agent_rows(1, "2024-01-02", "2025-01-01", 365).to_parquet(agents / "agents_step_1.parquet", index=False)
    flow_rows(0, "2023-01-01", "2024-01-02", 366).to_parquet(flows / "flows_step_0.parquet", index=False)
    return spec, attempt, tuple(argv), run_directory


def pass_receipt(spec: ScenarioSpec, attempt: Path) -> dict[str, object]:
    manifest = attempt / "output_manifest.json"
    contents = json.loads(manifest.read_text(encoding="utf-8"))
    return {
        "contract_version": 1,
        "status": "PASS",
        "scenario_key": spec.key,
        "scenario_ordinal": spec.ordinal,
        "scenario_argv_sha256": spec.argv_sha256,
        "attempt": 1,
        "executed_argv_sha256": contents["executed_argv_sha256"],
        "run_directory": contents["run_directory"],
        "manifest_path": "output_manifest.json",
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        **contents["counts"],
        "completed_at_utc": "2026-08-14T00:00:00Z",
    }


def test_complete_two_step_attempt_produces_sorted_manifest_and_valid_receipt(tmp_path: Path) -> None:
    """Removing a complete required agent output must make this PASS boundary fail."""
    require_verifier()
    spec, attempt, argv, run_directory = write_attempt(tmp_path)

    summary = verifier.verify_engine_output(spec, attempt, argv)

    assert summary.run_directory == run_directory.resolve()
    assert (attempt / "output_manifest.json").exists()
    manifest = json.loads((attempt / "output_manifest.json").read_text(encoding="utf-8"))
    assert [entry["relative_path"] for entry in manifest["files"]] == sorted(entry["relative_path"] for entry in manifest["files"])
    assert manifest["counts"] == {"agent_files": 2, "completed_steps": 2, "flow_files": 1, "total_rows_agents": 4, "total_rows_flows": 1}
    (attempt / "PASS.json").write_text(json.dumps(pass_receipt(spec, attempt)), encoding="utf-8")
    assert verifier.validate_pass_receipt(spec, attempt, rehash_outputs=True)["status"] == "PASS"


def test_missing_agent_step_fails_agent_inventory_check(tmp_path: Path) -> None:
    """A dropped terminal agent parquet must not be mistaken for an empty step."""
    require_verifier()
    spec, attempt, argv, run_directory = write_attempt(tmp_path)
    (run_directory / "agents" / "agents_step_1.parquet").unlink()

    with pytest.raises(ValueError, match="agent inventory"):
        verifier.verify_engine_output(spec, attempt, argv)


def test_executed_argv_seed_mutation_is_rejected_even_when_outputs_match_it(tmp_path: Path) -> None:
    """Changing the dispatched seed must not escape the ScenarioSpec identity."""
    require_verifier()
    spec, attempt, argv, run_directory = write_attempt(tmp_path)
    mutated = list(argv)
    mutated[mutated.index("--random-seed") + 1] = "999"
    metadata_path = run_directory / "run_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["random_seed"] = 999
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    for path in [
        *(run_directory / "agents").glob("agents_step_*.parquet"),
        *(run_directory / "flows").glob("flows_step_*.parquet"),
    ]:
        frame = pd.read_parquet(path)
        frame["random_seed"] = 999
        frame.to_parquet(path, index=False)

    with pytest.raises(ValueError, match="executed argv|ScenarioSpec"):
        verifier.verify_engine_output(spec, attempt, tuple(mutated))


def test_manifest_or_output_drift_fails_full_rehash_validation(tmp_path: Path) -> None:
    """Changing a scientific parquet after scanning must invalidate a PASS receipt."""
    require_verifier()
    spec, attempt, argv, run_directory = write_attempt(tmp_path)
    verifier.verify_engine_output(spec, attempt, argv)
    (attempt / "PASS.json").write_text(json.dumps(pass_receipt(spec, attempt)), encoding="utf-8")
    frame = pd.read_parquet(run_directory / "flows" / "flows_step_0.parquet")
    frame.loc[0, "mass"] = 4.0
    frame.to_parquet(run_directory / "flows" / "flows_step_0.parquet", index=False)

    with pytest.raises(ValueError, match="manifest|hash"):
        verifier.validate_pass_receipt(spec, attempt, rehash_outputs=True)


def test_pass_receipt_always_checks_recorded_output_sizes(tmp_path: Path) -> None:
    """A non-audit receipt check must still reject a changed recorded file size."""
    require_verifier()
    spec, attempt, argv, run_directory = write_attempt(tmp_path)
    verifier.verify_engine_output(spec, attempt, argv)
    (attempt / "PASS.json").write_text(json.dumps(pass_receipt(spec, attempt)), encoding="utf-8")
    path = run_directory / "flows" / "flows_step_0.parquet"
    frame = pd.read_parquet(path)
    changed = pd.concat([frame] * 100, ignore_index=True)
    changed["mass"] = np.arange(100, dtype=float)
    changed.to_parquet(path, index=False)

    with pytest.raises(ValueError, match="size|manifest"):
        verifier.validate_pass_receipt(spec, attempt)


def test_pass_receipt_requires_manifest_metadata_entry(tmp_path: Path) -> None:
    """A signed count list without the engine metadata file is not a scientific receipt."""
    require_verifier()
    spec, attempt, argv, _ = write_attempt(tmp_path)
    verifier.verify_engine_output(spec, attempt, argv)
    manifest_path = attempt / "output_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"] = [entry for entry in manifest["files"] if entry["relative_path"] != "run_metadata.json"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (attempt / "PASS.json").write_text(json.dumps(pass_receipt(spec, attempt)), encoding="utf-8")

    with pytest.raises(ValueError, match="metadata"):
        verifier.validate_pass_receipt(spec, attempt)


def test_duplicate_run_directories_are_rejected(tmp_path: Path) -> None:
    """A second engine-created child makes run provenance ambiguous."""
    require_verifier()
    spec, attempt, argv, _ = write_attempt(tmp_path)
    (attempt / "engine_output" / "second-run").mkdir()

    with pytest.raises(ValueError, match="exactly one"):
        verifier.verify_engine_output(spec, attempt, argv)


@pytest.mark.parametrize("field, value", [("random_seed", 18), ("terminal_step_policy", "legacy-floor"), ("steps", [])])
def test_wrong_metadata_seed_policy_or_schedule_is_rejected(tmp_path: Path, field: str, value: object) -> None:
    """Metadata must bind the engine outputs to the executed simulation schedule."""
    require_verifier()
    spec, attempt, argv, run_directory = write_attempt(tmp_path)
    metadata = json.loads((run_directory / "run_metadata.json").read_text(encoding="utf-8"))
    metadata[field] = value
    (run_directory / "run_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="metadata"):
        verifier.verify_engine_output(spec, attempt, argv)


def test_unexpected_step_file_and_invalid_flow_index_are_rejected(tmp_path: Path) -> None:
    """Extra agent files and out-of-range flow files cannot inflate a completed run."""
    require_verifier()
    spec, attempt, argv, run_directory = write_attempt(tmp_path)
    agent_rows(1, "2024-01-02", "2025-01-01", 365).to_parquet(run_directory / "agents" / "agents_step_9.parquet", index=False)
    with pytest.raises(ValueError, match="agent inventory"):
        verifier.verify_engine_output(spec, attempt, argv)
    (run_directory / "agents" / "agents_step_9.parquet").unlink()
    flow_rows(1, "2024-01-02", "2025-01-01", 365).to_parquet(run_directory / "flows" / "flows_step_9.parquet", index=False)
    with pytest.raises(ValueError, match="flow"):
        verifier.verify_engine_output(spec, attempt, argv)


@pytest.mark.parametrize("mutation", ["schema", "nan", "negative", "capacity_left"])
def test_bad_schema_or_physical_agent_values_are_rejected(tmp_path: Path, mutation: str) -> None:
    """Scientific telemetry must have its full schema and obey physical bounds."""
    require_verifier()
    spec, attempt, argv, run_directory = write_attempt(tmp_path)
    path = run_directory / "agents" / "agents_step_0.parquet"
    frame = pd.read_parquet(path)
    if mutation == "schema":
        frame = frame.drop(columns="inventory")
    elif mutation == "nan":
        frame.loc[0, "capacity"] = np.nan
    elif mutation == "negative":
        frame.loc[0, "inventory"] = -1.0
    else:
        frame.loc[0, "capacity_left"] = 10.1
    frame.to_parquet(path, index=False)

    with pytest.raises(ValueError, match="schema|finite|nonnegative|capacity_left"):
        verifier.verify_engine_output(spec, attempt, argv)


def test_bad_context_columns_and_temporary_files_are_rejected(tmp_path: Path) -> None:
    """Stale context or an interrupted temporary output must block authorization."""
    require_verifier()
    spec, attempt, argv, run_directory = write_attempt(tmp_path)
    path = run_directory / "flows" / "flows_step_0.parquet"
    frame = pd.read_parquet(path)
    frame.loc[0, "step_days"] = 365
    frame.to_parquet(path, index=False)
    with pytest.raises(ValueError, match="context"):
        verifier.verify_engine_output(spec, attempt, argv)
    flow_rows(0, "2023-01-01", "2024-01-02", 366).to_parquet(path, index=False)
    (run_directory / "agents" / ".agents_step_1.parquet.1.tmp").write_text("interrupted", encoding="utf-8")
    with pytest.raises(ValueError, match="temporary"):
        verifier.verify_engine_output(spec, attempt, argv)


def test_wrong_scenario_or_attempt_pass_receipt_is_rejected(tmp_path: Path) -> None:
    """A receipt is valid only for the exact ScenarioSpec and immutable attempt root."""
    require_verifier()
    spec, attempt, argv, _ = write_attempt(tmp_path)
    verifier.verify_engine_output(spec, attempt, argv)
    receipt = pass_receipt(spec, attempt)
    receipt["scenario_key"] = "scenario_002_wrong"
    (attempt / "PASS.json").write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(ValueError, match="scenario"):
        verifier.validate_pass_receipt(spec, attempt)
    receipt = pass_receipt(spec, attempt)
    receipt["attempt"] = 0
    (attempt / "PASS.json").write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="attempt"):
        verifier.validate_pass_receipt(spec, attempt)


def test_pass_receipt_rejects_positive_attempt_that_differs_from_root(tmp_path: Path) -> None:
    """A receipt for attempt-0001 cannot claim the distinct positive attempt 2."""
    require_verifier()
    spec, attempt, argv, _ = write_attempt(tmp_path)
    verifier.verify_engine_output(spec, attempt, argv)
    receipt = pass_receipt(spec, attempt)
    receipt["attempt"] = 2
    (attempt / "PASS.json").write_text(json.dumps(receipt), encoding="utf-8")

    with pytest.raises(ValueError, match="attempt"):
        verifier.validate_pass_receipt(spec, attempt)
