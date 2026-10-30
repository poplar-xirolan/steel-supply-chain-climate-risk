"""Portable new runs using the paper engine and exact scientific scenario grid.

Historical Gate-C permissions and receipts belong to the preserved 2026-08 run.
They are not imported or read by this new-run path. Each new attempt has its own
validated outputs and manifest. Failed attempts are never silently retried.
"""
from __future__ import annotations
from steel_supply.pipeline_paths import science_path, bundle_root, data_path

import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from .scenario_grid import scenario_commands, command_arg
from ..pipeline_paths import data_path


def commands(root: Path, experiment: str = "all") -> list[list[str]]:
    root = Path(root).resolve()
    result = scenario_commands(experiment)
    for command in result:
        command[0] = sys.executable
        for flag, value in {
            "--mfa-path": science_path(root, 'results/mfa/2023.parquet'),
            "--hazard-data-dir": science_path(root, 'results/hazard_country_level_loss'),
            "--output-dir": science_path(root, 'results/simulation/scenarios'),
        }.items():
            command[command.index(flag) + 1] = str(value)
        if "--terminal-step-policy" not in command:
            command += ["--terminal-step-policy", "complete"]
    return result


def resolve_run_path(value: str, root: Path | None = None) -> Path:
    """Resolve catalogue paths under selected data, never a retired machine path."""
    base = Path(root).resolve() if root is not None else data_path()
    normalized = str(value).replace("\\", "/")
    if "/data/results/" in normalized:
        normalized = "results/" + normalized.split("/data/results/", 1)[1]
    candidate = Path(normalized)
    if candidate.is_absolute():
        candidate = candidate.resolve()
        if not candidate.is_relative_to(base):
            raise ValueError(f"Run is outside selected data root: {value}")
    else:
        candidate = (base / candidate).resolve()
    if not candidate.is_relative_to(base):
        raise ValueError(f"Run escapes data root: {value}")
    if not (candidate / "run_metadata.json").is_file():
        raise FileNotFoundError(f"Missing run metadata in {candidate}")
    return candidate


def catalogue_row(ordinal: int, command: list[str], run: Path, root: Path) -> dict:
    return {
        "ordinal": ordinal,
        "climate": command_arg(command, "--climate-scenario"),
        "trade": command_arg(command, "--trade-scenario"),
        "inventory": float(command_arg(command, "--inventory-to-capacity-ratio")),
        "capacity": float(command_arg(command, "--capacity-to-initial-flow")),
        "random_ratio": command_arg(command, "--random-disruption-ratio", ""),
        "random_seed": command_arg(command, "--random-seed", ""),
        "run_folder": run.resolve().relative_to(root.resolve()).as_posix(),
    }


def validate_catalogue(result, root: Path) -> None:
    import pandas as pd
    if len(result) != 411 or set(result.ordinal) != set(range(1, 412)):
        raise ValueError("The full manuscript analysis requires exactly 411 unique scenarios")
    normalized_paths = result.run_folder.map(lambda p: str((root / str(p)).resolve()).casefold())
    if not normalized_paths.is_unique:
        raise ValueError("Each scenario must bind a unique run directory")
    expected = pd.DataFrame([catalogue_row(i, cmd, root / f"run-{i}", root)
                             for i, cmd in enumerate(commands(root), 1)])
    columns = ["climate", "trade", "inventory", "capacity", "random_ratio", "random_seed"]
    left = result.sort_values("ordinal")[columns].reset_index(drop=True).copy()
    right = expected[columns].copy()
    for col in columns[2:]:
        left[col] = pd.to_numeric(left[col], errors="raise")
        right[col] = pd.to_numeric(right[col], errors="raise")
    try:
        pd.testing.assert_frame_equal(left, right, check_dtype=False, check_exact=True)
    except AssertionError as exc:
        raise ValueError("Catalogue parameters differ from the manuscript scenario grid") from exc


def validate_run_binding(row, root: Path, planned: list[str]) -> None:
    """Bind catalogue parameters to a verified run, without historical authority.

    Only current per-run manifests are read. Legacy argv paths are hash evidence,
    never opened or executed. Output-file rehashing is the independent verifier's
    job; this check binds identity and metadata before postprocessing.
    """
    from .run_contract import canonical_argv_sha256
    import pandas as pd
    run = resolve_run_path(row["run_folder"], root)
    attempt = run.parent.parent
    receipt = json.loads((attempt / "PASS.json").read_text(encoding="utf-8"))
    manifest_bytes = (attempt / "output_manifest.json").read_bytes()
    if receipt.get("status") != "PASS" or hashlib.sha256(manifest_bytes).hexdigest() != receipt.get("manifest_sha256"):
        raise ValueError(f"Run receipt/manifest mismatch: {run}")
    manifest = json.loads(manifest_bytes)
    if (attempt / manifest["run_directory"]).resolve() != run or manifest["scenario_ordinal"] != int(row["ordinal"]):
        raise ValueError(f"Catalogue run identity mismatch: {run}")
    if receipt.get("schema") == "steel-reproduction-1":
        command = receipt["command"]
        if canonical_argv_sha256(command) != manifest["executed_argv_sha256"]:
            raise ValueError(f"Executed command differs from manifest: {run}")
        # Ignore relocation-only paths, not simulation settings.
        ignored = {"--mfa-path", "--hazard-data-dir", "--output-dir"}
        def science(argv):
            return {flag: argv[i + 1] for i, flag in enumerate(argv[:-1])
                    if flag.startswith("--") and flag not in ignored}
        if science(command) != science(planned):
            raise ValueError(f"Run parameters differ from selected scenario: {run}")
    else:
        command = planned
        if canonical_argv_sha256(planned) != manifest["scenario_argv_sha256"]:
            raise ValueError(f"Legacy parameters differ from verified manifest: {run}")
    bound = catalogue_row(int(row["ordinal"]), command, run, root)
    for key in ["climate", "trade", "inventory", "capacity", "random_ratio", "random_seed"]:
        actual, expected = row[key], bound[key]
        if key in {"random_ratio", "random_seed"} and (pd.isna(actual) or actual == "") and expected == "":
            continue
        if key in {"inventory", "capacity", "random_ratio", "random_seed"}:
            actual, expected = float(actual), float(expected)
        if actual != expected:
            raise ValueError(f"Catalogue parameters do not describe run: {key}, {run}")
    metadata_entry = next(p for p in manifest["files"] if p["relative_path"] == "run_metadata.json")
    if hashlib.sha256((run / "run_metadata.json").read_bytes()).hexdigest() != metadata_entry["sha256"]:
        raise ValueError(f"Run metadata differs from verified manifest: {run}")


def load_catalogue(root: Path | None = None):
    """Read only a newly produced, verified catalogue in this workspace."""
    import pandas as pd
    root = Path(root).resolve() if root is not None else data_path()
    simulation = science_path(root, 'results/simulation')
    fresh = simulation / "reproduction/scenario_catalog.csv"
    if not fresh.is_file():raise FileNotFoundError(fresh)
    result = pd.read_csv(fresh)
    result["run_folder"] = result.run_folder.map(lambda p: str(resolve_run_path(p, root)))
    grid = commands(root)
    validate_catalogue(result, root)
    for _, row in result.iterrows():
        validate_run_binding(row, root, grid[int(row.ordinal) - 1])
    return result


def run_grid(root: Path, experiment: str = "all", max_scenarios: int | None = None, *, resume: bool = False) -> Path:
    # Pure output verifier does not call the historical permission/authority checker.
    from .run_contract import ScenarioSpec, hidden_subprocess_kwargs
    from .abm_scenario_verifier import verify_engine_output

    root = Path(root).resolve()
    grid = commands(root, experiment)
    if max_scenarios is not None:
        if max_scenarios < 1:
            raise ValueError("max_scenarios must be positive")
        grid = grid[:max_scenarios]
    simulation = science_path(root, 'results/simulation')
    control = simulation / "reproduction"
    from ..reproduction.execution import fingerprint, code_fingerprint, runtime_signature, write_json, sha256
    sources = ['results/mfa/2023.parquet']
    if any(command_arg(cmd, '--hazard-type') != 'none' for cmd in grid):
        sources.append('results/hazard_country_level_loss/*_multi-hazard-max_*.parquet')
    sources = [science_path(root, source).relative_to(root).as_posix() for source in sources]
    contract = json.loads(json.dumps({'grid': grid, 'inputs': fingerprint(root, sources),
                'code': code_fingerprint(), 'environment': runtime_signature()}))
    contract_path = control/'execution_contract.json'
    if control.exists():
        if not resume:
            raise FileExistsError('Existing ABM batch; use --resume or a new data root')
        if not contract_path.is_file() or json.loads(contract_path.read_text(encoding='utf-8')) != contract:
            raise ValueError('ABM execution contract/input/code/environment changed; existing results retained')
    else:
        existing_scenarios = simulation/'scenarios'
        if existing_scenarios.exists() and any(existing_scenarios.iterdir()):
            raise ValueError('Existing scenario outputs have no matching batch contract; use a new root, not unbound attempts')
        control.mkdir(parents=True, exist_ok=False)
        write_json(contract_path, contract)
        write_json(control/'scenario_grid.json', grid)
    rows = []
    for ordinal, planned in enumerate(grid, 1):
        scenario = simulation / 'scenarios' / f'scenario_{ordinal:03d}'
        attempts = sorted(scenario.glob('attempt-*'))
        passed = [p for p in attempts if (p/'PASS.json').is_file()]
        if passed:
            if len(passed) != 1:
                raise ValueError(f'Multiple successful attempts for scenario {ordinal}')
            attempt = passed[0]
            receipt = json.loads((attempt/'PASS.json').read_text(encoding='utf-8'))
            manifest_bytes = (attempt/'output_manifest.json').read_bytes()
            if hashlib.sha256(manifest_bytes).hexdigest() != receipt.get('manifest_sha256'):
                raise ValueError('ABM manifest hash changed')
            manifest = json.loads(manifest_bytes)
            run = (attempt/manifest['run_directory']).resolve()
            if not run.is_relative_to(attempt.resolve()):
                raise ValueError('ABM manifest escapes attempt')
            row = catalogue_row(ordinal, planned, run, root)
            validate_run_binding(row, root, planned)
            for entry in manifest['files']:
                path = (run/entry['relative_path']).resolve()
                if not path.is_relative_to(run) or not path.is_file() or sha256(path) != entry['sha256']:
                    raise ValueError(f'ABM output hash changed: {path}')
            rows.append(row)
            print(f'Verified; skipping scenario {ordinal}/{len(grid)}', flush=True)
            continue
        number = max([int(p.name.split('-')[-1]) for p in attempts], default=0)+1
        attempt = scenario / f'attempt-{number:04d}'
        attempt.mkdir(parents=True, exist_ok=False)
        command = list(planned)
        command[command.index("--output-dir") + 1] = str(attempt / "engine_output")
        with (attempt / "stdout.log").open("wb") as out, (attempt / "stderr.log").open("wb") as err:
            subprocess.run(command, check=True, stdout=out, stderr=err, **hidden_subprocess_kwargs())
        summary = verify_engine_output(ScenarioSpec.from_argv(ordinal, planned), attempt, command)
        # Only publish discoverable PASS after real output validation; this is a new
        # schema, not a fabricated copy of a historical dispatcher authorization.
        receipt = {
            "schema": "steel-reproduction-1", "status": "PASS", "ordinal": ordinal,
            "command": command, "manifest_sha256": summary.manifest_sha256,
            "completed_steps": summary.completed_steps,
        }
        (attempt / "PASS.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
        rows.append(catalogue_row(ordinal, command, summary.run_directory, root))
        print(f"Verified scenario {ordinal}/{len(grid)}", flush=True)
    destination = control / "scenario_catalog.csv"
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return destination

