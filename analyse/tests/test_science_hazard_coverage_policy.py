from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys


def _builder_path() -> Path:
    return Path(__file__).resolve().parents[1] / "05_country_constraints/10a_build_hazard_coverage_policy.py"


def _write_complete_fixture(root: Path) -> tuple[Path, Path, Path]:
    raw = root / "raw"
    extracted = root / "extracted"
    loss = root / "loss"
    raw.mkdir()
    extracted.mkdir()
    loss.mkdir()
    facilities = (
        "global_energy_monitor_iron_steel_basic_info",
        "spglobal_iron_mine_basic_info",
    )
    for year in range(2061, 2101):
        for member in range(8):
            source = f"er{member:02d}_mit_gcm{member}_ssp126_windlifetimemax_{year}"
            (raw / f"{source}.nc").write_bytes(b"raw")
            for facility in facilities:
                (extracted / f"result_{facility}_{source}.parquet").write_bytes(b"extract")
        for facility in facilities:
            hyphenated = facility.replace("_", "-")
            (loss / f"production_loss_result-{hyphenated}_{year}_ssp126.parquet").write_bytes(
                b"loss"
            )
    return raw, extracted, loss


def test_policy_builder_writes_current_self_consistent_source_identity(tmp_path: Path):
    raw, extracted, loss = _write_complete_fixture(tmp_path)
    inventory = tmp_path / "out" / "inventory.csv"
    policy = tmp_path / "out" / "policy.json"

    result = subprocess.run(
        [
            sys.executable,
            str(_builder_path()),
            "--wind-input-dir",
            str(raw),
            "--wind-extract-dir",
            str(extracted),
            "--cyclone-loss-dir",
            str(loss),
            "--inventory-path",
            str(inventory),
            "--policy-path",
            str(policy),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    payload = json.loads(policy.read_text(encoding="utf-8"))
    inventory_hash = hashlib.sha256(inventory.read_bytes()).hexdigest()
    assert payload["status"] == "PASS"
    assert len(payload["unavailable_components"]) == 114
    assert len(payload["source_provenance"]) == 1
    provenance = payload["source_provenance"][0]
    assert provenance["path"] == str(inventory.resolve())
    assert provenance["sha256"] == inventory_hash
    assert payload["input_hashes"] == {str(inventory.resolve()): inventory_hash}
    assert [item["file_count"] for item in provenance["evidence_layers"]] == [
        320,
        640,
        80,
    ]


def test_policy_builder_fails_closed_when_one_available_year_is_missing(tmp_path: Path):
    raw, extracted, loss = _write_complete_fixture(tmp_path)
    next(raw.glob("*_2061.nc")).unlink()

    result = subprocess.run(
        [
            sys.executable,
            str(_builder_path()),
            "--wind-input-dir",
            str(raw),
            "--wind-extract-dir",
            str(extracted),
            "--cyclone-loss-dir",
            str(loss),
            "--inventory-path",
            str(tmp_path / "out" / "inventory.csv"),
            "--policy-path",
            str(tmp_path / "out" / "policy.json"),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "does not prove exact SSP126 cyclone availability" in result.stderr
    assert not (tmp_path / "out" / "policy.json").exists()
