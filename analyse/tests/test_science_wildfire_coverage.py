from __future__ import annotations

from pathlib import Path
import hashlib
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'library'))

from steel_supply.hazard_dynamic.wildfire_coverage import (  # noqa: E402
    fixed_ensemble_mean,
    select_formal_wildfire_sources,
    selected_ensemble_keys,
    validate_model_coverage,
)


def _metadata() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "dataset_source": "steel",
                "climate_scenario": "ssp126",
                "model": "gcm-a",
                "forcing_member": "impact-a",
                "sens_scenario": "default",
                "soc_scenario": "2015soc",
                "start_year": "2015",
                "end_year": "2100",
                "source_nc": "old-a.nc",
                "raw_filename": "old-a.parquet",
                "filepath": "/inputs/old-a.parquet",
            },
            {
                "dataset_source": "steel",
                "climate_scenario": "ssp126",
                "model": "gcm-a",
                "forcing_member": "impact-a",
                "sens_scenario": "default",
                "soc_scenario": "2015soc-from-histsoc",
                "start_year": "2015",
                "end_year": "2100",
                "source_nc": "preferred-a.nc",
                "raw_filename": "preferred-a.parquet",
                "filepath": "/inputs/preferred-a.parquet",
            },
            {
                "dataset_source": "steel",
                "climate_scenario": "ssp126",
                "model": "gcm-b",
                "forcing_member": "impact-b",
                "sens_scenario": "default",
                "soc_scenario": "2015soc",
                "start_year": "2015",
                "end_year": "2100",
                "source_nc": "fallback-b.nc",
                "raw_filename": "fallback-b.parquet",
                "filepath": "/inputs/fallback-b.parquet",
            },
            {
                "dataset_source": "steel",
                "climate_scenario": "ssp370",
                "model": "gcm-c",
                "forcing_member": "impact-c",
                "sens_scenario": "default",
                "soc_scenario": "2015soc-from-histsoc",
                "start_year": "2015",
                "end_year": "2100",
                "source_nc": "other-scenario.nc",
                "raw_filename": "other-scenario.parquet",
                "filepath": "/inputs/other-scenario.parquet",
            },
        ]
    )


def _expected_models() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "model": ["gcm-a", "gcm-b"],
            "forcing_member": ["impact-a", "impact-b"],
            "source_nc": ["a.nc", "b.nc"],
        }
    )


def _complete_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id_internal": ["facility-1"] * 4,
            "month_start": pd.to_datetime(
                ["2015-01-01", "2015-01-01", "2015-02-01", "2015-02-01"]
            ),
            "model": ["gcm-a", "gcm-b", "gcm-a", "gcm-b"],
            "forcing_member": ["impact-a", "impact-b", "impact-a", "impact-b"],
            "source_nc": ["a.nc", "b.nc", "a.nc", "b.nc"],
            "burntarea-total": [2.0, 4.0, 0.0, 0.0],
        }
    )


def test_selected_ensemble_prefers_from_histsoc_per_model_forcing_pair():
    selected = selected_ensemble_keys(_metadata(), "steel", "ssp126")

    assert selected[["model", "forcing_member", "soc_scenario", "source_nc"]].to_dict(
        "records"
    ) == [
        {
            "model": "gcm-a",
            "forcing_member": "impact-a",
            "soc_scenario": "2015soc-from-histsoc",
            "source_nc": "preferred-a.nc",
        },
        {
            "model": "gcm-b",
            "forcing_member": "impact-b",
            "soc_scenario": "2015soc",
            "source_nc": "fallback-b.nc",
        },
    ]


def test_selected_ensemble_rejects_empty_source_scenario():
    with pytest.raises(ValueError, match="empty selected wildfire ensemble"):
        selected_ensemble_keys(_metadata(), "mines", "ssp126")


def test_selected_ensemble_rejects_duplicate_preferred_model_key():
    metadata = pd.concat([_metadata(), _metadata().iloc[[1]]], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate selected ensemble key"):
        selected_ensemble_keys(metadata, "steel", "ssp126")


def test_formal_source_selection_matches_step_09_fixed_ensemble(tmp_path: Path):
    paths = []
    for scenario in ("picontrol", "ssp126", "ssp370", "ssp585"):
        for soc in ("2015soc", "2015soc-from-histsoc"):
            paths.append(
                tmp_path
                / (
                    f"impact-a_gcm-a_w5e5_{scenario}_{soc}_default_"
                    "burntarea-total_global_monthly_2015_2100.nc"
                )
            )
    paths.extend(
        [
            tmp_path
            / "impact-a_gcm-a_w5e5_historical_histsoc_default_"
            "burntarea-total_global_monthly_1850_2014.nc",
            tmp_path
            / "impact-b_gcm-b_w5e5_ssp585_2015soc_nondep_"
            "burntarea-total_global_monthly_2015_2100.nc",
        ]
    )

    selected = select_formal_wildfire_sources(paths)

    assert len(selected) == 4
    assert all("2015soc-from-histsoc" in path.name for path in selected)
    assert {path.name.split("_")[3] for path in selected} == {
        "picontrol",
        "ssp126",
        "ssp370",
        "ssp585",
    }


def test_formal_source_selection_rejects_nonformal_selected_bounds(tmp_path: Path):
    paths = [
        tmp_path
        / (
            f"impact-a_gcm-a_w5e5_{scenario}_2015soc-from-histsoc_default_"
            "burntarea-total_global_monthly_2015_2100.nc"
        )
        for scenario in ("picontrol", "ssp126", "ssp370")
    ]
    paths.append(
        tmp_path
        / "impact-a_gcm-a_w5e5_ssp585_2015soc-from-histsoc_default_"
        "burntarea-total_global_monthly_2015_2099.nc"
    )

    with pytest.raises(ValueError, match="formal interval.*ssp585"):
        select_formal_wildfire_sources(paths)


def test_validate_model_coverage_rejects_missing_model_month():
    data = _complete_data().drop(index=1)

    with pytest.raises(ValueError, match="missing.*facility-model-month"):
        validate_model_coverage(
            data,
            _expected_models(),
            pd.Index(["facility-1"]),
            pd.date_range("2015-01-01", "2015-02-01", freq="MS"),
        )


def test_validate_model_coverage_rejects_duplicate_model_key():
    data = pd.concat([_complete_data(), _complete_data().iloc[[0]]], ignore_index=True)

    with pytest.raises(ValueError, match="duplicate facility-model-month key"):
        validate_model_coverage(
            data,
            _expected_models(),
            pd.Index(["facility-1"]),
            pd.date_range("2015-01-01", "2015-02-01", freq="MS"),
        )


def test_validate_model_coverage_rejects_nan_model_value():
    data = _complete_data()
    data.loc[0, "burntarea-total"] = np.nan

    with pytest.raises(ValueError, match="non-finite wildfire value"):
        validate_model_coverage(
            data,
            _expected_models(),
            pd.Index(["facility-1"]),
            pd.date_range("2015-01-01", "2015-02-01", freq="MS"),
        )


def test_validate_model_coverage_rejects_wrong_source_for_all_pair_rows():
    data = _complete_data()
    data.loc[data["model"].eq("gcm-a"), "source_nc"] = "wrong-a.nc"

    with pytest.raises(ValueError, match="model/forcing/source-file keys"):
        validate_model_coverage(
            data,
            _expected_models(),
            pd.Index(["facility-1"]),
            pd.date_range("2015-01-01", "2015-02-01", freq="MS"),
        )


def test_validate_model_coverage_rejects_mixed_sources_with_expected_last():
    data = _complete_data()
    data.loc[0, "source_nc"] = "wrong-a.nc"
    assert data.loc[2, "source_nc"] == "a.nc"

    with pytest.raises(ValueError, match="multiple source files"):
        validate_model_coverage(
            data,
            _expected_models(),
            pd.Index(["facility-1"]),
            pd.date_range("2015-01-01", "2015-02-01", freq="MS"),
        )


def test_validate_model_coverage_rejects_duplicate_selected_model_pair():
    expected_models = pd.concat(
        [
            _expected_models(),
            pd.DataFrame(
                {
                    "model": ["gcm-a"],
                    "forcing_member": ["impact-a"],
                    "source_nc": ["alternate-a.nc"],
                }
            ),
        ],
        ignore_index=True,
    )

    with pytest.raises(ValueError, match="duplicate expected model/forcing key"):
        validate_model_coverage(
            _complete_data(),
            expected_models,
            pd.Index(["facility-1"]),
            pd.date_range("2015-01-01", "2015-02-01", freq="MS"),
        )


def test_validate_model_coverage_returns_fixed_observed_count_for_complete_matrix():
    coverage = validate_model_coverage(
        _complete_data(),
        _expected_models(),
        pd.Index(["facility-1"]),
        pd.date_range("2015-01-01", "2015-02-01", freq="MS"),
    )

    assert coverage.to_dict("records") == [
        {
            "id_internal": "facility-1",
            "month_start": pd.Timestamp("2015-01-01"),
            "expected_model_count": 2,
            "observed_model_count": 2,
        },
        {
            "id_internal": "facility-1",
            "month_start": pd.Timestamp("2015-02-01"),
            "expected_model_count": 2,
            "observed_model_count": 2,
        },
    ]


def test_fixed_ensemble_mean_returns_hand_calculated_arithmetic_mean():
    result = fixed_ensemble_mean(_complete_data(), expected_model_count=2)

    assert result.to_dict("records") == [
        {
            "id_internal": "facility-1",
            "month_start": pd.Timestamp("2015-01-01"),
            "burntarea-total": 3.0,
            "observed_model_count": 2,
        },
        {
            "id_internal": "facility-1",
            "month_start": pd.Timestamp("2015-02-01"),
            "burntarea-total": 0.0,
            "observed_model_count": 2,
        },
    ]


def test_fixed_ensemble_mean_rejects_available_case_average():
    data = _complete_data().drop(index=1)

    with pytest.raises(ValueError, match="observed model count"):
        fixed_ensemble_mean(data, expected_model_count=2)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _write_cli_fixture(input_dir: Path) -> tuple[Path, Path]:
    input_dir.mkdir()
    facilities_dir = input_dir.parent / "facilities"
    facilities_dir.mkdir()
    pd.DataFrame({"ID_internal": ["facility-1", "facility-2"]}).to_parquet(
        facilities_dir / "global_energy_monitor_iron_steel_basic_info.parquet",
        index=False,
    )
    source = "result_global_energy_monitor_iron_steel_basic_info"
    months = pd.date_range("2015-01-01", "2015-12-01", freq="MS")
    paths = []
    for impact_model, climate_model, soc, positive in [
        ("impact-a", "gcm-a", "2015soc-from-histsoc", 2.0),
        ("impact-b", "gcm-b", "2015soc", 4.0),
    ]:
        source_nc = (
            f"{impact_model}_{climate_model}_w5e5_ssp126_{soc}_default_"
            "burntarea-total_global_monthly_2015_2015.nc"
        )
        path = input_dir / f"{source}_{source_nc.removesuffix('.nc')}.parquet"
        values = []
        for facility in ("facility-1", "facility-2"):
            for month in months:
                value = positive if facility == "facility-1" and month.month > 1 else 0.0
                values.append((facility, source_nc, str(month), value))
        pd.DataFrame(
            values,
            columns=["id_internal", "source_nc", "time", "burntarea-total"],
        ).to_parquet(path, index=False)
        paths.append(path.resolve())

    receipt = {
        "stage": "wildfire_extraction",
        "status": "PASS",
        "code_commit": "fixture",
        "input_paths": [],
        "input_hashes": {},
        "parameters": {},
        "row_counts": {"facility_rows": 2},
        "coverage": {
            "expected_facility_model_months": 48,
            "finite_facility_model_months": 48,
            "duplicate_facility_model_months": 0,
        },
        "output_paths": [str(path) for path in paths],
        "output_hashes": {str(path): _sha256(path) for path in paths},
        "checks": {"complete_finite_unique_keys": True},
        "created_at_utc": "2026-08-12T00:00:00+00:00",
    }
    receipt_path = input_dir / "wildfire_extraction_receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    return receipt_path, facilities_dir


def _builder_path() -> Path:
    return Path(__file__).resolve().parents[1] / "04_facility_hazards/09_build_wildfire_loss.py"


def test_builder_writes_sparse_daily_output_only_after_complete_coverage(tmp_path):
    input_dir = tmp_path / "input"
    receipt_path, facilities_dir = _write_cli_fixture(input_dir)
    output_dir = tmp_path / "output"

    completed = subprocess.run(
        [
            sys.executable,
            str(_builder_path()),
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--extraction-receipt",
            str(receipt_path),
            "--facilities-dir",
            str(facilities_dir),
            "--sources",
            "result_global_energy_monitor_iron_steel_basic_info",
            "--scenarios",
            "ssp126",
            "--start-year",
            "2015",
            "--end-year",
            "2015",
        ],
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    result_path = output_dir / (
        "production_loss_result-global-energy-monitor-iron-steel-basic-info_"
        "2015_ssp126.parquet"
    )
    result = pd.read_parquet(result_path)
    assert len(result) == 334
    assert result["id_internal"].unique().tolist() == ["facility-1"]
    assert result["day_of_year"].min() == 32
    assert result["day_of_year"].max() == 365
    assert result["production_loss"].unique().tolist() == [0.03]

    receipt = json.loads(
        (output_dir / "wildfire_coverage_receipt.json").read_text(encoding="utf-8")
    )
    assert receipt["status"] == "PASS"
    ensemble = receipt["coverage"]["ensembles"][0]
    assert ensemble["expected_facility_model_month_keys"] == 48
    assert ensemble["observed_facility_model_month_keys"] == 48
    assert ensemble["missing_key_count"] == 0
    assert ensemble["duplicate_key_count"] == 0
    assert ensemble["ensemble_zero_month_count"] == 13
    assert ensemble["ensemble_positive_month_count"] == 11
    assert ensemble["daily_output_rows"] == 334
    assert receipt["checks"]["selected_extraction_output_hashes_match"] is True
    assert receipt["checks"]["facility_inputs_hashed"] is True
    assert receipt["checks"]["sparse_outputs_written_after_coverage_pass"] is True


def test_builder_fails_empty_selection_without_writing_empty_annual_files(tmp_path):
    input_dir = tmp_path / "input"
    receipt_path, facilities_dir = _write_cli_fixture(input_dir)
    output_dir = tmp_path / "output"

    completed = subprocess.run(
        [
            sys.executable,
            str(_builder_path()),
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--extraction-receipt",
            str(receipt_path),
            "--facilities-dir",
            str(facilities_dir),
            "--sources",
            "result_global_energy_monitor_iron_steel_basic_info",
            "--scenarios",
            "ssp585",
            "--start-year",
            "2015",
            "--end-year",
            "2015",
        ],
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    assert list(output_dir.glob("production_loss_*.parquet")) == []
    receipt = json.loads(
        (output_dir / "wildfire_coverage_receipt.json").read_text(encoding="utf-8")
    )
    assert receipt["status"] == "FAILED"
    assert "empty selected wildfire ensemble" in receipt["checks"]["error"]
