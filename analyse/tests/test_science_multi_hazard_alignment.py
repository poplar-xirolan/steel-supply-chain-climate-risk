from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "05_country_constraints/10b_build_multi_hazard_max_loss.py"
sys.path.insert(0, str(MODULE_PATH.parent))
SPEC = importlib.util.spec_from_file_location("multi_hazard_builder", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
multi_hazard = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(multi_hazard)

TEST_POLICY_INDUSTRIES = ("iron-factory", "iron-mine", "steel-factory")
TEST_POLICY_SCENARIOS = ("picontrol", "ssp126", "ssp370", "ssp585")


def _expected_declared_unavailable_keys() -> list[tuple[str, str, int, str]]:
    return [
        (industry, "tropical-cyclone", year, "ssp126")
        for industry in TEST_POLICY_INDUSTRIES
        for year in range(2023, 2061)
    ]


def _index(*days: int) -> pd.MultiIndex:
    return pd.MultiIndex.from_tuples(
        [("facility-a", day) for day in days],
        names=["ID_internal", "day_of_year"],
    )


def _write_loss(path: Path, days: list[int], values: list[float]) -> Path:
    pd.DataFrame(
        {
            "ID_internal": ["facility-a"] * len(days),
            "day_of_year": days,
            "production_loss_fraction": np.asarray(values, dtype="float64"),
        }
    ).to_parquet(path, index=False)
    return path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_wildfire_receipt(
    path: Path,
    wildfire_path: Path,
    *,
    status: str = "PASS",
    source: str = "result_source",
    scenario: str = "ssp370",
    recorded_hash: str | None = None,
) -> Path:
    payload = {
        "stage": "wildfire_fixed_ensemble_loss",
        "status": status,
        "code_commit": "test-commit",
        "input_paths": [],
        "input_hashes": {},
        "parameters": {
            "sparse_output_rule": (
                "write positive daily rows only after complete monthly coverage PASS"
            )
        },
        "row_counts": {},
        "coverage": {
            "ensembles": [
                {
                    "source": source,
                    "scenario": scenario,
                    "missing_key_count": 0,
                    "duplicate_key_count": 0,
                    "nan_value_count": 0,
                }
            ]
        },
        "output_paths": [str(wildfire_path.resolve())],
        "output_hashes": {
            str(wildfire_path.resolve()): recorded_hash or _sha256(wildfire_path)
        },
        "checks": {
            "all_monthly_matrices_complete": True,
            "all_omitted_monthly_values_exact_zeros": True,
            "sparse_outputs_written_after_coverage_pass": True,
        },
        "created_at_utc": "2026-08-12T00:00:00+00:00",
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_coverage_policy(
    path: Path,
    evidence_path: Path,
    *,
    unavailable_components: list[dict[str, object]] | None = None,
    source_identity: str = "authoritative_ssp126_cyclone_inventory_manifest",
    recorded_hash: str | None = None,
) -> Path:
    evidence_hash = recorded_hash or _sha256(evidence_path)
    payload = {
        "stage": "hazard_coverage_policy",
        "status": "PASS",
        "policy": "ssp126_cyclone_2023_2060_declared_unavailable",
        "version": "1.0.0",
        "created_at_utc": "2026-08-12T00:00:00+00:00",
        "unavailable_components": (
            unavailable_components
            if unavailable_components is not None
            else [
                {
                    "industry": industry,
                    "hazard": hazard,
                    "year": year,
                    "scenario": scenario,
                    "availability": "unavailable",
                }
                for industry, hazard, year, scenario
                in _expected_declared_unavailable_keys()
            ]
        ),
        "source_provenance": [
            {
                "source_identity": source_identity,
                "path": str(evidence_path.resolve()),
                "sha256": evidence_hash,
                "scenario": "ssp126",
                "hazard": "tropical-cyclone",
                "available_start_year": 2061,
                "available_end_year": 2100,
            }
        ],
        "input_hashes": {str(evidence_path.resolve()): evidence_hash},
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _files(
    current: Path,
    baseline: Path | None,
    *,
    hazard: str = "heatwave",
    year: int = 2024,
    scenario: str = "ssp370",
) -> dict[tuple[str, str, int, str], Path]:
    result = {("iron-mine", hazard, year, scenario): current}
    if baseline is not None:
        result[("iron-mine", "heatwave", 2023, scenario)] = baseline
    return result


def test_non_leap_heatwave_alignment_is_unchanged(tmp_path: Path):
    baseline = _write_loss(tmp_path / "baseline.parquet", [59, 60, 61], [0.1, 0.2, 0.3])
    current = _write_loss(tmp_path / "current.parquet", [59, 60, 61], [0.2, 0.4, 0.1])

    result = multi_hazard.load_hazard_component(
        _files(current, baseline, year=2025),
        {},
        _index(59, 60, 61),
        "iron-mine",
        "heatwave",
        2025,
        "ssp370",
        2023,
        None,
    )

    np.testing.assert_allclose(result.to_numpy(), [0.1, 0.2, 0.0])


def test_leap_march_1_maps_to_2023_march_1_not_march_2(tmp_path: Path):
    baseline = _write_loss(tmp_path / "baseline.parquet", [60, 61], [0.2, 0.8])
    current = _write_loss(tmp_path / "current.parquet", [61], [0.5])

    result = multi_hazard.load_hazard_component(
        _files(current, baseline),
        {},
        _index(61),
        "iron-mine",
        "heatwave",
        2024,
        "ssp370",
        2023,
        None,
    )

    assert result.item() == pytest.approx(0.3)


def test_february_29_uses_mean_of_adjacent_2023_days(tmp_path: Path):
    baseline = _write_loss(tmp_path / "baseline.parquet", [59, 60], [0.2, 0.6])
    current = _write_loss(tmp_path / "current.parquet", [60], [0.9])

    result = multi_hazard.load_hazard_component(
        _files(current, baseline),
        {},
        _index(60),
        "iron-mine",
        "heatwave",
        2024,
        "ssp370",
        2023,
        None,
    )

    assert result.item() == pytest.approx(0.5)


def test_missing_heatwave_baseline_file_fails(tmp_path: Path):
    current = _write_loss(tmp_path / "current.parquet", [1], [0.2])
    with pytest.raises(FileNotFoundError, match="heatwave baseline"):
        multi_hazard.load_hazard_component(
            _files(current, None), {}, _index(1), "iron-mine", "heatwave", 2024,
            "ssp370", 2023, None,
        )


def test_missing_heatwave_baseline_key_fails(tmp_path: Path):
    baseline = _write_loss(tmp_path / "baseline.parquet", [2], [0.1])
    current = _write_loss(tmp_path / "current.parquet", [1], [0.2])
    with pytest.raises(ValueError, match="missing required 2023 baseline key"):
        multi_hazard.load_hazard_component(
            _files(current, baseline), {}, _index(1), "iron-mine", "heatwave", 2024,
            "ssp370", 2023, None,
        )


def test_heatwave_alignment_rejects_non_2023_baseline_year(tmp_path: Path):
    current = _write_loss(tmp_path / "current.parquet", [1], [0.2])
    files = {("iron-mine", "heatwave", 2024, "ssp370"): current}
    with pytest.raises(ValueError, match="baseline year must be 2023"):
        multi_hazard.load_hazard_component(
            files, {}, _index(1), "iron-mine", "heatwave", 2024,
            "ssp370", 2022, None,
        )


@pytest.mark.parametrize("hazard", ["tropical-cyclone", "wildfire"])
def test_missing_absolute_hazard_file_fails(hazard: str):
    with pytest.raises(FileNotFoundError, match=hazard):
        multi_hazard.load_hazard_component(
            {}, {}, _index(1), "iron-mine", hazard, 2024, "ssp370", 2023, None
        )


def test_build_combos_requires_all_components_and_heatwave_baseline(tmp_path: Path):
    def path(name: str) -> Path:
        return tmp_path / name

    files = {
        ("iron-mine", "heatwave", 2023, "ssp370"): path("baseline.parquet"),
        ("iron-mine", "heatwave", 2024, "ssp370"): path("heatwave.parquet"),
        ("iron-mine", "tropical-cyclone", 2024, "ssp370"): path("cyclone.parquet"),
        ("iron-mine", "wildfire", 2024, "ssp370"): path("wildfire.parquet"),
        ("iron-mine", "heatwave", 2025, "ssp370"): path("heatwave-2025.parquet"),
        ("iron-mine", "wildfire", 2025, "ssp370"): path("wildfire-2025.parquet"),
        ("iron-factory", "heatwave", 2024, "ssp370"): path("factory-heatwave.parquet"),
        ("iron-factory", "tropical-cyclone", 2024, "ssp370"): path("factory-cyclone.parquet"),
        ("iron-factory", "wildfire", 2024, "ssp370"): path("factory-wildfire.parquet"),
    }

    combos = multi_hazard.build_combos(
        files,
        2024,
        2025,
        industries=None,
        scenarios=None,
        heatwave_baseline_year=2023,
    )

    assert combos == [("iron-mine", 2024, "ssp370")]


def test_skip_existing_requires_complete_schema_and_row_counts(tmp_path: Path):
    facility, country = multi_hazard.output_paths(
        "iron-mine",
        "multi-hazard-max",
        2024,
        "ssp370",
        tmp_path / "facility",
        tmp_path / "country",
    )
    assert facility == tmp_path / "facility/iron-mine_multi-hazard-max_2024_ssp370.parquet"
    assert country == tmp_path / "country/iron-mine_multi-hazard-max_2024_ssp370.parquet"
    inventory = pd.DataFrame(
        {
            "ID_internal": ["facility-a"],
            "country_a3": ["AUS"],
            "fe_eq_weight_kg_scaled": [366.0],
        }
    )
    assert not multi_hazard.validate_existing_output_pair(facility, country, inventory, 2024)
    facility.parent.mkdir()
    pd.DataFrame(columns=multi_hazard.OUTPUT_COLUMNS).to_parquet(facility)
    with pytest.raises(ValueError, match="partial existing output pair"):
        multi_hazard.validate_existing_output_pair(facility, country, inventory, 2024)
    country.parent.mkdir()
    pd.DataFrame(columns=[
        "country_a3", "day_of_year", "daily_fe_eq_weight_kg_scaled",
        "production_loss_kg_scaled", "production_loss_fraction",
    ]).to_parquet(country)
    with pytest.raises(ValueError, match="facility row count"):
        multi_hazard.validate_existing_output_pair(facility, country, inventory, 2024)


def test_skip_existing_accepts_complete_pair(tmp_path: Path):
    inventory = pd.DataFrame(
        {"ID_internal": ["facility-a"], "country_a3": ["AUS"], "fe_eq_weight_kg_scaled": [366.0]}
    )
    base = multi_hazard.load_base_frame(inventory, "iron-mine", 2024)
    base["production_loss_fraction"] = 0.0
    base["production_loss_kg_scaled"] = 0.0
    facility = tmp_path / "facility.parquet"
    country = tmp_path / "country.parquet"
    base[multi_hazard.OUTPUT_COLUMNS].to_parquet(facility, index=False)
    multi_hazard.aggregate_country_loss(base[multi_hazard.OUTPUT_COLUMNS]).to_parquet(
        country, index=False
    )

    assert multi_hazard.validate_existing_output_pair(facility, country, inventory, 2024)


def _valid_resume_pair(tmp_path: Path, year: int = 2023):
    inventory = pd.DataFrame(
        {
            "ID_internal": ["facility-a"],
            "country_a3": ["AUS"],
            "fe_eq_weight_kg_scaled": [365.0 if year == 2023 else 366.0],
        }
    )
    facility_frame = multi_hazard.load_base_frame(inventory, "iron-mine", year)
    facility_frame["production_loss_fraction"] = np.linspace(
        0.0, 0.8, len(facility_frame), dtype="float64"
    )
    facility_frame["production_loss_kg_scaled"] = (
        facility_frame["daily_fe_eq_weight_kg_scaled"]
        * facility_frame["production_loss_fraction"]
    )
    country_frame = multi_hazard.aggregate_country_loss(
        facility_frame[multi_hazard.OUTPUT_COLUMNS]
    )
    facility_path = tmp_path / "facility.parquet"
    country_path = tmp_path / "country.parquet"
    facility_frame[multi_hazard.OUTPUT_COLUMNS].to_parquet(facility_path, index=False)
    country_frame.to_parquet(country_path, index=False)
    return inventory, facility_path, country_path


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("duplicate_missing_key", "facility-day keys"),
        ("nan_loss", "finite"),
        ("out_of_range_loss", "bounded"),
        ("wrong_daily_weight", "daily weights"),
        ("wrong_loss_mass", "production loss mass"),
    ],
)
def test_skip_existing_rejects_row_count_correct_facility_corruption(
    tmp_path: Path,
    mutation: str,
    message: str,
):
    inventory, facility_path, country_path = _valid_resume_pair(tmp_path)
    frame = pd.read_parquet(facility_path)
    if mutation == "duplicate_missing_key":
        frame.loc[frame.index[-1], "day_of_year"] = 1
    elif mutation == "nan_loss":
        frame.loc[0, "production_loss_fraction"] = np.nan
    elif mutation == "out_of_range_loss":
        frame.loc[0, "production_loss_fraction"] = 1.01
        frame.loc[0, "production_loss_kg_scaled"] = 1.01
    elif mutation == "wrong_daily_weight":
        frame.loc[0, "daily_fe_eq_weight_kg_scaled"] += 0.5
    elif mutation == "wrong_loss_mass":
        frame.loc[0, "production_loss_kg_scaled"] += 0.5
    else:  # pragma: no cover - guarded by the parametrization
        raise AssertionError(mutation)
    frame.to_parquet(facility_path, index=False)

    with pytest.raises(ValueError, match=message):
        multi_hazard.validate_existing_output_pair(
            facility_path, country_path, inventory, 2023
        )


@pytest.mark.parametrize("mutation", ["duplicate_missing_key", "wrong_aggregation"])
def test_skip_existing_rejects_row_count_correct_country_corruption(
    tmp_path: Path,
    mutation: str,
):
    inventory, facility_path, country_path = _valid_resume_pair(tmp_path)
    frame = pd.read_parquet(country_path)
    if mutation == "duplicate_missing_key":
        frame.loc[frame.index[-1], "day_of_year"] = 1
    else:
        frame.loc[0, "production_loss_kg_scaled"] += 0.5
    frame.to_parquet(country_path, index=False)

    with pytest.raises(ValueError, match="country"):
        multi_hazard.validate_existing_output_pair(
            facility_path, country_path, inventory, 2023
        )


def test_dense_loss_series_requires_exact_target_keys(tmp_path: Path):
    path = _write_loss(tmp_path / "dense.parquet", [1], [0.25])
    with pytest.raises(ValueError, match="missing required facility-day keys"):
        multi_hazard.read_loss_series(
            path, _index(1, 2), allow_sparse_zero=False, coverage_receipt=None
        )


def test_sparse_wildfire_requires_passing_matching_receipt(tmp_path: Path):
    path = _write_loss(
        tmp_path / "production_loss_result-source_2024_ssp370.parquet", [1], [0.25]
    )
    with pytest.raises(ValueError, match="coverage receipt"):
        multi_hazard.read_loss_series(
            path, _index(1, 2), allow_sparse_zero=True, coverage_receipt=None
        )

    failed = _write_wildfire_receipt(tmp_path / "failed.json", path, status="FAILED")
    with pytest.raises(ValueError, match="PASS"):
        multi_hazard.read_loss_series(
            path, _index(1, 2), allow_sparse_zero=True, coverage_receipt=failed
        )


@pytest.mark.parametrize(
    ("source", "scenario", "recorded_hash", "message"),
    [
        ("other_source", "ssp370", None, "source/scenario"),
        ("result_source", "ssp585", None, "source/scenario"),
        ("result_source", "ssp370", "0" * 64, "hash"),
    ],
)
def test_sparse_wildfire_receipt_identity_must_match(
    tmp_path: Path,
    source: str,
    scenario: str,
    recorded_hash: str | None,
    message: str,
):
    path = _write_loss(
        tmp_path / "production_loss_result-source_2024_ssp370.parquet", [1], [0.25]
    )
    receipt = _write_wildfire_receipt(
        tmp_path / "receipt.json",
        path,
        source=source,
        scenario=scenario,
        recorded_hash=recorded_hash,
    )
    with pytest.raises(ValueError, match=message):
        multi_hazard.read_loss_series(
            path, _index(1, 2), allow_sparse_zero=True, coverage_receipt=receipt
        )


def test_sparse_wildfire_zero_fill_is_allowed_after_exact_receipt(tmp_path: Path):
    path = _write_loss(
        tmp_path / "production_loss_result-source_2024_ssp370.parquet", [1], [0.25]
    )
    receipt = _write_wildfire_receipt(tmp_path / "receipt.json", path)

    result = multi_hazard.read_loss_series(
        path, _index(1, 2), allow_sparse_zero=True, coverage_receipt=receipt
    )

    np.testing.assert_array_equal(result.to_numpy(), [0.25, 0.0])


def test_base_frame_is_full_inventory_cross_gregorian_calendar():
    inventory = pd.DataFrame(
        {
            "ID_internal": ["b", "a"],
            "country_a3": ["USA", "AUS"],
            "fe_eq_weight_kg_scaled": [3660.0, 7320.0],
        }
    )

    result = multi_hazard.load_base_frame(inventory, "iron-mine", 2024)

    assert len(result) == 2 * 366
    assert result["day_of_year"].min() == 1
    assert result["day_of_year"].max() == 366
    assert not result.duplicated(["ID_internal", "day_of_year"]).any()
    assert result.loc[result["ID_internal"].eq("a"), "daily_fe_eq_weight_kg_scaled"].unique().item() == 20.0


def test_multi_hazard_formula_is_clipped_rowwise_max(tmp_path: Path):
    inventory = pd.DataFrame(
        {
            "ID_internal": ["facility-a"],
            "country_a3": ["AUS"],
            "fe_eq_weight_kg_scaled": [365.0],
        }
    )
    days = list(range(1, 366))
    baseline_values = [0.1] * 365
    heatwave_values = [0.1] * 365
    cyclone_values = [0.0] * 365
    wildfire_values = [0.0] * 365
    heatwave_values[0] = 0.6
    cyclone_values[1] = 0.8
    wildfire_values[2] = 1.4
    baseline = _write_loss(tmp_path / "baseline.parquet", days, baseline_values)
    heatwave = _write_loss(tmp_path / "heatwave.parquet", days, heatwave_values)
    cyclone = _write_loss(tmp_path / "cyclone.parquet", days, cyclone_values)
    wildfire = _write_loss(tmp_path / "wildfire.parquet", days, wildfire_values)
    files = {
        ("iron-mine", "heatwave", 2023, "ssp370"): baseline,
        ("iron-mine", "heatwave", 2025, "ssp370"): heatwave,
        ("iron-mine", "tropical-cyclone", 2025, "ssp370"): cyclone,
        ("iron-mine", "wildfire", 2025, "ssp370"): wildfire,
    }

    result = multi_hazard.build_one_combo(
        files, {}, inventory, "iron-mine", 2025, "ssp370", 2023, None
    )

    np.testing.assert_allclose(
        result.set_index("day_of_year").loc[[1, 2, 3], "production_loss_fraction"],
        [0.5, 0.8, 1.0],
    )


@pytest.mark.parametrize("year", [2023, 2060])
def test_declared_ssp126_cyclone_gap_uses_literal_two_hazard_max(
    tmp_path: Path,
    year: int,
):
    evidence = tmp_path / "source_inventory.csv"
    evidence.write_text("layer,first_year,last_year\nraw,2061,2100\n", encoding="utf-8")
    policy_path = _write_coverage_policy(tmp_path / "coverage.json", evidence)
    policy = multi_hazard.load_coverage_policy(policy_path)
    inventory = pd.DataFrame(
        {
            "ID_internal": ["facility-a"],
            "country_a3": ["AUS"],
            "fe_eq_weight_kg_scaled": [365.0 if year == 2023 else 366.0],
        }
    )
    days = list(range(1, 366 if year == 2023 else 367))
    baseline = _write_loss(tmp_path / "baseline.parquet", days[:365], [0.125] * 365)
    heatwave_values = [0.375] * len(days)
    wildfire_values = [0.125, 0.75] + [0.125] * (len(days) - 2)
    heatwave = _write_loss(tmp_path / "heatwave.parquet", days, heatwave_values)
    wildfire = _write_loss(tmp_path / "wildfire.parquet", days, wildfire_values)
    files = {
        ("iron-mine", "heatwave", 2023, "ssp126"): baseline,
        ("iron-mine", "heatwave", year, "ssp126"): heatwave,
        ("iron-mine", "wildfire", year, "ssp126"): wildfire,
    }

    result = multi_hazard.build_one_combo(
        files,
        {},
        inventory,
        "iron-mine",
        year,
        "ssp126",
        2023,
        None,
        coverage_policy=policy,
    )
    coverage = multi_hazard.describe_combo_coverage(
        files, "iron-mine", year, "ssp126", policy
    )

    expected_first = 0.125 if year == 2023 else 0.25
    np.testing.assert_array_equal(
        result["production_loss_fraction"].to_numpy()[:2],
        np.array([expected_first, 0.75]),
    )
    assert coverage == {
        "observed_hazards": ["heatwave", "wildfire"],
        "unavailable_hazards": ["tropical-cyclone"],
    }
    assert "tropical-cyclone" not in coverage["observed_hazards"]


def test_ssp126_cyclone_is_required_from_2061(tmp_path: Path):
    evidence = tmp_path / "source_inventory.csv"
    evidence.write_text("first_year,last_year\n2061,2100\n", encoding="utf-8")
    policy = multi_hazard.load_coverage_policy(
        _write_coverage_policy(tmp_path / "coverage.json", evidence)
    )
    with pytest.raises(FileNotFoundError, match="tropical-cyclone"):
        multi_hazard.load_hazard_component(
            {}, {}, _index(1), "iron-mine", "tropical-cyclone", 2061,
            "ssp126", 2023, None, coverage_policy=policy,
        )


@pytest.mark.parametrize(
    ("industry", "hazard", "year", "scenario"),
    [
        ("iron-mine", "tropical-cyclone", 2023, "ssp370"),
        ("iron-mine", "wildfire", 2023, "ssp126"),
        ("copper-mine", "tropical-cyclone", 2023, "ssp126"),
        ("iron-mine", "tropical-cyclone", 2022, "ssp126"),
        ("iron-mine", "tropical-cyclone", 2061, "ssp126"),
    ],
)
def test_coverage_exception_never_generalizes_to_other_missing_components(
    tmp_path: Path,
    industry: str,
    hazard: str,
    year: int,
    scenario: str,
):
    evidence = tmp_path / "source_inventory.csv"
    evidence.write_text("first_year,last_year\n2061,2100\n", encoding="utf-8")
    policy = multi_hazard.load_coverage_policy(
        _write_coverage_policy(tmp_path / "coverage.json", evidence)
    )
    with pytest.raises(FileNotFoundError, match="Missing required"):
        multi_hazard.load_hazard_component(
            {}, {}, _index(1), industry, hazard, year, scenario, 2023, None,
            coverage_policy=policy,
        )


@pytest.mark.parametrize("mutation", ["missing", "extra"])
def test_coverage_policy_rejects_incomplete_or_extra_declaration(
    tmp_path: Path,
    mutation: str,
):
    evidence = tmp_path / "source_inventory.csv"
    evidence.write_text("first_year,last_year\n2061,2100\n", encoding="utf-8")
    keys = _expected_declared_unavailable_keys()
    if mutation == "missing":
        keys.pop()
    else:
        keys.append(("iron-mine", "tropical-cyclone", 2061, "ssp126"))
    components = [
        {
            "industry": industry,
            "hazard": hazard,
            "year": year,
            "scenario": scenario,
            "availability": "unavailable",
        }
        for industry, hazard, year, scenario in keys
    ]
    policy_path = _write_coverage_policy(
        tmp_path / "coverage.json", evidence, unavailable_components=components
    )
    with pytest.raises(ValueError, match="exact 114"):
        multi_hazard.load_coverage_policy(policy_path)


@pytest.mark.parametrize(
    ("source_identity", "recorded_hash", "message"),
    [
        ("other_inventory", None, "source identity"),
        ("authoritative_ssp126_cyclone_inventory_manifest", "0" * 64, "hash"),
    ],
)
def test_coverage_policy_rejects_mismatched_source_identity_or_hash(
    tmp_path: Path,
    source_identity: str,
    recorded_hash: str | None,
    message: str,
):
    evidence = tmp_path / "source_inventory.csv"
    evidence.write_text("first_year,last_year\n2061,2100\n", encoding="utf-8")
    policy_path = _write_coverage_policy(
        tmp_path / "coverage.json",
        evidence,
        source_identity=source_identity,
        recorded_hash=recorded_hash,
    )
    with pytest.raises(ValueError, match=message):
        multi_hazard.load_coverage_policy(policy_path)


def test_strict_combo_coverage_rejects_any_unexpected_missing_file(tmp_path: Path):
    evidence = tmp_path / "source_inventory.csv"
    evidence.write_text("first_year,last_year\n2061,2100\n", encoding="utf-8")
    policy = multi_hazard.load_coverage_policy(
        _write_coverage_policy(tmp_path / "coverage.json", evidence)
    )
    marker = tmp_path / "present.parquet"
    files = {
        (industry, hazard, year, scenario): marker
        for industry in TEST_POLICY_INDUSTRIES
        for scenario in TEST_POLICY_SCENARIOS
        for year in range(2023, 2101)
        for hazard in multi_hazard.SOURCE_HAZARDS
        if (industry, hazard, year, scenario) not in policy["unavailable_keys"]
    }
    del files[("iron-mine", "wildfire", 2050, "ssp370")]

    with pytest.raises(FileNotFoundError, match="Unexpected missing hazard component"):
        multi_hazard.build_combos(
            files, 2023, 2100, None, None, 2023, coverage_policy=policy
        )


def test_strict_combo_coverage_rejects_stale_policy_when_declared_file_appears(tmp_path: Path):
    evidence = tmp_path / "source_inventory.csv"
    evidence.write_text("first_year,last_year\n2061,2100\n", encoding="utf-8")
    policy = multi_hazard.load_coverage_policy(
        _write_coverage_policy(tmp_path / "coverage.json", evidence)
    )
    marker = tmp_path / "present.parquet"
    files = {
        (industry, hazard, 2023, scenario): marker
        for industry in TEST_POLICY_INDUSTRIES
        for scenario in TEST_POLICY_SCENARIOS
        for hazard in multi_hazard.SOURCE_HAZARDS
    }

    with pytest.raises(ValueError, match="unexpectedly observed"):
        multi_hazard.build_combos(
            files, 2023, 2023, None, None, 2023, coverage_policy=policy
        )


def test_strict_combo_coverage_builds_all_936_policy_complete_combinations(tmp_path: Path):
    evidence = tmp_path / "source_inventory.csv"
    evidence.write_text("first_year,last_year\n2061,2100\n", encoding="utf-8")
    policy = multi_hazard.load_coverage_policy(
        _write_coverage_policy(tmp_path / "coverage.json", evidence)
    )
    marker = tmp_path / "present.parquet"
    files = {
        (industry, hazard, year, scenario): marker
        for industry in TEST_POLICY_INDUSTRIES
        for scenario in TEST_POLICY_SCENARIOS
        for year in range(2023, 2101)
        for hazard in multi_hazard.SOURCE_HAZARDS
        if (industry, hazard, year, scenario) not in policy["unavailable_keys"]
    }

    combos = multi_hazard.build_combos(
        files, 2023, 2100, None, None, 2023, coverage_policy=policy
    )

    assert len(combos) == 936
    assert ("iron-mine", 2023, "ssp126") in combos
    assert ("iron-mine", 2061, "ssp126") in combos


@pytest.fixture(scope="module")
def complete_stage_receipt_inputs(tmp_path_factory: pytest.TempPathFactory):
    tmp_path = tmp_path_factory.mktemp("complete-stage-receipt")
    evidence = tmp_path / "source_inventory.csv"
    evidence.write_text("first_year,last_year\n2061,2100\n", encoding="utf-8")
    policy_path = _write_coverage_policy(tmp_path / "coverage.json", evidence)
    policy = multi_hazard.load_coverage_policy(policy_path)
    marker = tmp_path / "source.parquet"
    marker.write_bytes(b"source")
    files = {
        (industry, hazard, year, scenario): marker
        for industry in TEST_POLICY_INDUSTRIES
        for scenario in TEST_POLICY_SCENARIOS
        for year in range(2023, 2101)
        for hazard in multi_hazard.SOURCE_HAZARDS
        if (industry, hazard, year, scenario) not in set(
            _expected_declared_unavailable_keys()
        )
    }
    combos = [
        (industry, year, scenario)
        for industry in sorted(TEST_POLICY_INDUSTRIES)
        for scenario in sorted(TEST_POLICY_SCENARIOS)
        for year in range(2023, 2101)
    ]
    facility_dir = tmp_path / "facility"
    country_dir = tmp_path / "country"
    facility_dir.mkdir()
    country_dir.mkdir()
    output_pairs = []
    for industry, year, scenario in combos:
        facility, country = multi_hazard.output_paths(
            industry,
            "multi-hazard-max",
            year,
            scenario,
            facility_dir,
            country_dir,
        )
        facility.write_bytes(f"facility|{industry}|{year}|{scenario}".encode())
        country.write_bytes(f"country|{industry}|{year}|{scenario}".encode())
        output_pairs.append((facility, country))
    return files, combos, policy, policy_path, output_pairs


def _build_stage_receipt(
    inputs,
    *,
    combos=None,
    output_pairs=None,
    validated_output_combos="complete",
):
    files, complete_combos, policy, policy_path, complete_output_pairs = inputs
    validation_evidence = (
        list(complete_combos)
        if validated_output_combos == "complete"
        else validated_output_combos
    )
    return multi_hazard.build_stage_receipt_payload(
        files=files,
        combos=list(complete_combos if combos is None else combos),
        coverage_policy=policy,
        coverage_policy_path=policy_path,
        output_pairs=list(complete_output_pairs if output_pairs is None else output_pairs),
        validated_output_combos=validation_evidence,
        code_commit="test-commit",
        created_at_utc="2026-08-12T00:00:00+00:00",
    )


def test_stage_receipt_pass_requires_exact_complete_coverage_and_outputs(
    complete_stage_receipt_inputs,
):
    payload = _build_stage_receipt(complete_stage_receipt_inputs)

    assert payload["status"] == "PASS"
    assert payload["coverage"]["complete_under_declared_availability_policy"] is True
    assert payload["coverage"]["complete_three_hazard_source_coverage"] is False
    assert payload["coverage"]["declared_unavailable_component_count"] == 114
    assert payload["coverage"]["three_hazard_combination_count"] == 822
    assert payload["coverage"]["declared_two_hazard_combination_count"] == 114
    assert payload["coverage"]["unexpected_missing_component_count"] == 0
    assert payload["row_counts"] == {
        "combination_count": 936,
        "validated_output_pairs": 936,
        "facility_output_files": 936,
        "country_output_files": 936,
    }
    assert payload["checks"]["all_output_pairs_semantically_validated"] is True
    assert len(payload["output_paths"]) == 1872
    assert len(payload["output_hashes"]) == 1872
    early = next(
        item for item in payload["coverage"]["combinations"]
        if (item["industry"], item["year"], item["scenario"])
        == ("iron-mine", 2023, "ssp126")
    )
    assert early["observed_hazards"] == ["heatwave", "wildfire"]
    assert early["unavailable_hazards"] == ["tropical-cyclone"]


def test_stage_receipt_rejects_absent_output_validation_evidence(
    complete_stage_receipt_inputs,
):
    with pytest.raises(ValueError, match="validated output pairs"):
        _build_stage_receipt(
            complete_stage_receipt_inputs,
            validated_output_combos=None,
        )


@pytest.mark.parametrize("mutation", ["partial", "duplicate", "wrong_set"])
def test_stage_receipt_rejects_invalid_output_validation_evidence(
    complete_stage_receipt_inputs,
    mutation: str,
):
    _, combos, _, _, _ = complete_stage_receipt_inputs
    mutated = list(combos)
    if mutation == "partial":
        mutated.pop()
    elif mutation == "duplicate":
        mutated.append(mutated[0])
    else:
        mutated[-1] = mutated[0]
    with pytest.raises(ValueError, match="validated output pairs"):
        _build_stage_receipt(
            complete_stage_receipt_inputs,
            validated_output_combos=mutated,
        )


@pytest.mark.parametrize("mutation", ["empty", "partial", "duplicate", "wrong_set"])
def test_stage_receipt_rejects_wrong_combo_set(
    complete_stage_receipt_inputs,
    mutation: str,
):
    _, combos, _, _, _ = complete_stage_receipt_inputs
    mutated = list(combos)
    if mutation == "empty":
        mutated = []
    elif mutation == "partial":
        mutated.pop()
    elif mutation == "duplicate":
        mutated.append(mutated[0])
    else:
        mutated[-1] = mutated[0]
    with pytest.raises(ValueError, match="exact 936"):
        _build_stage_receipt(complete_stage_receipt_inputs, combos=mutated)


@pytest.mark.parametrize("mutation", ["empty", "partial", "duplicate", "missing", "extra"])
def test_stage_receipt_rejects_invalid_output_pairs(
    complete_stage_receipt_inputs,
    mutation: str,
):
    _, _, _, _, output_pairs = complete_stage_receipt_inputs
    mutated = list(output_pairs)
    if mutation == "empty":
        mutated = []
    elif mutation == "partial":
        mutated.pop()
    elif mutation == "duplicate":
        mutated[-1] = mutated[0]
    elif mutation == "missing":
        missing = mutated[0][0].parent.parent / "missing" / mutated[0][0].name
        mutated[0] = (missing, mutated[0][1])
    else:
        extra_facility = mutated[0][0].with_name("extra_multi-hazard-max_2023_ssp126.parquet")
        extra_country = mutated[0][1].with_name("extra_multi-hazard-max_2023_ssp126.parquet")
        extra_facility.write_bytes(b"extra facility")
        extra_country.write_bytes(b"extra country")
        mutated.append((extra_facility, extra_country))
    with pytest.raises(ValueError, match="output pair"):
        _build_stage_receipt(complete_stage_receipt_inputs, output_pairs=mutated)


def test_abm_loader_uses_shared_gregorian_helpers():
    import ast

    stage_root = Path(__file__).resolve().parents[1]
    tree = ast.parse((stage_root / "06_dynamic_simulation/11_run_simulation.py").read_text(encoding="utf-8"))
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
        and node.module == "steel_supply.hazard_dynamic.hazard_calendar"
        for alias in node.names
    }

    assert imports == {"align_2023_baseline", "date_from_day_of_year", "iter_dates"}
