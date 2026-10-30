from pathlib import Path
import subprocess
import sys

import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'library'))

from steel_supply.mfa.mfa_validation import (  # noqa: E402
    calculate_node_balance,
    compare_mfa,
    validate_mfa_schema,
)


REQUIRED_COLUMNS = [
    "entity_from",
    "entity_to",
    "sector_from",
    "sector_to",
    "mass",
    "unit",
    "extra",
]


def mfa_frame(rows: list[dict[str, object]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=REQUIRED_COLUMNS)


def test_schema_reports_a_missing_required_column():
    frame = mfa_frame([]).drop(columns="sector_to")

    result = validate_mfa_schema(frame)

    assert result["status"] == "FAIL"
    assert result["checks"]["required_columns"] is False
    assert result["missing_required_columns"] == ["sector_to"]


def test_schema_rejects_nonfinite_and_negative_mass():
    frame = mfa_frame(
        [
            {
                "entity_from": "AAA",
                "entity_to": "BBB",
                "sector_from": "M1",
                "sector_to": "M1",
                "mass": float("inf"),
                "unit": "kg",
                "extra": "TRADE_SITC_281",
            },
            {
                "entity_from": "AAA",
                "entity_to": "AAA",
                "sector_from": "M1",
                "sector_to": "P2",
                "mass": -1.0,
                "unit": "kg",
                "extra": None,
            },
        ]
    )

    result = validate_mfa_schema(frame)

    assert result["checks"]["finite_mass"] is False
    assert result["checks"]["nonnegative_mass"] is False
    assert result["invalid_counts"]["nonfinite_mass"] == 1
    assert result["invalid_counts"]["negative_mass"] == 1


def test_schema_requires_every_unit_to_be_kg():
    frame = mfa_frame(
        [
            {
                "entity_from": "AAA",
                "entity_to": "AAA",
                "sector_from": "P1",
                "sector_to": "M1",
                "mass": 1.0,
                "unit": "tonne",
                "extra": None,
            }
        ]
    )

    result = validate_mfa_schema(frame)

    assert result["checks"]["unit_kg"] is False
    assert result["invalid_counts"]["non_kg_unit"] == 1


def test_schema_accepts_only_the_declared_phantom_and_correction_labels():
    valid = mfa_frame(
        [
            {
                "entity_from": "AAA",
                "entity_to": "AAA",
                "sector_from": "Phantom Flow",
                "sector_to": "M1",
                "mass": 1.0,
                "unit": "kg",
                "extra": "Phantom Flow Correction",
            },
            {
                "entity_from": "AAA",
                "entity_to": "AAA",
                "sector_from": "M1",
                "sector_to": "P2",
                "mass": 1.0,
                "unit": "kg",
                "extra": "Process Balance Adjustment",
            },
            {
                "entity_from": "AAA",
                "entity_to": "AAA",
                "sector_from": "M8",
                "sector_to": "S1",
                "mass": 1.0,
                "unit": "kg",
                "extra": "ASU-FSP final demand floor for non-positive raw M8 balance",
            },
        ]
    )
    invalid = valid.copy()
    invalid.loc[1, "extra"] = "Undeclared Correction"

    valid_result = validate_mfa_schema(valid)
    invalid_result = validate_mfa_schema(invalid)

    assert valid_result["checks"]["correction_labels"] is True
    assert invalid_result["checks"]["correction_labels"] is False
    assert invalid_result["invalid_correction_labels"] == ["Undeclared Correction"]


def test_schema_rejects_a_phantom_label_on_a_nonphantom_edge():
    frame = mfa_frame(
        [
            {
                "entity_from": "AAA",
                "entity_to": "AAA",
                "sector_from": "M1",
                "sector_to": "P2",
                "mass": 1.0,
                "unit": "kg",
                "extra": "Phantom Flow Correction",
            }
        ]
    )

    result = validate_mfa_schema(frame)

    assert result["checks"]["phantom_edge_labels"] is False
    assert result["invalid_counts"]["phantom_edge_label"] == 1


def test_schema_requires_unique_descriptive_row_keys():
    row = {
        "entity_from": "AAA",
        "entity_to": "BBB",
        "sector_from": "M1",
        "sector_to": "M1",
        "mass": 1.0,
        "unit": "kg",
        "extra": "TRADE_SITC_281",
    }
    frame = mfa_frame([row, row | {"mass": 2.0}])

    result = validate_mfa_schema(frame)

    assert result["checks"]["unique_row_keys"] is False
    assert result["invalid_counts"]["duplicate_row_keys"] == 2


def test_schema_detects_duplicate_row_keys_when_extra_is_none():
    row = {
        "entity_from": "AAA",
        "entity_to": "AAA",
        "sector_from": "P1",
        "sector_to": "M1",
        "mass": 1.0,
        "unit": "kg",
        "extra": None,
    }
    frame = mfa_frame([row, row | {"mass": 2.0}])

    result = validate_mfa_schema(frame)

    assert result["checks"]["unique_row_keys"] is False
    assert result["invalid_counts"]["duplicate_row_keys"] == 2


def test_schema_rejects_process_balance_label_on_an_undeclared_edge():
    frame = mfa_frame(
        [
            {
                "entity_from": "AAA",
                "entity_to": "AAA",
                "sector_from": "P1",
                "sector_to": "M1",
                "mass": 1.0,
                "unit": "kg",
                "extra": "Process Balance Adjustment",
            }
        ]
    )

    result = validate_mfa_schema(frame)

    assert result["checks"]["correction_labels"] is False
    assert result["invalid_counts"]["process_balance_edge_label"] == 1


def test_schema_rejects_asu_fsp_label_outside_the_domestic_m8_s1_edge():
    frame = mfa_frame(
        [
            {
                "entity_from": "AAA",
                "entity_to": "BBB",
                "sector_from": "M7",
                "sector_to": "S1",
                "mass": 1.0,
                "unit": "kg",
                "extra": "ASU-FSP final demand floor for non-positive raw M8 balance",
            }
        ]
    )

    result = validate_mfa_schema(frame)

    assert result["checks"]["correction_labels"] is False
    assert result["invalid_counts"]["asu_fsp_edge_label"] == 1


def test_schema_reports_global_bilateral_trade_balance_by_material():
    valid = mfa_frame(
        [
            {
                "entity_from": "AAA",
                "entity_to": "BBB",
                "sector_from": "M1",
                "sector_to": "M1",
                "mass": 10.0,
                "unit": "kg",
                "extra": "TRADE_SITC_281",
            },
            {
                "entity_from": "CCC",
                "entity_to": "AAA",
                "sector_from": "M2",
                "sector_to": "M2",
                "mass": 7.0,
                "unit": "kg",
                "extra": "TRADE_SITC_671",
            },
        ]
    )
    cross_material = valid.copy()
    cross_material.loc[1, "sector_to"] = "M3"

    valid_result = validate_mfa_schema(valid)
    invalid_result = validate_mfa_schema(cross_material)

    assert valid_result["checks"]["bilateral_trade_edges"] is True
    assert valid_result["checks"]["global_bilateral_balance_by_material"] is True
    assert valid_result["bilateral_balance"] == [
        {
            "material": "M1",
            "export_mass_kg": 10.0,
            "import_mass_kg": 10.0,
            "net_mass_kg": 0.0,
        },
        {
            "material": "M2",
            "export_mass_kg": 7.0,
            "import_mass_kg": 7.0,
            "net_mass_kg": 0.0,
        },
    ]
    assert invalid_result["checks"]["bilateral_trade_edges"] is False
    assert invalid_result["checks"]["global_bilateral_balance_by_material"] is False


def test_calculate_node_balance_reports_country_sector_inflow_and_outflow():
    frame = mfa_frame(
        [
            {
                "entity_from": "AAA",
                "entity_to": "AAA",
                "sector_from": "P1",
                "sector_to": "M1",
                "mass": 10.0,
                "unit": "kg",
                "extra": None,
            },
            {
                "entity_from": "AAA",
                "entity_to": "AAA",
                "sector_from": "M1",
                "sector_to": "P2",
                "mass": 10.0,
                "unit": "kg",
                "extra": None,
            },
        ]
    )

    balances = calculate_node_balance(frame)

    m1 = balances.set_index(["entity", "sector"]).loc[("AAA", "M1")]
    p1 = balances.set_index(["entity", "sector"]).loc[("AAA", "P1")]
    p2 = balances.set_index(["entity", "sector"]).loc[("AAA", "P2")]
    assert m1.to_dict() == {
        "inflow_kg": 10.0,
        "outflow_kg": 10.0,
        "net_inflow_kg": 0.0,
        "abs_imbalance_kg": 0.0,
        "is_boundary": False,
        "is_balanced": True,
    }
    assert p1["is_boundary"]
    assert not p2["is_boundary"]
    assert p2["net_inflow_kg"] == 10.0
    assert not p2["is_balanced"]


def test_compare_mfa_separates_trade_market_and_reconciliation_impacts():
    reference = mfa_frame(
        [
            {
                "entity_from": "AAA",
                "entity_to": "BBB",
                "sector_from": "M1",
                "sector_to": "M1",
                "mass": 100.0,
                "unit": "kg",
                "extra": "TRADE_SITC_281",
            },
            {
                "entity_from": "AAA",
                "entity_to": "AAA",
                "sector_from": "M1",
                "sector_to": "Phantom Flow",
                "mass": 10.0,
                "unit": "kg",
                "extra": "Phantom Flow Correction",
            },
            {
                "entity_from": "AAA",
                "entity_to": "AAA",
                "sector_from": "P2",
                "sector_to": "M2",
                "mass": 5.0,
                "unit": "kg",
                "extra": "Process Balance Adjustment",
            },
        ]
    )
    candidate = reference.copy()
    candidate.loc[0, "mass"] = 120.0
    candidate.loc[1, "mass"] = 30.0

    comparisons = compare_mfa(reference, candidate)

    assert set(comparisons) == {
        "row_key",
        "edge",
        "country",
        "stage",
        "material_market",
        "correction_label",
    }
    market = comparisons["material_market"].set_index("material_market")
    country = comparisons["country"].set_index(["entity", "direction"])
    corrections = comparisons["correction_label"].set_index("correction_label")
    assert market.loc["M1", "mass_delta_kg"] == 20.0
    assert country.loc[("BBB", "inflow"), "mass_delta_kg"] == 20.0
    assert corrections.loc["Phantom Flow Correction", "mass_delta_kg"] == 20.0
    assert corrections.loc["Process Balance Adjustment", "mass_delta_kg"] == 0.0


def test_builder_help_exposes_optional_receipt_path():
    builder = Path(__file__).resolve().parents[1] / "01_mfa/05_build_mfa.py"

    completed = subprocess.run(
        [sys.executable, str(builder), "--help"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )

    assert "--receipt-path" in completed.stdout
