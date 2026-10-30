from pathlib import Path
import subprocess
import sys

import pandas as pd
import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'library'))

from steel_supply.mfa import ore_grade_adjustment as ore_grades  # noqa: E402
from steel_supply.mfa.ore_grade_adjustment import (  # noqa: E402
    apply_exporter_ore_grades,
    parse_grade_table,
    resolve_grade,
)


def synthetic_grades(year=2023, **country_grades):
    return pd.DataFrame(
        {
            "country": list(country_grades),
            "year": [year] * len(country_grades),
            "grade": list(country_grades.values()),
        }
    )


def test_parse_grade_table_extracts_only_nonmissing_fe_equivalent_values():
    frame = pd.DataFrame(
        {
            "Country": ["Exporter B", "World"],
            "Country_code_iso3166-2_alpha_3": ["BBB", "Total"],
            "2023 铁矿重量": [200.0, 400.0],
            "2023 Fe eq（以1计）": [0.65, 0.55],
            "2022 Fe eq（以1计）": [None, 0.54],
        }
    )

    parsed = parse_grade_table(frame)

    expected = pd.DataFrame(
        {
            "country": ["BBB", "Total", "Total"],
            "year": [2023, 2022, 2023],
            "grade": [0.65, 0.54, 0.55],
        }
    )
    pd.testing.assert_frame_equal(parsed, expected)


@pytest.mark.parametrize(
    ("raw_value", "reason"),
    [
        (1.01, "greater_than_one"),
        (0.0, "nonpositive"),
        (-0.1, "nonpositive"),
        (float("inf"), "nonfinite"),
        (float("-inf"), "nonfinite"),
    ],
)
def test_out_of_domain_grade_is_rejected_and_recorded(raw_value, reason):
    frame = pd.DataFrame(
        {
            "Country_code_iso3166-2_alpha_3": ["BAD", "Total"],
            "2023 Fe eq（以1计）": [raw_value, 0.55],
        }
    )

    usable, rejected = ore_grades.parse_grade_table_with_qa(frame)

    assert usable.to_dict(orient="records") == [
        {"country": "Total", "year": 2023, "grade": 0.55}
    ]
    assert rejected.loc[0, "country"] == "BAD"
    assert rejected.loc[0, "year"] == 2023
    assert rejected.loc[0, "raw_value"] == raw_value
    assert rejected.loc[0, "reason"] == reason


def test_argentina_like_grade_is_rejected_then_uses_nearest_exporter_year():
    frame = pd.DataFrame(
        {
            "Country_code_iso3166-2_alpha_3": ["ARG", "Total"],
            "2017 Fe eq（以1计）": [5.93942957953543, 0.55],
            "2016 Fe eq（以1计）": [0.62, 0.54],
        }
    )

    usable, rejected = ore_grades.parse_grade_table_with_qa(frame)
    resolved = resolve_grade("ARG", 2017, usable)

    assert resolved == (0.62, "exporter_nearest_year", "ARG", 2016)
    assert rejected.to_dict(orient="records") == [
        {
            "country": "ARG",
            "year": 2017,
            "raw_value": 5.93942957953543,
            "reason": "greater_than_one",
        }
    ]


def test_applied_grades_and_corrected_mass_remain_in_physical_domain():
    trade = pd.DataFrame(
        {
            "commodity_code": [281],
            "trade_year": [2023],
            "reporter_iso_a3": ["AAA"],
            "partner_iso_a3": ["BAD"],
            "weight_unified_kg": [100.0],
            "iron_eq_ratio_unit_1": [0.5],
        }
    )
    frame = pd.DataFrame(
        {
            "Country_code_iso3166-2_alpha_3": ["BAD", "Total"],
            "2023 Fe eq（以1计）": [1.25, 0.55],
        }
    )
    usable, _ = ore_grades.parse_grade_table_with_qa(frame)

    corrected, receipt = apply_exporter_ore_grades(trade, usable)

    assert receipt.loc[0, "grade"] == 0.55
    assert 0 < corrected.loc[0, "iron_eq_ratio_unit_1"] <= 1
    assert corrected.loc[0, "iron_eq_weight_unified_kg"] <= 100.0


def test_sitc_281_uses_exporter_not_importer():
    trade = pd.DataFrame(
        {
            "commodity_code": [281],
            "trade_year": [2023],
            "reporter_iso_a3": ["AAA"],
            "partner_iso_a3": ["BBB"],
            "weight_unified_kg": [100.0],
            "iron_eq_ratio_unit_1": [0.5],
        }
    )
    grades = synthetic_grades(AAA=0.40, BBB=0.65, Total=0.55)

    corrected, receipt = apply_exporter_ore_grades(trade, grades)

    assert corrected.loc[0, "iron_eq_ratio_unit_1"] == 0.65
    assert corrected.loc[0, "iron_eq_weight_unified_kg"] == 65.0
    assert receipt.loc[0, "fallback_level"] == "exporter_exact_year"
    assert receipt.loc[0, "importer_iso_a3"] == "AAA"
    assert receipt.loc[0, "exporter_iso_a3"] == "BBB"


def test_nearest_exporter_year_breaks_ties_toward_earlier_year():
    grades = pd.DataFrame(
        {
            "country": ["BBB", "BBB", "Total"],
            "year": [2021, 2023, 2022],
            "grade": [0.61, 0.69, 0.55],
        }
    )

    resolved = resolve_grade("BBB", 2022, grades)

    assert resolved == (0.61, "exporter_nearest_year", "BBB", 2021)


def test_unknown_exporter_uses_exact_world_year():
    grades = synthetic_grades(BBB=0.65, Total=0.55)

    resolved = resolve_grade("ZZZ", 2023, grades)

    assert resolved == (0.55, "world_exact_year", "Total", 2023)


def test_nearest_world_year_breaks_ties_toward_earlier_year():
    grades = pd.DataFrame(
        {
            "country": ["Total", "Total"],
            "year": [2021, 2023],
            "grade": [0.53, 0.57],
        }
    )

    resolved = resolve_grade("ZZZ", 2022, grades)

    assert resolved == (0.53, "world_nearest_year", "Total", 2021)


def test_non_281_rows_are_not_adjusted_or_receipted():
    trade = pd.DataFrame(
        {
            "commodity_code": [282],
            "trade_year": [2023],
            "reporter_iso_a3": ["AAA"],
            "partner_iso_a3": ["BBB"],
            "weight_unified_kg": [100.0],
            "iron_eq_ratio_unit_1": [0.5],
            "iron_eq_weight_unified_kg": [50.0],
        }
    )

    corrected, receipt = apply_exporter_ore_grades(
        trade, synthetic_grades(BBB=0.65, Total=0.55)
    )

    pd.testing.assert_frame_equal(corrected, trade)
    assert receipt.empty


def test_unresolved_exporter_code_is_preserved_in_world_fallback_receipt():
    trade = pd.DataFrame(
        {
            "commodity_code": [281],
            "trade_year": [2023],
            "reporter_iso_a3": ["AAA"],
            "partner_iso_a3": ["ZZZ"],
            "weight_unified_kg": [100.0],
            "iron_eq_ratio_unit_1": [0.5],
        }
    )

    corrected, receipt = apply_exporter_ore_grades(
        trade, synthetic_grades(Total=0.55)
    )

    assert corrected.loc[0, "iron_eq_ratio_unit_1"] == 0.55
    assert receipt.loc[0, "exporter_iso_a3"] == "ZZZ"
    assert receipt.loc[0, "fallback_level"] == "world_exact_year"
    assert receipt.loc[0, "matched_country"] == "Total"


def test_resolution_fails_when_no_exporter_or_world_fallback_exists():
    grades = synthetic_grades(BBB=0.65)

    with pytest.raises(ValueError, match="No ore grade fallback"):
        resolve_grade("ZZZ", 2023, grades)


def test_cli_reads_separate_input_and_writes_trade_and_receipt(tmp_path):
    script = Path(__file__).resolve().parents[1] / "01_mfa/04_adjust_iron_ore_grade.py"
    input_trade = tmp_path / "upstream" / "trade_data.parquet"
    input_trade.parent.mkdir()
    pd.DataFrame(
        {
            "commodity_code": [281],
            "trade_year": [2023],
            "reporter_iso_a3": ["AAA"],
            "partner_iso_a3": ["BBB"],
            "weight_unified_kg": [100.0],
            "monetary_usd": [10.0],
            "unit_price_usd_per_kg_interpolated": [0.1],
            "iron_eq_ratio_unit_1": [0.5],
            "iron_eq_weight_unified_kg": [50.0],
        }
    ).to_parquet(input_trade, index=False)
    workbook = tmp_path / "mfa_inputs.xlsx"
    grade_frame = pd.DataFrame(
        {
            "Country": ["Importer A", "Exporter B", "Invalid", "World"],
            "Country_code_iso3166-2_alpha_3": ["AAA", "BBB", "BAD", "Total"],
            "2023 Fe eq（以1计）": [0.40, 0.65, 1.25, 0.55],
        }
    )
    with pd.ExcelWriter(workbook) as writer:
        grade_frame.to_excel(
            writer, sheet_name="铁矿石产量、品位信息_USGS", index=False
        )
    work_dir = tmp_path / "candidate" / "mfa_trade"
    receipt_path = tmp_path / "candidate" / "ore_grade_receipt.parquet"
    grade_qa_path = tmp_path / "candidate" / "ore_grade_input_qa.parquet"

    completed = subprocess.run(
        [
            sys.executable,
            str(script),
            "--work-dir",
            str(work_dir),
            "--input-trade",
            str(input_trade),
            "--workbook",
            str(workbook),
            "--receipt-path",
            str(receipt_path),
            "--grade-qa-path",
            str(grade_qa_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    output = pd.read_parquet(
        work_dir / "4_iron_ore_grade_adjusted" / "trade_data.parquet"
    )
    receipt = pd.read_parquet(receipt_path)
    grade_qa = pd.read_parquet(grade_qa_path)
    assert output.loc[0, "iron_eq_ratio_unit_1"] == 0.65
    assert output.loc[0, "iron_eq_weight_unified_kg"] == 65.0
    assert receipt.loc[0, "fallback_level"] == "exporter_exact_year"
    assert grade_qa.to_dict(orient="records") == [
        {
            "country": "BAD",
            "year": 2023,
            "raw_value": 1.25,
            "reason": "greater_than_one",
        }
    ]
