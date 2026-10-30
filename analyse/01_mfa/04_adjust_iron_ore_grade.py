#!/usr/bin/env python
# coding: utf-8

"""Step 4: replace the generic SITC 281 factor with exporter ore grades."""

from __future__ import annotations

from pathlib import Path as _DataPath
import sys as _data_sys
_ANALYSE_ROOT = next(p for p in _DataPath(__file__).resolve().parents if (p / 'run_analyse.py').is_file())
if str(_ANALYSE_ROOT / 'library') not in _data_sys.path:
    _data_sys.path.insert(0, str(_ANALYSE_ROOT / 'library'))
from steel_supply.pipeline_paths import data_path, science_path, bundle_root


import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = bundle_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from steel_supply.mfa.ore_grade_adjustment import (
    apply_exporter_ore_grades,
    parse_grade_table_with_qa,
)

FE_PROD_USGS_SHEET_NAME = "铁矿石产量、品位信息_USGS"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=data_path('work/mfa_trade'),
        help="Candidate pipeline work directory used for the step-4 output.",
    )
    parser.add_argument(
        "--input-trade",
        type=Path,
        default=None,
        help=(
            "Optional read-only stage-3 trade file. By default, read "
            "<work-dir>/3_2_weight_filled_trade/trade_data.parquet."
        ),
    )
    parser.add_argument(
        "--workbook",
        type=Path,
        default=data_path('input/mfa/mfa_inputs.xlsx'),
    )
    parser.add_argument(
        "--receipt-path",
        type=Path,
        default=None,
        help=(
            "Row-level SITC 281 grade-resolution receipt. By default, write "
            "beside the adjusted trade output."
        ),
    )
    parser.add_argument(
        "--grade-qa-path",
        type=Path,
        default=None,
        help=(
            "Rejected workbook grade-cell QA table. By default, write beside "
            "the adjusted trade output."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    work_dir = args.work_dir.resolve()
    input_trade = (
        args.input_trade.resolve()
        if args.input_trade is not None
        else work_dir / "3_2_weight_filled_trade" / "trade_data.parquet"
    )
    workbook = args.workbook.resolve()
    output_path = work_dir / "4_iron_ore_grade_adjusted" / "trade_data.parquet"
    receipt_path = (
        args.receipt_path.resolve()
        if args.receipt_path is not None
        else output_path.parent / "ore_grade_resolution_receipt.parquet"
    )
    grade_qa_path = (
        args.grade_qa_path.resolve()
        if args.grade_qa_path is not None
        else output_path.parent / "ore_grade_input_qa.parquet"
    )

    if not input_trade.is_file():
        raise FileNotFoundError(f"Stage-3 trade input not found: {input_trade}")
    if not workbook.is_file():
        raise FileNotFoundError(f"MFA input workbook not found: {workbook}")

    grade_wide = pd.read_excel(workbook, sheet_name=FE_PROD_USGS_SHEET_NAME)
    grade_table, grade_qa = parse_grade_table_with_qa(grade_wide)
    trade = pd.read_parquet(input_trade)
    corrected, receipt = apply_exporter_ore_grades(trade, grade_table)

    expected_receipts = int(trade["commodity_code"].eq(281).sum())
    if len(receipt) != expected_receipts or receipt["fallback_level"].isna().any():
        raise RuntimeError(
            "SITC 281 resolution gate failed: "
            f"expected={expected_receipts}, resolved={len(receipt)}"
        )
    valid_grades = (
        receipt["grade"].notna()
        & receipt["grade"].gt(0)
        & receipt["grade"].le(1)
    )
    roundoff_tolerance = (
        16
        * float.fromhex("0x1.0000000000000p-52")
        * receipt["raw_mass_kg"].abs().clip(lower=1.0)
    )
    mass_is_physical = receipt["corrected_fe_equivalent_mass_kg"].le(
        receipt["raw_mass_kg"] + roundoff_tolerance
    )
    if not valid_grades.all() or not mass_is_physical.all():
        raise RuntimeError(
            "SITC 281 physical-domain gate failed: "
            f"invalid_grades={int((~valid_grades).sum())}, "
            f"mass_above_raw={int((~mass_is_physical).sum())}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    grade_qa_path.parent.mkdir(parents=True, exist_ok=True)
    corrected.to_parquet(output_path, index=False)
    receipt.to_parquet(receipt_path, index=False)
    grade_qa.to_parquet(grade_qa_path, index=False)

    fallback_counts = receipt["fallback_level"].value_counts().sort_index()
    print(f"input_trade={input_trade}")
    print(f"output_trade={output_path}")
    print(f"receipt={receipt_path}")
    print(f"grade_qa={grade_qa_path}")
    print(f"sitc_281_records={expected_receipts}")
    print(f"rejected_grade_cells={len(grade_qa)}")
    for fallback_level, count in fallback_counts.items():
        print(f"fallback.{fallback_level}={int(count)}")
    for reason, count in grade_qa["reason"].value_counts().sort_index().items():
        print(f"grade_qa.{reason}={int(count)}")


if __name__ == "__main__":
    main()
