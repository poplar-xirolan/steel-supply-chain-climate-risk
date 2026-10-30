from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import pytest
from pathlib import Path

import numpy as np
import pandas as pd


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'library'))

from steel_supply.mfa.trade_equivalent_conversion import (  # noqa: E402
    build_interpolated_price_table,
    deterministic_trade_order,
    fill_trade_values,
    plot_unit_price_trends,
    sum_weight_and_money,
    ulp_compatibility,
    ulp_distance_float64,
)


SMALL_VALUE = 1e-5
SCRIPT = Path(__file__).resolve().parents[1] / "01_mfa/03_convert_trade_equivalent.py"
IRON_EQ_SHEET = "<参考资料>铁SITC 1贸易代码"


def sample_trade() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "commodity_code": [281, 281, 281, 281, 281, 2820, 2820],
            "trade_year": [2020, 2020, 2020, 2021, 2022, 2020, 2021],
            "trade_flow": ["X", "X", "X", "X", "X", "X", "X"],
            "reporter_iso_a3": ["AAA", "AAA", "AAA", "AAA", "AAA", "BBB", "BBB"],
            "partner_iso_a3": ["BBB", "BBB", "BBB", "BBB", "BBB", "AAA", "AAA"],
            "weight_unified_kg": [10.0, 10.0, -0.0, 20.0, 0.0, 4.0, 8.0],
            "monetary_usd": [50.0, 50.0, 25.0, 120.0, 35.0, 8.0, 24.0],
        }
    )


def assert_float_columns_bit_identical(
    left: pd.DataFrame, right: pd.DataFrame, columns: list[str]
) -> None:
    for column in columns:
        left_bits = left[column].to_numpy(dtype=np.float64).view(np.uint64)
        right_bits = right[column].to_numpy(dtype=np.float64).view(np.uint64)
        np.testing.assert_array_equal(left_bits, right_bits)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def write_fake_matplotlib(fake_root: Path) -> None:
    package = fake_root / "matplotlib"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text(
        "from . import pyplot\n", encoding="utf-8"
    )
    (package / "pyplot.py").write_text(
        """import os
from pathlib import Path

EVENTS = Path(os.environ["FAKE_MATPLOTLIB_EVENTS"])

def _record(event):
    with EVENTS.open("a", encoding="utf-8") as stream:
        stream.write(event + "\\n")

_record("IMPORT")

class Figure:
    pass

class Axis:
    def plot(self, *_args, **_kwargs): _record("AXIS_PLOT")
    def set_title(self, *_args, **_kwargs): pass
    def set_xlabel(self, *_args, **_kwargs): pass
    def set_ylabel(self, *_args, **_kwargs): pass
    def grid(self, *_args, **_kwargs): pass

def subplots():
    _record("SUBPLOTS")
    return Figure(), Axis()

def show(*_args, **kwargs):
    _record(f"SHOW block={kwargs.get('block')}")

def pause(_seconds):
    _record("PAUSE")

def close(_figure):
    _record("CLOSE")

# Legacy script API, used only for symptom-specific RED verification.
def figure(): _record("FIGURE")
def plot(*_args, **_kwargs): _record("PLOT")
def title(*_args, **_kwargs): pass
def xlabel(*_args, **_kwargs): pass
def ylabel(*_args, **_kwargs): pass
def grid(*_args, **_kwargs): pass
""",
        encoding="utf-8",
    )


def write_cli_fixture(work_dir: Path, workbook: Path) -> Path:
    stage2 = work_dir / "2_country_mapped" / "trade_data.parquet"
    stage2.parent.mkdir(parents=True)
    sample_trade().to_parquet(stage2, index=False)
    iron_equivalence = pd.DataFrame(
        {
            "SITCCode": [281, 2820],
            "IronConcentration (Unit:1)": [0.6, 0.9],
        }
    )
    with pd.ExcelWriter(workbook) as writer:
        iron_equivalence.to_excel(writer, sheet_name=IRON_EQ_SHEET, index=False)
    return stage2


def run_conversion_cli(
    script: Path,
    work_dir: Path,
    workbook: Path,
    fake_root: Path,
    event_path: Path,
    plot: bool,
) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        str(script),
        "--work-dir",
        str(work_dir),
        "--workbook",
        str(workbook),
    ]
    if plot:
        command.append("--plot-unit-prices")
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(fake_root), str(SCRIPT.parent), environment.get("PYTHONPATH", "")]
    )
    environment["FAKE_MATPLOTLIB_EVENTS"] = str(event_path)
    return subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )


def test_shuffled_rows_produce_bit_identical_aggregates_prices_and_fills():
    trade = sample_trade()
    shuffled = trade.sample(frac=1, random_state=712).reset_index(drop=True)

    ordered = deterministic_trade_order(trade)
    shuffled_ordered = deterministic_trade_order(shuffled)
    pd.testing.assert_frame_equal(ordered, shuffled_ordered, check_exact=True)
    assert_float_columns_bit_identical(
        ordered, shuffled_ordered, ["weight_unified_kg", "monetary_usd"]
    )

    aggregate = sum_weight_and_money(ordered, 281, 2020, SMALL_VALUE)
    shuffled_aggregate = sum_weight_and_money(
        shuffled_ordered, 281, 2020, SMALL_VALUE
    )
    assert np.asarray(aggregate, dtype=np.float64).view(np.uint64).tolist() == (
        np.asarray(shuffled_aggregate, dtype=np.float64).view(np.uint64).tolist()
    )

    prices = build_interpolated_price_table(ordered, SMALL_VALUE)
    shuffled_prices = build_interpolated_price_table(
        shuffled_ordered, SMALL_VALUE
    )
    pd.testing.assert_frame_equal(prices, shuffled_prices, check_exact=True)
    assert_float_columns_bit_identical(
        prices,
        shuffled_prices,
        [
            "weight_unified_kg",
            "monetary_usd",
            "unit_price_usd_per_kg_interpolated",
        ],
    )

    filled = fill_trade_values(ordered, prices, SMALL_VALUE)
    shuffled_filled = fill_trade_values(
        shuffled_ordered, shuffled_prices, SMALL_VALUE
    )
    pd.testing.assert_frame_equal(filled, shuffled_filled, check_exact=True)
    assert_float_columns_bit_identical(
        filled,
        shuffled_filled,
        [
            "weight_unified_kg",
            "monetary_usd",
            "unit_price_usd_per_kg_interpolated",
        ],
    )


def test_ulp_distance_handles_adjacent_values_negatives_and_signed_zero():
    values = np.array([1.0, -1.0, 0.0, -0.0])
    adjacent = np.array(
        [
            np.nextafter(1.0, np.inf),
            np.nextafter(-1.0, -np.inf),
            -0.0,
            0.0,
        ]
    )

    assert ulp_distance_float64(values, adjacent).tolist() == [1, 1, 0, 0]


def test_ulp_compatibility_accepts_16_ulps_and_rejects_17():
    reference = np.array([1.0, -1.0, 0.0, np.inf])
    candidate = reference.copy()
    for _ in range(16):
        candidate[0] = np.nextafter(candidate[0], np.inf)
    for _ in range(17):
        candidate[1] = np.nextafter(candidate[1], -np.inf)
    candidate[2] = -0.0

    result = ulp_compatibility(reference, candidate, max_ulps=16)

    assert result == {
        "changed_cell_count": 2,
        "max_ulp_distance": 17,
        "violation_count": 1,
    }


def test_ulp_compatibility_rejects_nonfinite_mismatch():
    result = ulp_compatibility(
        np.array([np.inf, -np.inf, np.nan]),
        np.array([-np.inf, np.inf, np.nan]),
        max_ulps=16,
    )

    assert result["changed_cell_count"] == 2
    assert result["violation_count"] == 2
    assert result["max_ulp_distance"] is None


def test_price_plotting_is_disabled_by_default(monkeypatch):
    price_table = build_interpolated_price_table(
        deterministic_trade_order(sample_trade()), SMALL_VALUE
    )

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("batch default must not create or show plot windows")

    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", None)


    assert plot_unit_price_trends(price_table) == 0


def test_opt_in_price_plotting_is_nonblocking_and_closes_figures(monkeypatch):
    price_table = build_interpolated_price_table(
        deterministic_trade_order(sample_trade()), SMALL_VALUE
    ).rename(
        columns={
            "unit_price_usd_per_kg_interpolated": "unit_price_usd_per_kg"
        }
    )
    plt = pytest.importorskip("matplotlib.pyplot", reason="optional price diagnostic renderer")

    show_blocks = []
    monkeypatch.setattr(
        plt, "show", lambda *args, **kwargs: show_blocks.append(kwargs.get("block"))
    )
    monkeypatch.setattr(plt, "pause", lambda _seconds: None)

    plotted = plot_unit_price_trends(price_table, enabled=True)

    assert plotted == price_table["commodity_code"].nunique()
    assert show_blocks == [False] * plotted
    assert plt.get_fignums() == []


def test_conversion_cli_defaults_to_no_plots_and_opt_in_is_nonblocking(tmp_path):
    script = Path(os.environ.get("TRADE_EQUIVALENT_SCRIPT_UNDER_TEST", SCRIPT))
    fake_root = tmp_path / "fake_packages"
    write_fake_matplotlib(fake_root)
    workbook = tmp_path / "mfa_inputs.xlsx"
    default_work = tmp_path / "default" / "mfa_trade"
    opt_in_work = tmp_path / "opt_in" / "mfa_trade"
    default_input = write_cli_fixture(default_work, workbook)
    opt_in_input = opt_in_work / "2_country_mapped" / "trade_data.parquet"
    opt_in_input.parent.mkdir(parents=True)
    opt_in_input.write_bytes(default_input.read_bytes())
    default_input_before = sha256_file(default_input)
    opt_in_input_before = sha256_file(opt_in_input)

    default_events = tmp_path / "default_events.log"
    default_run = run_conversion_cli(
        script, default_work, workbook, fake_root, default_events, plot=False
    )
    assert default_run.returncode == 0, default_run.stderr
    assert not default_events.exists(), (
        "batch default imported or invoked matplotlib: "
        f"{default_events.read_text(encoding='utf-8')}"
    )

    opt_in_events = tmp_path / "opt_in_events.log"
    opt_in_run = run_conversion_cli(
        script, opt_in_work, workbook, fake_root, opt_in_events, plot=True
    )
    assert opt_in_run.returncode == 0, opt_in_run.stderr
    events = opt_in_events.read_text(encoding="utf-8").splitlines()
    commodity_count = sample_trade()["commodity_code"].nunique()
    assert events.count("IMPORT") == 1
    assert events.count("SUBPLOTS") == commodity_count
    assert events.count("SHOW block=False") == commodity_count
    assert events.count("CLOSE") == commodity_count
    assert "SHOW block=None" not in events

    assert sha256_file(default_input) == default_input_before
    assert sha256_file(opt_in_input) == opt_in_input_before
    default_output = pd.read_parquet(
        default_work / "3_2_weight_filled_trade" / "trade_data.parquet"
    )
    opt_in_output = pd.read_parquet(
        opt_in_work / "3_2_weight_filled_trade" / "trade_data.parquet"
    )
    pd.testing.assert_frame_equal(default_output, opt_in_output, check_exact=True)
