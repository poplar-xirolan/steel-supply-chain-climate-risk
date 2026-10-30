from __future__ import annotations

import json
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1] / 'library'
sys.path.insert(0, str(PROJECT_ROOT))

from steel_supply.abm_figure_utils import (  # noqa: E402
    add_baseline_loss,
    aggregate_step_mass,
    annualize_step_totals,
    load_global_s1_step_totals,
    seasonal_step_series,
    select_run_folders,
    steps_overlapping_date_range,
)




class RunSelectionTests(unittest.TestCase):
    def _folder_name(self, sensitivity_id: str, trade: str = "free-trade") -> str:
        return (
            "multi-hazard-max_ssp370_abun-cap1.1_"
            f"{trade}_inv4.0_2023-2100_sd7_sensitivity-id-{sensitivity_id}"
        )

    def test_deterministic_scenarios_select_lexicographic_first(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            for sensitivity_id in ["bbbbbbbb", "aaaaaaaa", "cccccccc"]:
                (root / self._folder_name(sensitivity_id)).mkdir()
            with self.assertWarns(UserWarning):
                selected = select_run_folders(
                    root,
                    hazard_type="multi-hazard-max",
                    climate_scenario="ssp370",
                    trade_scenario="free-trade",
                    inventory_ratio=4,
                    capacity_ratio=1.1,
                )
            self.assertEqual([path.name for path in selected], [self._folder_name("aaaaaaaa")])

    def test_random_scenarios_keep_all_replicates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            trade = "random-disruption0.1"
            for sensitivity_id in ["bbbbbbbb", "aaaaaaaa", "cccccccc"]:
                (root / self._folder_name(sensitivity_id, trade=trade)).mkdir()
            selected = select_run_folders(
                root,
                hazard_type="multi-hazard-max",
                climate_scenario="ssp370",
                trade_scenario="random-disruption",
                random_disruption_ratio=0.1,
                inventory_ratio=4,
                capacity_ratio=1.1,
            )
            self.assertEqual(len(selected), 3)

    def test_formal_grid_nested_attempt_outputs_are_discovered(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            run = (
                root
                / "scenario_001_aaaaaaaaaaaa"
                / "attempt-0001"
                / "engine_output"
                / self._folder_name("aaaaaaaa")
            )
            run.mkdir(parents=True)
            (run.parent.parent / "PASS.json").write_text("{}", encoding="utf-8")
            selected = select_run_folders(
                root,
                hazard_type="multi-hazard-max",
                climate_scenario="ssp370",
                trade_scenario="free-trade",
                inventory_ratio=4,
                capacity_ratio=1.1,
            )
            self.assertEqual(selected, [run])


class AggregationTests(unittest.TestCase):
    def test_global_loader_batches_files_and_sums_s1_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            run = root / (
                "multi-hazard-max_ssp370_abun-cap1.1_free-trade_"
                "inv4.0_2023-2100_sd7_sensitivity-id-aaaaaaaa"
            )
            flows = run / "flows"
            flows.mkdir(parents=True)
            for step, year, masses in [(0, 2023, [40.0, 60.0]), (1, 2023, [25.0, 75.0])]:
                pd.DataFrame(
                    {
                        "sector_to": ["S1", "S1", "P2"],
                        "mass": [*masses, 999.0],
                        "year": [year, year, year],
                        "step": [step, step, step],
                    }
                ).to_parquet(flows / f"flows_step_{step}.parquet", index=False)
            result = load_global_s1_step_totals(
                root,
                hazard_type="multi-hazard-max",
                climate_scenario="ssp370",
                trade_scenario="free-trade",
                inventory_ratio=4,
                capacity_ratio=1.1,
                start_date="2023-01-01",
                end_date="2023-01-14",
            )
            self.assertEqual(result["step"].tolist(), [0, 1])
            self.assertEqual(result["mass"].tolist(), [100.0, 100.0])

    def test_global_loader_uses_actual_six_day_terminal_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            run = root / (
                "multi-hazard-max_ssp370_abun-cap1.1_free-trade_"
                "inv4.0_2023-2100_sd7_sensitivity-id-aaaaaaaa"
            )
            flows = run / "flows"
            flows.mkdir(parents=True)
            pd.DataFrame(
                {
                    "sector_to": ["S1", "S1", "P2"],
                    "mass": [20.0, 40.0, 999.0],
                    "year": [2100, 2100, 2100],
                    "step": [4069, 4069, 4069],
                    "step_start_date": ["2100-12-26"] * 3,
                    "step_end_date_exclusive": ["2101-01-01"] * 3,
                    "step_days": [6] * 3,
                    "time_scale_ratio": [6.0 / 365.25] * 3,
                }
            ).to_parquet(flows / "flows_step_4069.parquet", index=False)
            result = load_global_s1_step_totals(
                root,
                hazard_type="multi-hazard-max",
                climate_scenario="ssp370",
                trade_scenario="free-trade",
                inventory_ratio=4,
                capacity_ratio=1.1,
                start_date="2100-12-26",
                end_date="2100-12-31",
            )
            self.assertEqual(result["step_end_date"].iloc[0], pd.Timestamp("2101-01-01"))
            annual = annualize_step_totals(result)
            self.assertAlmostEqual(float(annual["annual_mass"].iloc[0]), 60.0)
            self.assertAlmostEqual(float(annual["mean_step_mass"].iloc[0]), 70.0)

    def test_flow_row_splitting_does_not_change_step_total(self) -> None:
        frame = pd.DataFrame(
            {
                "sensitivity_id": ["run", "run", "run"],
                "step": [0, 0, 1],
                "step_start_date": pd.to_datetime(["2023-01-01", "2023-01-01", "2023-01-08"]),
                "step_end_date": pd.to_datetime(["2023-01-08", "2023-01-08", "2023-01-15"]),
                "range_overlap_start": pd.to_datetime(["2023-01-01", "2023-01-01", "2023-01-08"]),
                "range_overlap_end": pd.to_datetime(["2023-01-08", "2023-01-08", "2023-01-15"]),
                "range_overlap_days": [7, 7, 7],
                "mass": [60.0, 40.0, 100.0],
            }
        )
        totals = aggregate_step_mass(frame)
        self.assertEqual(totals["mass"].tolist(), [100.0, 100.0])
        annual = annualize_step_totals(totals)
        self.assertAlmostEqual(float(annual.loc[annual["year"] == 2023, "mean_step_mass"].iloc[0]), 100.0)

    def test_calendar_boundary_allocates_step_mass_without_duplication(self) -> None:
        step_totals = pd.DataFrame(
            {
                "sensitivity_id": ["run", "run"],
                "step": [0, 1],
                "step_start_date": pd.to_datetime(["2023-12-27", "2024-01-03"]),
                "step_end_date": pd.to_datetime(["2024-01-03", "2024-01-10"]),
                "range_overlap_start": pd.to_datetime(["2023-12-27", "2024-01-03"]),
                "range_overlap_end": pd.to_datetime(["2024-01-03", "2024-01-10"]),
                "range_overlap_days": [7, 7],
                "mass": [70.0, 140.0],
            }
        )
        annual = annualize_step_totals(step_totals)
        annual_mass_2023 = float(annual.loc[annual["year"] == 2023, "annual_mass"].iloc[0])
        annual_mass_2024 = float(annual.loc[annual["year"] == 2024, "annual_mass"].iloc[0])
        value_2023 = float(annual.loc[annual["year"] == 2023, "mean_step_mass"].iloc[0])
        value_2024 = float(annual.loc[annual["year"] == 2024, "mean_step_mass"].iloc[0])
        self.assertAlmostEqual(annual_mass_2023, 50.0)
        self.assertAlmostEqual(annual_mass_2024, 160.0)
        self.assertAlmostEqual(annual_mass_2023 + annual_mass_2024, 210.0)
        self.assertAlmostEqual(value_2023, 70.0)
        self.assertAlmostEqual(value_2024, (70.0 * 2 + 140.0 * 7) / 9)

    def test_six_day_terminal_step_is_normalized_by_its_own_duration(self) -> None:
        step_totals = pd.DataFrame(
            {
                "sensitivity_id": ["run", "run"],
                "step": [0, 1],
                "step_start_date": pd.to_datetime(["2100-12-19", "2100-12-26"]),
                "step_end_date": pd.to_datetime(["2100-12-26", "2101-01-01"]),
                "range_overlap_start": pd.to_datetime(["2100-12-19", "2100-12-26"]),
                "range_overlap_end": pd.to_datetime(["2100-12-26", "2101-01-01"]),
                "range_overlap_days": [7, 6],
                "mass": [70.0, 60.0],
            }
        )
        annual = annualize_step_totals(step_totals)
        self.assertAlmostEqual(float(annual["annual_mass"].iloc[0]), 130.0)
        self.assertAlmostEqual(float(annual["mean_step_mass"].iloc[0]), 70.0)

    def test_seasonal_series_normalizes_six_day_terminal_mass(self) -> None:
        step_totals = pd.DataFrame(
            {
                "sensitivity_id": ["run"],
                "step": [4069],
                "step_start_date": pd.to_datetime(["2100-12-26"]),
                "step_end_date": pd.to_datetime(["2101-01-01"]),
                "range_overlap_start": pd.to_datetime(["2100-12-26"]),
                "range_overlap_end": pd.to_datetime(["2101-01-01"]),
                "range_overlap_days": [6],
                "mass": [60.0],
            }
        )
        seasonal = seasonal_step_series(step_totals)
        self.assertAlmostEqual(float(seasonal["normalized_step_mass"].iloc[0]), 70.0)

    def test_annualization_excludes_days_outside_requested_window(self) -> None:
        step_totals = pd.DataFrame(
            {
                "sensitivity_id": ["run"],
                "step": [4017],
                "step_start_date": pd.to_datetime(["2099-12-28"]),
                "step_end_date": pd.to_datetime(["2100-01-04"]),
                "range_overlap_start": pd.to_datetime(["2100-01-01"]),
                "range_overlap_end": pd.to_datetime(["2100-01-04"]),
                "range_overlap_days": [3],
                "mass": [70.0],
            }
        )
        annual = annualize_step_totals(step_totals)
        self.assertEqual(annual["year"].tolist(), [2100])
        self.assertEqual(annual["coverage_days"].tolist(), [3])
        self.assertAlmostEqual(float(annual["mean_step_mass"].iloc[0]), 70.0)

    def test_baseline_loss_is_calculated_within_run(self) -> None:
        annual = pd.DataFrame(
            {
                "sensitivity_id": ["a", "a", "b", "b"],
                "year": [2023, 2100, 2023, 2100],
                "mean_step_mass": [100.0, 80.0, 200.0, 100.0],
            }
        )
        result = add_baseline_loss(annual)
        losses = result.loc[result["year"] == 2100, "loss_percent"].tolist()
        self.assertAlmostEqual(losses[0], 20.0)
        self.assertAlmostEqual(losses[1], 50.0)

    def test_2100_range_includes_cross_year_boundary_step(self) -> None:
        steps = steps_overlapping_date_range(date(2100, 1, 1), date(2100, 12, 31))
        self.assertEqual(steps[0], 4017)
        self.assertEqual(steps[-1], 4069)


if __name__ == "__main__":
    unittest.main()
