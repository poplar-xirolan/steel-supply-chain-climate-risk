from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "05_country_constraints/10_aggregate_country_loss.py"
sys.path.insert(0, str(SCRIPT_PATH.parent))


def _load_module():
    spec = importlib.util.spec_from_file_location("cyclone_aggregation", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


cyclone_aggregation = _load_module()


def aggregate_synthetic_cyclone(
    *, year: int, annual_production: float
) -> pd.DataFrame:
    production = pd.DataFrame(
        {
            "ID_internal": ["facility-a"],
            "country_a3": ["AAA"],
            "fe_eq_weight_kg_scaled": [annual_production],
        }
    )
    events = pd.DataFrame(
        {
            "id_internal": ["facility-a"],
            "day_of_year": [60],
            "production_loss_fraction": [0.125],
        }
    )
    facility, _country = cyclone_aggregation.aggregate_facility_losses(
        production,
        events,
        {"year_int": year},
        hazard_id_column="id_internal",
        hazard_day_column="day_of_year",
        hazard_loss_column="production_loss_fraction",
    )
    return facility


def test_aggregation_module_import_has_no_pipeline_side_effects(tmp_path):
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import importlib.util, sys; "
                f"sys.path.insert(0, {str(SCRIPT_PATH.parent)!r}); "
                f"spec = importlib.util.spec_from_file_location('cyclone_aggregation', {str(SCRIPT_PATH)!r}); "
                "module = importlib.util.module_from_spec(spec); "
                "spec.loader.exec_module(module)"
            ),
            "--facility-output-dir",
            str(tmp_path / "must-not-be-created"),
        ],
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert not (tmp_path / "must-not-be-created").exists()


@pytest.mark.parametrize(("year", "expected_days"), [(2023, 365), (2024, 366)])
def test_cyclone_uses_full_calendar_denominator(year, expected_days):
    result = aggregate_synthetic_cyclone(
        year=year, annual_production=expected_days * 100.0
    )

    assert result["day_of_year"].nunique() == expected_days
    assert result["daily_fe_eq_weight_kg_scaled"].unique().tolist() == [100.0]


@pytest.mark.parametrize(
    ("year", "event_days"),
    [(2023, [59, 365]), (2024, [60, 366])],
)
def test_cyclone_preserves_february_and_december_event_days(year, event_days):
    production = pd.DataFrame(
        {
            "ID_internal": ["facility-a"],
            "country_a3": ["AAA"],
            "fe_eq_weight_kg_scaled": [36_600.0],
        }
    )
    source_fractions = np.array(
        [np.nextafter(0.125, 1.0), np.nextafter(0.875, 0.0)], dtype=np.float64
    )
    events = pd.DataFrame(
        {
            "id_internal": ["facility-a", "facility-a"],
            "day_of_year": event_days,
            "production_loss_fraction": source_fractions,
        }
    )

    result, _country = cyclone_aggregation.aggregate_facility_losses(
        production,
        events,
        {"year_int": year},
        hazard_id_column="id_internal",
        hazard_day_column="day_of_year",
        hazard_loss_column="production_loss_fraction",
    )
    observed = result.loc[
        result["day_of_year"].isin(event_days), "production_loss_fraction"
    ].to_numpy()

    assert observed.dtype == source_fractions.dtype
    assert np.array_equal(observed.view(np.uint64), source_fractions.view(np.uint64))


def test_cyclone_fills_only_absent_event_day_with_physical_zero():
    result = aggregate_synthetic_cyclone(year=2024, annual_production=36_600.0)

    assert result.loc[result["day_of_year"].eq(59), "production_loss_fraction"].item() == 0.0
    assert result.loc[result["day_of_year"].eq(60), "production_loss_fraction"].item() == 0.125
    assert result.loc[result["day_of_year"].eq(366), "production_loss_fraction"].item() == 0.0


def test_cyclone_rejects_source_event_nan_instead_of_zero_filling():
    production = pd.DataFrame(
        {
            "ID_internal": ["facility-a"],
            "country_a3": ["AAA"],
            "fe_eq_weight_kg_scaled": [36_500.0],
        }
    )
    events = pd.DataFrame(
        {
            "id_internal": ["facility-a"],
            "day_of_year": [1],
            "production_loss_fraction": [np.nan],
        }
    )

    with pytest.raises(ValueError, match="missing.*production_loss_fraction"):
        cyclone_aggregation.aggregate_facility_losses(
            production,
            events,
            {"year_int": 2023},
            hazard_id_column="id_internal",
            hazard_day_column="day_of_year",
            hazard_loss_column="production_loss_fraction",
        )


def test_cyclone_rejects_duplicate_source_event_key():
    production = pd.DataFrame(
        {
            "ID_internal": ["facility-a"],
            "country_a3": ["AAA"],
            "fe_eq_weight_kg_scaled": [36_500.0],
        }
    )
    events = pd.DataFrame(
        {
            "id_internal": ["facility-a", "facility-a"],
            "day_of_year": [1, 1],
            "production_loss_fraction": [0.25, 0.5],
        }
    )

    with pytest.raises(ValueError, match="duplicate.*event"):
        cyclone_aggregation.aggregate_facility_losses(
            production,
            events,
            {"year_int": 2023},
            hazard_id_column="id_internal",
            hazard_day_column="day_of_year",
            hazard_loss_column="production_loss_fraction",
        )


@pytest.mark.parametrize(("year", "invalid_day"), [(2023, 0), (2023, 366), (2024, 367)])
def test_cyclone_rejects_out_of_range_source_event_day(year, invalid_day):
    production = pd.DataFrame(
        {
            "ID_internal": ["facility-a"],
            "country_a3": ["AAA"],
            "fe_eq_weight_kg_scaled": [36_500.0],
        }
    )
    events = pd.DataFrame(
        {
            "id_internal": ["facility-a"],
            "day_of_year": [invalid_day],
            "production_loss_fraction": [0.25],
        }
    )

    with pytest.raises(ValueError, match="outside.*year"):
        cyclone_aggregation.aggregate_facility_losses(
            production,
            events,
            {"year_int": year},
            hazard_id_column="id_internal",
            hazard_day_column="day_of_year",
            hazard_loss_column="production_loss_fraction",
        )


def test_cyclone_absolute_loss_uses_calendar_scaled_daily_production():
    non_leap = aggregate_synthetic_cyclone(year=2023, annual_production=36_600.0)
    leap = aggregate_synthetic_cyclone(year=2024, annual_production=36_600.0)

    non_leap_event = non_leap.loc[non_leap["day_of_year"].eq(60)].iloc[0]
    leap_event = leap.loc[leap["day_of_year"].eq(60)].iloc[0]
    assert non_leap_event["daily_fe_eq_weight_kg_scaled"] == pytest.approx(
        36_600.0 / 365
    )
    assert leap_event["daily_fe_eq_weight_kg_scaled"] == 100.0
    assert non_leap_event["production_loss_kg_scaled"] == pytest.approx(
        (36_600.0 / 365) * 0.125
    )
    assert leap_event["production_loss_kg_scaled"] == 12.5


def test_non_cyclone_dispatch_preserves_existing_event_span_behavior():
    production = pd.DataFrame(
        {
            "ID_internal": ["facility-a"],
            "country_a3": ["AAA"],
            "fe_eq_weight_kg_scaled": [200.0],
        }
    )
    events = pd.DataFrame(
        {
            "id_internal": ["facility-a", "facility-a"],
            "day_of_year": [100, 101],
            "gross_production_loss_fraction": [0.25, 0.5],
        }
    )

    facility, _country = cyclone_aggregation.aggregate_hazard_losses(
        production,
        events,
        {"year_int": 2023},
        hazard_name="heatwave",
        hazard_id_column="id_internal",
        hazard_day_column="day_of_year",
        hazard_loss_column="gross_production_loss_fraction",
    )

    assert facility["day_of_year"].tolist() == [100, 101]
    assert facility["daily_fe_eq_weight_kg_scaled"].tolist() == [100.0, 100.0]
