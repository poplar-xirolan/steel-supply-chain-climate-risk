"""Focused checks for pre-scaling facility coverage and correspondence."""

from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'library'))

from steel_supply.postprocessing.facility_coverage import StageSpec, summarize_stage


def fixture_data():
    mfa = pd.DataFrame(
        {
            "entity_from": ["AAA", "BBB", "CCC", "DDD"],
            "entity_to": ["AAA", "BBB", "CCC", "DDD"],
            "sector_from": ["P1", "P1", "P1", "P1"],
            "sector_to": ["M1", "M1", "M1", "M1"],
            "mass": [60.0, 30.0, 10.0, 5.0],
            "unit": ["kg", "kg", "kg", "kg"],
        }
    )
    facilities = pd.DataFrame(
        {
            "ID_internal": ["f1", "f2", "f3"],
            "country_a3": ["AAA", "BBB", "CCC"],
            "unscaled": [50.0, 20.0, 8.0],
            "scaled": [60.0, 30.0, 10.0],
        }
    )
    eligible = pd.DataFrame({"ID_internal": ["f1", "f2", "f3", "f4"]})
    spec = StageSpec(
        "Source",
        "Mining",
        "Production",
        "P1",
        "M1",
        Path("unused"),
        "unscaled",
        "scaled",
    )
    return mfa, facilities, eligible, spec


def test_summary_uses_unscaled_source_values_and_mfa_coverage():
    mfa, facilities, eligible, spec = fixture_data()
    summary, by_country = summarize_stage(
        mfa=mfa,
        facilities=facilities,
        eligible_source_ids=eligible,
        spec=spec,
    )
    assert summary["facilities_retained"] == 3
    assert summary["source_facilities_eligible"] == 4
    assert summary["economies_represented"] == 3
    assert summary["producing_economies"] == 4
    assert summary["production_coverage_pct"] == pytest.approx(100.0 / 105.0 * 100)
    assert summary["unscaled_source_total_fe_eq_kg"] == pytest.approx(78.0)
    assert summary["source_to_represented_mfa_ratio"] == pytest.approx(78.0 / 100.0)
    assert summary["mfa_to_source_scaling_median"] == pytest.approx(1.25)
    assert by_country.loc[by_country.country.eq("DDD"), "represented"].item() is False
    assert pd.isna(by_country.loc[by_country.country.eq("DDD"), "source_to_mfa_ratio"].item())


def test_scaled_facility_mass_must_reconcile_by_country():
    mfa, facilities, eligible, spec = fixture_data()
    facilities.loc[0, "scaled"] = 59.0
    with pytest.raises(ValueError, match="does not reconcile by country"):
        summarize_stage(
            mfa=mfa,
            facilities=facilities,
            eligible_source_ids=eligible,
            spec=spec,
        )


def test_ineligible_or_duplicate_facilities_fail_closed():
    mfa, facilities, eligible, spec = fixture_data()
    facilities.loc[0, "ID_internal"] = "unknown"
    with pytest.raises(ValueError, match="ineligible"):
        summarize_stage(
            mfa=mfa,
            facilities=facilities,
            eligible_source_ids=eligible,
            spec=spec,
        )
