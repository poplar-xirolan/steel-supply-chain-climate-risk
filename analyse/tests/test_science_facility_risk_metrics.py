from __future__ import annotations

import importlib
import pandas as pd
import pytest


def test_facility_relative_risk_uses_production_weighted_global_denominator():
    module = importlib.import_module("steel_supply.postprocessing.facility_risk_metrics")
    # 10% at a 10-unit site and 30% at a 30-unit site => global 25%, not 20%.
    frame = pd.DataFrame({"ID_internal": [1, 2], "country_a3": ["AAA", "BBB"],
                          "production": [10., 30.], "loss": [1., 9.], "year": [2060, 2060]})
    result, denominator = module.relative_metrics(frame, [2060])
    assert denominator == pytest.approx(.25)
    assert result.set_index("ID_internal").facility_to_global_ratio.to_dict() == pytest.approx({1: .4, 2: 1.2})


def test_missing_year_is_not_silently_averaged():
    module = importlib.import_module("steel_supply.postprocessing.facility_risk_metrics")
    frame = pd.DataFrame({"ID_internal": [1], "country_a3": ["AAA"],
                          "production": [10.], "loss": [1.], "year": [2060]})
    with pytest.raises(ValueError, match="year"):
        module.relative_metrics(frame, [2059, 2060])


def test_zero_production_is_not_zero_risk():
    module = importlib.import_module("steel_supply.postprocessing.facility_risk_metrics")
    frame = pd.DataFrame({"ID_internal": [1, 2], "country_a3": ["AAA", "BBB"],
                          "production": [0., 10.], "loss": [0., 1.], "year": [2060, 2060]})
    result, _ = module.relative_metrics(frame, [2060])
    assert pd.isna(result.set_index("ID_internal").loc[1, "facility_to_global_ratio"])
