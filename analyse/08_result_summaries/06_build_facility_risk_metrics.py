"""Build the current Fig. 3 facility/global ratios, without cached audit tables."""

from pathlib import Path as _DataPath
import sys as _data_sys
_ANALYSE_ROOT = next(p for p in _DataPath(__file__).resolve().parents if (p / 'run_analyse.py').is_file())
if str(_ANALYSE_ROOT / 'library') not in _data_sys.path:
    _data_sys.path.insert(0, str(_ANALYSE_ROOT / 'library'))
from steel_supply.pipeline_paths import data_path, science_path, bundle_root
from pathlib import Path
import sys

ROOT = bundle_root()
sys.path.insert(0, str(ROOT))
from steel_supply.pipeline_paths import data_path
from steel_supply.postprocessing.facility_risk_metrics import build


def main():
    result = build(data_path())
    output = data_path("results/postprocessing/facility_risk")
    output.mkdir(parents=True, exist_ok=True)
    result.to_csv(output / "facility_relative_impacts.csv", index=False)
    print(f"Wrote {len(result)} facility/hazard records to {output}")


if __name__ == "__main__":
    main()
