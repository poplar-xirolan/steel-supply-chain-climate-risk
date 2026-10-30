"""Build portable plotting inputs from canonical model outputs, without rerunning ABM."""

from pathlib import Path as _DataPath
import sys as _data_sys
_ANALYSE_ROOT = next(p for p in _DataPath(__file__).resolve().parents if (p / 'run_analyse.py').is_file())
if str(_ANALYSE_ROOT / 'library') not in _data_sys.path:
    _data_sys.path.insert(0, str(_ANALYSE_ROOT / 'library'))
from steel_supply.pipeline_paths import data_path, science_path, bundle_root
from pathlib import Path
import sys
ROOT = bundle_root()
sys.path.insert(0,str(ROOT))
from steel_supply.pipeline_paths import data_path
from steel_supply.postprocessing.compact_figure_metrics import build

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root',type=Path,default=data_path())
    parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args()
    print(build(args.source_root,args.output_dir))
