"""Plan or execute validated, relocatable paper ABM scenarios.

All 411 scenarios are required for the full manuscript analysis. Selected groups
are diagnostics and must use their own empty data root.
"""
from __future__ import annotations

from pathlib import Path as _DataPath
import sys as _data_sys
_ANALYSE_ROOT = next(p for p in _DataPath(__file__).resolve().parents if (p / 'run_analyse.py').is_file())
if str(_ANALYSE_ROOT / 'library') not in _data_sys.path:
    _data_sys.path.insert(0, str(_ANALYSE_ROOT / 'library'))
from steel_supply.pipeline_paths import data_path, science_path, bundle_root
import argparse
from pathlib import Path
import re
import sys

ROOT = bundle_root()
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from steel_supply.hazard_dynamic.scenario_grid import scenario_commands, command_arg
from steel_supply.hazard_dynamic.reproduction_grid import commands, run_grid


def command_label(run_index: int, command: list[str]) -> str:
    parts = [
        f"run{run_index:03d}",
        command_arg(command, "--hazard-type"),
        command_arg(command, "--climate-scenario"),
        command_arg(command, "--trade-scenario"),
        f"cap{command_arg(command, '--capacity-to-initial-flow')}",
        f"inv{command_arg(command, '--inventory-to-capacity-ratio')}",
    ]
    ratio = command_arg(command, "--random-disruption-ratio", default="")
    if ratio:
        parts.append(f"rand{ratio}")
    random_seed = command_arg(command, "--random-seed", default="")
    if random_seed:
        parts.append(f"seed{random_seed}")
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", "_".join(parts))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--experiment", default="all", choices=["all", "capacity", "climate-inventory", "climate-trade", "random-disruption"])
    parser.add_argument("--run", action="store_true", help="Execute; otherwise print the plan without writing files.")
    parser.add_argument("--resume", action="store_true", help="Revalidate and reuse completed attempts")
    args = parser.parse_args()
    if args.run:
        run_grid(args.data_root, args.experiment, resume=args.resume)
    else:
        for command in commands(args.data_root, args.experiment):
            print(" ".join(command))


if __name__ == "__main__":
    main()
