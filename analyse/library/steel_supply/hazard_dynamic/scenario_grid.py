"""Build the shared exact paper ABM sensitivity grids without executing them."""

from __future__ import annotations

from pathlib import Path as _DataPath
import sys as _data_sys
_ANALYSE_ROOT = next(p for p in _DataPath(__file__).resolve().parents if (p / 'run_analyse.py').is_file())
if str(_ANALYSE_ROOT / 'library') not in _data_sys.path:
    _data_sys.path.insert(0, str(_ANALYSE_ROOT / 'library'))
from steel_supply.pipeline_paths import data_path, science_path, bundle_root


import itertools
import sys
from pathlib import Path


ROOT = bundle_root()
ENGINE = bundle_root() / '06_dynamic_simulation/11_run_simulation.py'
DEFAULT_HAZARD_TYPE = "multi-hazard-max"
RANDOM_REPLICATE_SEEDS = (
    2026081100,
    2026081101,
    2026081102,
    2026081103,
    2026081104,
    2026081105,
    2026081106,
    2026081107,
    2026081108,
    2026081109,
)


def base_command() -> list[str]:
    return [
        sys.executable,
        "-u",
        str(ENGINE),
        "--start-year",
        "2023",
        "--end-year",
        "2100",
        "--fair-match-chunks",
        "100",
        "--mfa-path",
        str(data_path('results/mfa/2023.parquet')),
        "--hazard-data-dir",
        str(data_path('results/hazard_country_level_loss')),
        "--output-dir",
        str(data_path('results/simulation')),
    ]


def scenario_commands(
    experiment: str,
    hazard_type: str = DEFAULT_HAZARD_TYPE,
) -> list[list[str]]:
    base = base_command()
    result: list[list[str]] = []

    if experiment in {"all", "capacity"}:
        for capacity in [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]:
            result.append(
                base
                + [
                    "--hazard-type", hazard_type,
                    "--climate-scenario", "ssp370",
                    "--trade-scenario", "free-trade",
                    "--inventory-to-capacity-ratio", "4",
                    "--capacity-to-initial-flow", str(capacity),
                ]
            )

    if experiment in {"all", "climate-inventory"}:
        for climate, inventory, capacity in itertools.product(
            ["picontrol", "ssp126", "ssp370", "ssp585"],
            range(1, 9),
            [1.1, 1.2, 1.3, 1.4, 1.5],
        ):
            result.append(
                base
                + [
                    "--hazard-type", hazard_type,
                    "--climate-scenario", climate,
                    "--trade-scenario", "free-trade",
                    "--inventory-to-capacity-ratio", str(inventory),
                    "--capacity-to-initial-flow", str(capacity),
                ]
            )

    if experiment in {"all", "climate-trade"}:
        for climate, trade, capacity in itertools.product(
            ["picontrol", "ssp126", "ssp370", "ssp585"],
            [
                "free-trade",
                "us-cn-decoupling",
                "bloc-confrontation",
                "global-decoupling",
            ],
            [1.1, 1.2, 1.3, 1.4, 1.5],
        ):
            result.append(
                base
                + [
                    "--hazard-type", hazard_type,
                    "--climate-scenario", climate,
                    "--trade-scenario", trade,
                    "--inventory-to-capacity-ratio", "4",
                    "--capacity-to-initial-flow", str(capacity),
                ]
            )

    if experiment in {"all", "random-disruption"}:
        # Ten independent Monte Carlo realizations are intentional for every ratio.
        for ratio, random_seed in itertools.product(
            [x / 100 for x in range(5, 100, 5)],
            RANDOM_REPLICATE_SEEDS,
        ):
            result.append(
                base
                + [
                    "--hazard-type", hazard_type,
                    "--climate-scenario", "ssp370",
                    "--trade-scenario", "random-disruption",
                    "--random-disruption-ratio", f"{ratio:.2f}",
                    "--random-seed", str(random_seed),
                    "--inventory-to-capacity-ratio", "4",
                    "--capacity-to-initial-flow", "1.1",
                ]
            )

    if experiment == "all":
        result = deduplicate_deterministic_commands(result)
    return result


def deterministic_command_key(command: list[str]) -> tuple[str, ...]:
    """Identify one deterministic parameter combination independent of run ID."""

    return (
        command_arg(command, "--hazard-type"),
        command_arg(command, "--climate-scenario"),
        command_arg(command, "--trade-scenario"),
        command_arg(command, "--capacity-to-initial-flow"),
        command_arg(command, "--inventory-to-capacity-ratio"),
    )


def deduplicate_deterministic_commands(commands: list[list[str]]) -> list[list[str]]:
    """Keep the first deterministic command and every random replicate."""

    seen: set[tuple[str, ...]] = set()
    deduplicated: list[list[str]] = []
    for command in commands:
        if command_arg(command, "--trade-scenario") == "random-disruption":
            deduplicated.append(command)
            continue
        key = deterministic_command_key(command)
        if key in seen:
            continue
        seen.add(key)
        deduplicated.append(command)
    return deduplicated


def command_arg(command: list[str], flag: str, default: str = "na") -> str:
    try:
        return command[command.index(flag) + 1]
    except (ValueError, IndexError):
        return default
