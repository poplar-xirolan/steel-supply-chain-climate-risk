from __future__ import annotations

from dataclasses import FrozenInstanceError
from datetime import date, datetime, timedelta
import importlib.util
import math
from pathlib import Path
import random
import re
import sys

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "library/steel_supply/hazard_dynamic/simulation_runtime.py"


class DerivedDate(date):
    pass


class MissingRuntimeModule:
    def __getattr__(self, name):
        pytest.fail(f"runtime helper is missing: {MODULE_PATH}; requested {name}")


def load_runtime_module():
    if not MODULE_PATH.is_file():
        return MissingRuntimeModule()
    spec = importlib.util.spec_from_file_location("simulation_runtime", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def runtime():
    return load_runtime_module()


def test_2023_to_2100_schedule_has_six_day_terminal_step(runtime):
    steps = runtime.build_step_schedule(
        date(2023, 1, 1), date(2100, 12, 31), 7
    )

    assert len(steps) == 4070
    assert steps[0] == runtime.SimulationStep(
        index=0,
        start=date(2023, 1, 1),
        end_exclusive=date(2023, 1, 8),
        days=7,
        time_scale_ratio=7 / 365.25,
    )
    assert steps[-1] == runtime.SimulationStep(
        index=4069,
        start=date(2100, 12, 26),
        end_exclusive=date(2101, 1, 1),
        days=6,
        time_scale_ratio=6 / 365.25,
    )


def test_complete_schedule_uses_ceiling_and_exact_date_coverage(runtime):
    steps = runtime.build_step_schedule(date(2023, 1, 1), date(2023, 1, 10), 7)

    assert [(step.index, step.start, step.end_exclusive, step.days) for step in steps] == [
        (0, date(2023, 1, 1), date(2023, 1, 8), 7),
        (1, date(2023, 1, 8), date(2023, 1, 11), 3),
    ]
    covered = [
        current
        for step in steps
        for current in (
            step.start + timedelta(days=offset) for offset in range(step.days)
        )
    ]
    assert covered == [date(2023, 1, day) for day in range(1, 11)]
    assert len(covered) == len(set(covered))


def test_legacy_floor_schedule_drops_only_the_partial_terminal_step(runtime):
    steps = runtime.build_step_schedule(
        date(2023, 1, 1),
        date(2023, 1, 10),
        7,
        complete_terminal_step=False,
    )

    assert steps == [
        runtime.SimulationStep(
            index=0,
            start=date(2023, 1, 1),
            end_exclusive=date(2023, 1, 8),
            days=7,
            time_scale_ratio=7 / 365.25,
        )
    ]


@pytest.mark.parametrize("complete_terminal_step", [True, False])
def test_exact_multiple_has_no_extra_or_missing_step(runtime, complete_terminal_step):
    steps = runtime.build_step_schedule(
        date(2023, 1, 1),
        date(2023, 1, 14),
        7,
        complete_terminal_step=complete_terminal_step,
    )

    assert len(steps) == 2
    assert steps[-1].end_exclusive == date(2023, 1, 15)
    assert all(step.days == 7 for step in steps)


def test_complete_schedule_keeps_a_one_day_terminal_step(runtime):
    steps = runtime.build_step_schedule(date(2023, 1, 1), date(2023, 1, 8), 7)

    assert [(step.start, step.end_exclusive, step.days) for step in steps] == [
        (date(2023, 1, 1), date(2023, 1, 8), 7),
        (date(2023, 1, 8), date(2023, 1, 9), 1),
    ]
    assert steps[-1].time_scale_ratio == pytest.approx(1 / 365.25)


def test_floor_schedule_can_be_empty_when_interval_is_shorter_than_step(runtime):
    assert (
        runtime.build_step_schedule(
            date(2023, 1, 1),
            date(2023, 1, 1),
            7,
            complete_terminal_step=False,
        )
        == []
    )


def test_huge_step_is_capped_before_date_arithmetic(runtime):
    complete = runtime.build_step_schedule(
        date(2023, 1, 1),
        date(2023, 1, 1),
        3_000_000,
        complete_terminal_step=True,
    )
    legacy_floor = runtime.build_step_schedule(
        date(2023, 1, 1),
        date(2023, 1, 1),
        3_000_000,
        complete_terminal_step=False,
    )

    assert complete == [
        runtime.SimulationStep(
            index=0,
            start=date(2023, 1, 1),
            end_exclusive=date(2023, 1, 2),
            days=1,
            time_scale_ratio=1 / 365.25,
        )
    ]
    assert legacy_floor == []


def test_simulation_steps_are_immutable(runtime):
    step = runtime.build_step_schedule(date(2023, 1, 1), date(2023, 1, 1), 7)[0]

    with pytest.raises(FrozenInstanceError):
        step.days = 2


@pytest.mark.parametrize("index", [True, -1, 1.0, "0"])
def test_direct_step_construction_rejects_invalid_index(runtime, index):
    expected_error = ValueError if index == -1 else TypeError

    with pytest.raises(expected_error, match="index"):
        runtime.SimulationStep(
            index=index,
            start=date(2023, 1, 1),
            end_exclusive=date(2023, 1, 8),
            days=7,
            time_scale_ratio=7 / 365.25,
        )


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("start", "2023-01-01"),
        ("start", datetime(2023, 1, 1)),
        ("start", DerivedDate(2023, 1, 1)),
        ("end_exclusive", "2023-01-08"),
        ("end_exclusive", datetime(2023, 1, 8)),
        ("end_exclusive", DerivedDate(2023, 1, 8)),
    ],
)
def test_direct_step_construction_rejects_non_plain_dates(
    runtime, field, invalid_value
):
    values = {
        "index": 0,
        "start": date(2023, 1, 1),
        "end_exclusive": date(2023, 1, 8),
        "days": 7,
        "time_scale_ratio": 7 / 365.25,
    }
    values[field] = invalid_value

    with pytest.raises(TypeError, match=field):
        runtime.SimulationStep(**values)


@pytest.mark.parametrize("end_exclusive", [date(2023, 1, 1), date(2022, 12, 31)])
def test_direct_step_construction_requires_positive_date_interval(
    runtime, end_exclusive
):
    with pytest.raises(ValueError, match="end_exclusive.*start"):
        runtime.SimulationStep(
            index=0,
            start=date(2023, 1, 1),
            end_exclusive=end_exclusive,
            days=1,
            time_scale_ratio=1 / 365.25,
        )


@pytest.mark.parametrize("days", [True, 0, -1, 7.0, "7"])
def test_direct_step_construction_rejects_invalid_days(runtime, days):
    expected_error = ValueError if days in (0, -1) else TypeError

    with pytest.raises(expected_error, match="days"):
        runtime.SimulationStep(
            index=0,
            start=date(2023, 1, 1),
            end_exclusive=date(2023, 1, 8),
            days=days,
            time_scale_ratio=7 / 365.25,
        )


def test_direct_step_construction_requires_days_to_match_interval(runtime):
    with pytest.raises(ValueError, match="days.*date interval"):
        runtime.SimulationStep(
            index=0,
            start=date(2023, 1, 1),
            end_exclusive=date(2023, 1, 8),
            days=6,
            time_scale_ratio=6 / 365.25,
        )


@pytest.mark.parametrize(
    "time_scale_ratio",
    [True, "ratio", None, float("nan"), float("inf"), -float("inf")],
)
def test_direct_step_construction_rejects_invalid_time_scale_ratio(
    runtime, time_scale_ratio
):
    if isinstance(time_scale_ratio, (bool, str)) or time_scale_ratio is None:
        expected_error = TypeError
    else:
        expected_error = ValueError

    with pytest.raises(expected_error, match="time_scale_ratio"):
        runtime.SimulationStep(
            index=0,
            start=date(2023, 1, 1),
            end_exclusive=date(2023, 1, 8),
            days=7,
            time_scale_ratio=time_scale_ratio,
        )


@pytest.mark.parametrize(
    "wrong_ratio",
    [1.0, math.nextafter(7 / 365.25, math.inf)],
)
def test_direct_step_construction_requires_exact_annual_scale(runtime, wrong_ratio):
    with pytest.raises(ValueError, match="time_scale_ratio.*days / 365.25"):
        runtime.SimulationStep(
            index=0,
            start=date(2023, 1, 1),
            end_exclusive=date(2023, 1, 8),
            days=7,
            time_scale_ratio=wrong_ratio,
        )


def test_schedule_rejects_reversed_date_range(runtime):
    with pytest.raises(ValueError, match="start.*end_inclusive"):
        runtime.build_step_schedule(date(2023, 1, 2), date(2023, 1, 1), 7)


@pytest.mark.parametrize(
    ("start", "end_inclusive"),
    [
        ("2023-01-01", date(2023, 1, 2)),
        (date(2023, 1, 1), "2023-01-02"),
        (datetime(2023, 1, 1), date(2023, 1, 2)),
        (date(2023, 1, 1), datetime(2023, 1, 2)),
    ],
)
def test_schedule_rejects_non_date_boundaries(runtime, start, end_inclusive):
    with pytest.raises(TypeError, match="must be a date"):
        runtime.build_step_schedule(start, end_inclusive, 7)


@pytest.mark.parametrize("step_days", [0, -1])
def test_schedule_rejects_nonpositive_step_days(runtime, step_days):
    with pytest.raises(ValueError, match="step_days.*positive"):
        runtime.build_step_schedule(date(2023, 1, 1), date(2023, 1, 2), step_days)


@pytest.mark.parametrize("step_days", [True, 7.0, "7", None])
def test_schedule_rejects_noninteger_step_days(runtime, step_days):
    with pytest.raises(TypeError, match="step_days.*integer"):
        runtime.build_step_schedule(date(2023, 1, 1), date(2023, 1, 2), step_days)


def test_schedule_rejects_nonboolean_terminal_policy(runtime):
    with pytest.raises(TypeError, match="complete_terminal_step.*boolean"):
        runtime.build_step_schedule(
            date(2023, 1, 1),
            date(2023, 1, 2),
            7,
            complete_terminal_step=1,
        )


def test_ticket_ids_are_unique_repeatable_and_zero_padded(runtime):
    first = runtime.DeterministicIdFactory()
    second = runtime.DeterministicIdFactory()
    for factory in (first, second):
        factory.start_step(12)

    first_ids = [first.next_id("demand"), first.next_id("supply")]
    second_ids = [second.next_id("demand"), second.next_id("supply")]
    assert first_ids == second_ids == [
        "000012-demand-00000000",
        "000012-supply-00000001",
    ]
    assert len(first_ids) == len(set(first_ids))


def test_ticket_sequence_is_shared_across_demand_and_supply(runtime):
    factory = runtime.DeterministicIdFactory()
    factory.start_step(3)

    ids = [
        factory.next_id("supply"),
        factory.next_id("demand"),
        factory.next_id("supply"),
    ]

    assert [int(ticket_id.rsplit("-", 1)[1]) for ticket_id in ids] == [0, 1, 2]


def test_start_step_resets_sequence_and_collision_set_for_next_step(runtime):
    factory = runtime.DeterministicIdFactory()
    factory.start_step(8)
    assert factory.next_id("demand") == "000008-demand-00000000"
    factory.start_step(9)

    assert factory.next_id("demand") == "000009-demand-00000000"


def test_ticket_factory_uses_one_collision_set_across_demand_and_supply(runtime):
    factory = runtime.DeterministicIdFactory()
    factory.start_step(4)
    assert factory.next_id("demand") == "000004-demand-00000000"
    factory._seen_ids.add("000004-supply-00000001")

    with pytest.raises(RuntimeError, match="duplicate deterministic ticket ID"):
        factory.next_id("supply")


def test_ticket_factory_requires_step_before_id(runtime):
    with pytest.raises(RuntimeError, match="start_step"):
        runtime.DeterministicIdFactory().next_id("demand")


@pytest.mark.parametrize("ticket_type", ["Demand", "other", "", None, 1])
def test_ticket_factory_rejects_invalid_ticket_type(runtime, ticket_type):
    factory = runtime.DeterministicIdFactory()
    factory.start_step(0)

    with pytest.raises((TypeError, ValueError), match="ticket_type"):
        factory.next_id(ticket_type)


@pytest.mark.parametrize("step", [True, 1.0, "1", None])
def test_ticket_factory_rejects_noninteger_step(runtime, step):
    with pytest.raises(TypeError, match="step.*integer"):
        runtime.DeterministicIdFactory().start_step(step)


def test_ticket_factory_rejects_negative_or_nonincreasing_steps(runtime):
    factory = runtime.DeterministicIdFactory()
    with pytest.raises(ValueError, match="step.*non-negative"):
        factory.start_step(-1)

    factory.start_step(2)
    for invalid_step in (2, 1):
        with pytest.raises(ValueError, match="strictly increase"):
            factory.start_step(invalid_step)


def test_random_blacklists_are_stable_under_input_order_and_duplicates(runtime):
    first = runtime.build_random_blacklists(
        ["USA", "CHN", "AUS", "IND", "BRA", "CAN"],
        0.5,
        random.Random(2026081100),
    )
    second = runtime.build_random_blacklists(
        ["CAN", "BRA", "IND", "AUS", "CHN", "USA", "USA"],
        0.5,
        random.Random(2026081100),
    )

    assert first == second
    assert list(first) == ["AUS", "BRA", "CAN", "CHN", "IND", "USA"]
    assert all(blocked == sorted(blocked) for blocked in first.values())
    assert all(len(blocked) == len(set(blocked)) == 3 for blocked in first.values())
    assert all(country not in blocked for country, blocked in first.items())


@pytest.mark.parametrize(
    ("ratio", "expected_count"),
    [(0, 0), (0.2, 1), (0.99, 4), (1, 4)],
)
def test_random_blacklist_count_uses_truncation_and_nonself_cap(
    runtime, ratio, expected_count
):
    result = runtime.build_random_blacklists(
        ["A", "B", "C", "D", "E"], ratio, random.Random(9)
    )

    assert all(len(blocked) == expected_count for blocked in result.values())


def test_random_blacklists_are_repeatable_for_same_seed(runtime):
    countries = ["A", "B", "C", "D", "E", "F", "G", "H"]

    assert runtime.build_random_blacklists(
        countries, 0.5, random.Random(42)
    ) == runtime.build_random_blacklists(countries, 0.5, random.Random(42))


def test_random_blacklists_change_for_a_different_seed(runtime):
    countries = ["A", "B", "C", "D", "E", "F", "G", "H"]

    first = runtime.build_random_blacklists(countries, 0.5, random.Random(42))
    second = runtime.build_random_blacklists(countries, 0.5, random.Random(43))

    assert first != second
    assert {country: len(blocked) for country, blocked in first.items()} == {
        country: len(blocked) for country, blocked in second.items()
    }


def test_random_blacklists_advance_the_caller_owned_rng(runtime):
    rng = random.Random(12)
    before = rng.getstate()

    runtime.build_random_blacklists(["A", "B", "C", "D"], 0.5, rng)

    assert rng.getstate() != before


def test_random_blacklists_handle_empty_and_singleton_country_sets(runtime):
    assert runtime.build_random_blacklists([], 1.0, random.Random(1)) == {}
    assert runtime.build_random_blacklists(["A"], 1.0, random.Random(1)) == {"A": []}


@pytest.mark.parametrize("ratio", [-0.01, 1.01, float("nan"), float("inf")])
def test_random_blacklists_reject_out_of_range_or_nonfinite_ratio(runtime, ratio):
    with pytest.raises(ValueError, match="ratio"):
        runtime.build_random_blacklists(["A", "B"], ratio, random.Random(1))


@pytest.mark.parametrize("ratio", [True, "0.5", None])
def test_random_blacklists_reject_nonreal_ratio(runtime, ratio):
    with pytest.raises(TypeError, match="ratio.*real number"):
        runtime.build_random_blacklists(["A", "B"], ratio, random.Random(1))


@pytest.mark.parametrize(
    "countries",
    ["ABC", {"A", "B"}, ["A", 1], ["A", ""], ["A", " A"]],
)
def test_random_blacklists_reject_invalid_country_sequences(runtime, countries):
    with pytest.raises((TypeError, ValueError), match="countries|country"):
        runtime.build_random_blacklists(countries, 0.5, random.Random(1))


@pytest.mark.parametrize("rng", [None, random, random.SystemRandom()])
def test_random_blacklists_require_deterministic_random_instance(runtime, rng):
    with pytest.raises(TypeError, match=re.escape("random.Random")):
        runtime.build_random_blacklists(["A", "B"], 0.5, rng)
