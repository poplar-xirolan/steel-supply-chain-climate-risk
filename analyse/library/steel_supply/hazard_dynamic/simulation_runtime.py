"""Shared deterministic runtime helpers for the steel supply-chain ABM.

This module contains only scheduling, solver-ticket identity, and seeded random
blacklist construction. Integration with the simulation engine is deliberately
kept in the caller so these contracts can be tested without running the ABM.
"""

from __future__ import annotations
from steel_supply.pipeline_paths import science_path, bundle_root, data_path

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta
import math
from numbers import Integral, Real
import random
from typing import Literal


TicketType = Literal["demand", "supply"]

__all__ = [
    "DeterministicIdFactory",
    "SimulationStep",
    "build_random_blacklists",
    "build_step_schedule",
]


@dataclass(frozen=True, slots=True)
class SimulationStep:
    """One immutable, end-exclusive simulation interval.

    ``time_scale_ratio`` is the interval's share of the 365.25-day annual
    quantities used by the existing engine, not its share of a nominal step.
    """

    index: int
    start: date
    end_exclusive: date
    days: int
    time_scale_ratio: float

    def __post_init__(self) -> None:
        """Reject internally inconsistent steps, including direct construction."""

        normalized_index = _validate_integer(self.index, "index")
        if normalized_index < 0:
            raise ValueError("index must be non-negative")
        _validate_plain_date(self.start, "start")
        _validate_plain_date(self.end_exclusive, "end_exclusive")
        if self.end_exclusive <= self.start:
            raise ValueError("end_exclusive must be later than start")

        normalized_days = _validate_integer(self.days, "days")
        if normalized_days <= 0:
            raise ValueError("days must be positive")
        interval_days = (self.end_exclusive - self.start).days
        if normalized_days != interval_days:
            raise ValueError(
                "days must exactly match the date interval; "
                f"got days={normalized_days}, interval={interval_days}"
            )

        if isinstance(self.time_scale_ratio, bool) or not isinstance(
            self.time_scale_ratio, Real
        ):
            raise TypeError("time_scale_ratio must be a real number")
        normalized_ratio = float(self.time_scale_ratio)
        if not math.isfinite(normalized_ratio):
            raise ValueError("time_scale_ratio must be finite")
        expected_ratio = normalized_days / 365.25
        # The builder and public direct-construction contract use this same
        # division. Exact equality catches stale nominal-step ratios rather
        # than silently tolerating a scientifically different duration.
        if normalized_ratio != expected_ratio:
            raise ValueError(
                "time_scale_ratio must exactly equal days / 365.25; "
                f"got {normalized_ratio!r}, expected {expected_ratio!r}"
            )


def build_step_schedule(
    start: date,
    end_inclusive: date,
    step_days: int,
    complete_terminal_step: bool = True,
) -> list[SimulationStep]:
    """Build consecutive simulation steps over an inclusive date interval.

    With ``complete_terminal_step=True`` (the corrected default), a shortened
    final step covers any remainder. ``False`` preserves the historical floor
    behavior by returning complete nominal steps only.
    """

    _validate_plain_date(start, "start")
    _validate_plain_date(end_inclusive, "end_inclusive")
    normalized_step_days = _validate_integer(step_days, "step_days")
    if normalized_step_days <= 0:
        raise ValueError("step_days must be positive")
    if not isinstance(complete_terminal_step, bool):
        raise TypeError("complete_terminal_step must be a boolean")
    if start > end_inclusive:
        raise ValueError("start must not be later than end_inclusive")

    try:
        requested_end_exclusive = end_inclusive + timedelta(days=1)
    except OverflowError as exc:
        raise ValueError("end_inclusive is too late to represent an exclusive end") from exc

    total_days = (requested_end_exclusive - start).days
    complete_steps, remainder = divmod(total_days, normalized_step_days)
    step_count = complete_steps + int(complete_terminal_step and remainder > 0)

    schedule: list[SimulationStep] = []
    for index in range(step_count):
        step_start = start + timedelta(days=index * normalized_step_days)
        remaining_days = (requested_end_exclusive - step_start).days
        current_days = min(normalized_step_days, remaining_days)
        step_end = step_start + timedelta(days=current_days)
        schedule.append(
            SimulationStep(
                index=index,
                start=step_start,
                end_exclusive=step_end,
                days=current_days,
                time_scale_ratio=current_days / 365.25,
            )
        )
    return schedule


class DeterministicIdFactory:
    """Generate unique, repeatable solver-ticket IDs within ordered steps."""

    def __init__(self) -> None:
        self._current_step: int | None = None
        self._last_started_step: int | None = None
        self._next_sequence = 0
        self._seen_ids: set[str] = set()

    def start_step(self, step: int) -> None:
        """Start a strictly later step and reset its sequence and collision set."""

        normalized_step = _validate_integer(step, "step")
        if normalized_step < 0:
            raise ValueError("step must be non-negative")
        if (
            self._last_started_step is not None
            and normalized_step <= self._last_started_step
        ):
            raise ValueError(
                "step values must strictly increase within one ID factory; "
                f"last={self._last_started_step}, requested={normalized_step}"
            )

        self._current_step = normalized_step
        self._last_started_step = normalized_step
        self._next_sequence = 0
        self._seen_ids = set()

    def next_id(self, ticket_type: TicketType) -> str:
        """Return the next zero-padded ID, failing closed on any collision."""

        if self._current_step is None:
            raise RuntimeError("start_step must be called before next_id")
        if not isinstance(ticket_type, str):
            raise TypeError("ticket_type must be 'demand' or 'supply'")
        if ticket_type not in ("demand", "supply"):
            raise ValueError("ticket_type must be 'demand' or 'supply'")

        ticket_id = (
            f"{self._current_step:06d}-{ticket_type}-{self._next_sequence:08d}"
        )
        if ticket_id in self._seen_ids:
            raise RuntimeError(f"duplicate deterministic ticket ID: {ticket_id}")
        self._seen_ids.add(ticket_id)
        self._next_sequence += 1
        return ticket_id


def build_random_blacklists(
    countries: Sequence[str],
    ratio: float,
    rng: random.Random,
) -> dict[str, list[str]]:
    """Build deterministic unilateral trade blacklists using ``rng``.

    Country order and duplicates in the caller's sequence cannot affect the
    result. The accepted restriction count is ``int(n * ratio)``, capped by
    the ``n - 1`` non-self partners. The caller owns and seeds ``rng``.
    """

    if not isinstance(rng, random.Random) or isinstance(rng, random.SystemRandom):
        raise TypeError("rng must be a deterministic random.Random instance")
    if isinstance(countries, (str, bytes, bytearray)) or not isinstance(
        countries, Sequence
    ):
        raise TypeError("countries must be a sequence of country strings")

    normalized_countries: list[str] = []
    for country in countries:
        if not isinstance(country, str):
            raise TypeError("every country in countries must be a string")
        if not country or country != country.strip():
            raise ValueError("country strings must be non-empty and unpadded")
        normalized_countries.append(country)
    unique_countries = sorted(set(normalized_countries))

    if isinstance(ratio, bool) or not isinstance(ratio, Real):
        raise TypeError("ratio must be a real number")
    normalized_ratio = float(ratio)
    if not math.isfinite(normalized_ratio) or not 0.0 <= normalized_ratio <= 1.0:
        raise ValueError("ratio must be finite and within [0, 1]")

    available_partners = max(0, len(unique_countries) - 1)
    restriction_count = min(
        available_partners,
        int(len(unique_countries) * normalized_ratio),
    )

    blacklists: dict[str, list[str]] = {}
    for country in unique_countries:
        candidates = [candidate for candidate in unique_countries if candidate != country]
        blacklists[country] = sorted(rng.sample(candidates, restriction_count))
    return blacklists


def _validate_plain_date(value: date, label: str) -> None:
    if type(value) is not date:
        raise TypeError(
            f"{label} must be a date (plain datetime.date; not datetime or subclass)"
        )


def _validate_integer(value: int, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{label} must be an integer")
    return int(value)

