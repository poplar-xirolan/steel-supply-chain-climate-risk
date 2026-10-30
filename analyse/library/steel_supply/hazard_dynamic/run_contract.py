"""Portable run identities and atomic output publication; no historical paths."""
from __future__ import annotations
from steel_supply.pipeline_paths import science_path, bundle_root, data_path
from dataclasses import dataclass
import hashlib
import json
import os
import subprocess
from pathlib import Path
import tempfile
import time
from typing import Mapping, Sequence

_ATOMIC_REPLACE_RETRY_SECONDS = 1.0
_ATOMIC_REPLACE_RETRY_DELAY_SECONDS = 0.05
_RETRYABLE_WINDOWS_REPLACE_ERRORS = {5, 32}

def hidden_subprocess_kwargs() -> dict[str, object]:
    """Return the uniform no-console Windows child-process policy."""
    if os.name != "nt":
        return {}
    startupinfo = subprocess.STARTUPINFO()
    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    startupinfo.wShowWindow = subprocess.SW_HIDE
    return {
        "creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0),
        "startupinfo": startupinfo,
    }


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def validate_argv(argv: Sequence[str]) -> tuple[str, ...]:
    if not argv:
        raise ValueError("scenario argv must not be empty")
    if any(not isinstance(value, str) or not value for value in argv):
        raise ValueError("scenario argv must contain only non-empty strings")
    return tuple(argv)


def canonical_argv_sha256(argv: Sequence[str]) -> str:
    return hashlib.sha256(canonical_json_bytes(list(validate_argv(argv)))).hexdigest()


def scenario_key(ordinal: int, argv: Sequence[str]) -> str:
    if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal < 1:
        raise ValueError("scenario ordinal must be a positive integer")
    return f"scenario_{ordinal:03d}_{canonical_argv_sha256(argv)[:12]}"


@dataclass(frozen=True)
class ScenarioSpec:
    ordinal: int
    argv: tuple[str, ...]
    argv_sha256: str
    key: str

    @classmethod
    def from_argv(cls, ordinal: int, argv: Sequence[str]) -> "ScenarioSpec":
        normalized = validate_argv(argv)
        digest = canonical_argv_sha256(normalized)
        return cls(ordinal, normalized, digest, scenario_key(ordinal, normalized))


def atomic_write_json(path: Path, payload: Mapping[str, object], *, exclusive: bool = False) -> None:
    """Write a JSON object durably; exclusive final paths never get replaced.

    Windows ``os.rename`` is an atomic publish that raises ``FileExistsError`` if
    another writer has already published the final pathname.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.{os.getpid()}.",
        suffix=".tmp",
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        if exclusive:
            os.rename(temporary, path)
        else:
            _replace_with_windows_retry(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _replace_with_windows_retry(source: Path, destination: Path) -> None:
    deadline = time.monotonic() + _ATOMIC_REPLACE_RETRY_SECONDS
    while True:
        try:
            os.replace(source, destination)
            return
        except OSError as error:
            if (
                os.name != "nt"
                or getattr(error, "winerror", None) not in _RETRYABLE_WINDOWS_REPLACE_ERRORS
            ):
                raise
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise
            time.sleep(min(_ATOMIC_REPLACE_RETRY_DELAY_SECONDS, remaining))

