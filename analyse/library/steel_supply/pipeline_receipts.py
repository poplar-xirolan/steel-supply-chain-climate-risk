"""Utilities for reproducible pipeline receipts and stable data digests."""

from __future__ import annotations
from steel_supply.pipeline_paths import science_path, bundle_root, data_path

import hashlib
import json
import os
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.ipc as pa_ipc


REQUIRED_RECEIPT_FIELDS = (
    "stage",
    "status",
    "code_commit",
    "input_paths",
    "input_hashes",
    "parameters",
    "row_counts",
    "coverage",
    "output_paths",
    "output_hashes",
    "checks",
    "created_at_utc",
)


def _validate_receipt(payload: dict[str, object]) -> None:
    missing = [field for field in REQUIRED_RECEIPT_FIELDS if field not in payload]
    if missing:
        raise ValueError(f"Receipt is missing required fields: {', '.join(missing)}")


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Return a SHA-256 digest while reading a file in bounded chunks."""
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, Integral) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_parquet_digest(path: Path, sort_columns: list[str]) -> dict[str, object]:
    """Digest Parquet data after stable sorting by semantic row keys."""
    frame = pd.read_parquet(path)
    canonical_sort_columns = sort_columns + [
        column for column in frame.columns if column not in sort_columns
    ]
    working_frame = frame.copy()
    float_bit_sort_columns: list[str] = []
    for column in frame.columns:
        values = frame[column].to_numpy(copy=False)
        if not np.issubdtype(values.dtype, np.floating):
            continue
        bit_column = f"__pipeline_receipts_float_bits_{len(float_bit_sort_columns)}"
        while bit_column in working_frame.columns:
            bit_column = f"_{bit_column}"
        working_frame[bit_column] = values.view(f"uint{values.dtype.itemsize * 8}")
        float_bit_sort_columns.append(bit_column)
    sorted_frame = working_frame.sort_values(
        canonical_sort_columns + float_bit_sort_columns,
        kind="stable",
    ).drop(columns=float_bit_sort_columns).reset_index(drop=True)
    table = pa.Table.from_pandas(sorted_frame, preserve_index=False).replace_schema_metadata()
    sink = pa.BufferOutputStream()
    with pa_ipc.new_stream(sink, table.schema) as writer:
        writer.write_table(table)
    return {
        "path": str(path),
        "row_count": len(frame),
        "schema": str(table.schema),
        "sha256": hashlib.sha256(sink.getvalue().to_pybytes()).hexdigest(),
    }


def write_receipt(path: Path, payload: dict[str, object]) -> None:
    """Atomically write and validate a receipt using a sibling temporary file."""
    _validate_receipt(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    try:
        with temporary_path.open("w", encoding="utf-8", newline="\n") as temporary_file:
            json.dump(payload, temporary_file, ensure_ascii=False, indent=2, sort_keys=True)
            temporary_file.write("\n")
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    with path.open("r", encoding="utf-8") as receipt_file:
        written_payload: Any = json.load(receipt_file)
    if not isinstance(written_payload, dict):
        raise ValueError(f"Receipt {path} must contain a JSON object")
    _validate_receipt(written_payload)


def require_pass(receipt_path: Path) -> dict[str, object]:
    """Return a valid passing receipt or reject an upstream stage."""
    with receipt_path.open("r", encoding="utf-8") as receipt_file:
        payload: Any = json.load(receipt_file)
    if not isinstance(payload, dict):
        raise ValueError(f"Receipt {receipt_path} must contain a JSON object")
    _validate_receipt(payload)
    if payload["status"] != "PASS":
        raise ValueError(f"Upstream receipt {receipt_path} status must be PASS")
    return payload

