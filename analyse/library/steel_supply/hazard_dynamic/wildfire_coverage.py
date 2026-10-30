"""Shared fail-closed selection and coverage checks for wildfire ensembles."""

from __future__ import annotations
from steel_supply.pipeline_paths import science_path, bundle_root, data_path

from pathlib import Path
import re

import numpy as np
import pandas as pd


FORMAL_CLIMATE_SCENARIOS = ("picontrol", "ssp126", "ssp370", "ssp585")
FORMAL_START_YEAR = 2015
FORMAL_END_YEAR = 2100
WILDFIRE_FILENAME_PATTERN = re.compile(
    r"(?P<forcing_member>[^_]+)_"
    r"(?P<model>[^_]+)_"
    r"(?P<bias_adjustment>[^_]+)_"
    r"(?P<climate_scenario>[^_]+)_"
    r"(?P<soc_scenario>[^_+]+)"
    r"(?:\+(?P<lu_model>[^_]+))?_"
    r"(?P<sens_scenario>[^_]+)_"
    r"(?P<variable>[^_]+)_"
    r"(?P<region>[^_]+)_"
    r"(?P<time_step>[^_]+)_"
    r"(?P<start_year>\d{4})_"
    r"(?P<end_year>\d{4})"
)


_METADATA_COLUMNS = {
    "dataset_source",
    "climate_scenario",
    "model",
    "forcing_member",
    "sens_scenario",
    "soc_scenario",
    "start_year",
    "end_year",
    "source_nc",
    "raw_filename",
    "filepath",
}
_MODEL_COLUMNS = ["model", "forcing_member", "source_nc"]
_DATA_KEY_COLUMNS = ["id_internal", "month_start", "model", "forcing_member"]
_VALUE_COLUMN = "burntarea-total"


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{label} missing required columns: {', '.join(missing)}")


def selected_ensemble_keys(
    metadata: pd.DataFrame,
    source: str,
    scenario: str,
) -> pd.DataFrame:
    """Return one explicit source file per selected model/forcing pair.

    ``2015soc-from-histsoc`` is preferred independently for every
    model/forcing pair, with ``2015soc`` retained only as its fallback.
    """
    _require_columns(metadata, _METADATA_COLUMNS, "wildfire metadata")
    if metadata[list(_METADATA_COLUMNS)].isna().any().any():
        raise ValueError("wildfire metadata contains missing selection values")

    start_year = pd.to_numeric(metadata["start_year"], errors="raise")
    candidates = metadata.loc[
        metadata["dataset_source"].eq(source)
        & metadata["climate_scenario"].eq(scenario)
        & metadata["sens_scenario"].eq("default")
        & metadata["soc_scenario"].isin(["2015soc", "2015soc-from-histsoc"])
        & start_year.ge(2000)
    ].copy()
    if candidates.empty:
        raise ValueError(
            f"empty selected wildfire ensemble for source={source!r}, "
            f"scenario={scenario!r}"
        )

    selected_parts: list[pd.DataFrame] = []
    for _, group in candidates.groupby(["model", "forcing_member"], sort=False):
        preferred_soc = (
            "2015soc-from-histsoc"
            if group["soc_scenario"].eq("2015soc-from-histsoc").any()
            else "2015soc"
        )
        selected_parts.append(group.loc[group["soc_scenario"].eq(preferred_soc)])

    selected = pd.concat(selected_parts, ignore_index=True)
    duplicated = selected.duplicated(["model", "forcing_member"], keep=False)
    if duplicated.any():
        duplicate_count = int(duplicated.sum())
        raise ValueError(
            f"duplicate selected ensemble key for {duplicate_count} metadata rows"
        )
    if selected["source_nc"].duplicated().any():
        raise ValueError("duplicate selected source-file key")

    return selected.sort_values(
        ["model", "forcing_member", "source_nc"], kind="stable"
    ).reset_index(drop=True)


def select_formal_wildfire_sources(paths: list[Path]) -> list[Path]:
    """Select exactly the 2015--2100 fixed ensemble consumed by step 09."""
    records: list[dict[str, object]] = []
    malformed: list[str] = []
    for path in paths:
        match = WILDFIRE_FILENAME_PATTERN.fullmatch(path.stem)
        if match is None:
            malformed.append(path.name)
            continue
        records.append(
            {
                "dataset_source": "raw_wildfire",
                "source_nc": path.name,
                "raw_filename": path.name,
                "filepath": str(path.resolve()),
                **match.groupdict(),
            }
        )
    if malformed:
        raise ValueError(f"malformed wildfire NetCDF filenames: {malformed[:3]}")
    if not records:
        raise ValueError("no parseable wildfire NetCDF inputs found")

    metadata = pd.DataFrame(records)
    selected_parts: list[pd.DataFrame] = []
    for scenario in FORMAL_CLIMATE_SCENARIOS:
        selected = selected_ensemble_keys(metadata, "raw_wildfire", scenario)
        if not selected["start_year"].astype(int).eq(FORMAL_START_YEAR).all() or not selected[
            "end_year"
        ].astype(int).eq(FORMAL_END_YEAR).all():
            raise ValueError(
                "selected wildfire source bounds do not match the formal interval for "
                f"{scenario}"
            )
        selected_parts.append(selected)

    selected_metadata = pd.concat(selected_parts, ignore_index=True)
    if selected_metadata["filepath"].duplicated().any():
        raise ValueError("duplicate formal wildfire source path")
    return sorted(Path(value) for value in selected_metadata["filepath"])


def _validated_expected_models(expected_models: pd.DataFrame) -> pd.DataFrame:
    _require_columns(expected_models, set(_MODEL_COLUMNS), "expected models")
    if expected_models.empty:
        raise ValueError("expected wildfire model table is empty")
    models = expected_models[_MODEL_COLUMNS].copy()
    if models.isna().any().any():
        raise ValueError("expected wildfire model table contains missing values")
    if models.duplicated(["model", "forcing_member"]).any():
        raise ValueError("duplicate expected model/forcing key")
    if models["source_nc"].duplicated().any():
        raise ValueError("duplicate expected source-file key")
    return models


def _validated_axes(
    expected_ids: pd.Index,
    expected_months: pd.DatetimeIndex,
) -> tuple[pd.Index, pd.DatetimeIndex]:
    ids = pd.Index(expected_ids, name="id_internal")
    months = pd.DatetimeIndex(expected_months, name="month_start")
    if ids.empty or ids.hasnans or not ids.is_unique:
        raise ValueError("expected facility IDs must be nonempty, non-missing, and unique")
    if months.empty or months.hasnans or not months.is_unique:
        raise ValueError("expected months must be nonempty, non-missing, and unique")
    if not months.equals(months.to_period("M").to_timestamp()):
        raise ValueError("expected months must be calendar month starts")
    return ids, months


def validate_model_coverage(
    data: pd.DataFrame,
    expected_models: pd.DataFrame,
    expected_ids: pd.Index,
    expected_months: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Require the complete facility x month x selected-model matrix."""
    _require_columns(data, set(_DATA_KEY_COLUMNS + ["source_nc", _VALUE_COLUMN]), "wildfire data")
    models = _validated_expected_models(expected_models)
    ids, months = _validated_axes(expected_ids, expected_months)

    if data.empty:
        raise ValueError("wildfire data is empty")
    if data[_DATA_KEY_COLUMNS + ["source_nc"]].isna().any().any():
        raise ValueError("wildfire data contains missing key values")
    values = pd.to_numeric(data[_VALUE_COLUMN], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("non-finite wildfire value")
    if data.duplicated(_DATA_KEY_COLUMNS).any():
        raise ValueError("duplicate facility-model-month key")

    actual_ids = pd.Index(data["id_internal"].unique())
    if set(actual_ids) != set(ids):
        raise ValueError("wildfire facility IDs do not match the expected set")
    actual_months = pd.DatetimeIndex(data["month_start"].unique())
    if set(actual_months) != set(months):
        raise ValueError("wildfire months do not match the expected set")

    actual_model_rows = data[["model", "forcing_member", "source_nc"]].drop_duplicates()
    source_counts = actual_model_rows.groupby(
        ["model", "forcing_member"], observed=True
    )["source_nc"].nunique()
    if source_counts.gt(1).any():
        raise ValueError("observed model/forcing pair uses multiple source files")
    expected_model_keys = set(models.itertuples(index=False, name=None))
    actual_model_keys = set(actual_model_rows.itertuples(index=False, name=None))
    if (
        len(actual_model_rows) != len(models)
        or actual_model_keys != expected_model_keys
    ):
        raise ValueError("observed model/forcing/source-file keys do not match selection")

    observed = (
        data.groupby(["id_internal", "month_start"], sort=True, observed=True)
        .size()
        .rename("observed_model_count")
        .reset_index()
    )
    expected_model_count = len(models)
    expected_group_count = len(ids) * len(months)
    expected_row_count = expected_group_count * expected_model_count
    missing_count = expected_row_count - len(data)
    if (
        len(observed) != expected_group_count
        or missing_count != 0
        or not observed["observed_model_count"].eq(expected_model_count).all()
    ):
        raise ValueError(
            f"missing {max(missing_count, 0)} facility-model-month keys; "
            f"every facility-month must have observed model count "
            f"{expected_model_count}"
        )

    observed.insert(2, "expected_model_count", expected_model_count)
    return observed[
        [
            "id_internal",
            "month_start",
            "expected_model_count",
            "observed_model_count",
        ]
    ]


def fixed_ensemble_mean(
    data: pd.DataFrame,
    expected_model_count: int,
) -> pd.DataFrame:
    """Calculate an arithmetic mean only for complete fixed-size groups."""
    if isinstance(expected_model_count, bool) or expected_model_count <= 0:
        raise ValueError("expected_model_count must be a positive integer")
    _require_columns(data, set(_DATA_KEY_COLUMNS + [_VALUE_COLUMN]), "wildfire data")
    if data.duplicated(_DATA_KEY_COLUMNS).any():
        raise ValueError("duplicate facility-model-month key")
    values = pd.to_numeric(data[_VALUE_COLUMN], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("non-finite wildfire value")

    grouped = data.groupby(
        ["id_internal", "month_start"], sort=True, observed=True
    )[_VALUE_COLUMN]
    result = grouped.agg([("burntarea-total", "mean"), ("observed_model_count", "size")]).reset_index()
    if not result["observed_model_count"].eq(expected_model_count).all():
        raise ValueError(
            f"observed model count must equal selected ensemble count "
            f"{expected_model_count} for every facility-month"
        )
    return result

