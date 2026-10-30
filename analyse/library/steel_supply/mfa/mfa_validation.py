"""Shared validation and impact-comparison helpers for annual steel MFA tables."""

from __future__ import annotations
from steel_supply.pipeline_paths import science_path, bundle_root, data_path

import re

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = (
    "entity_from",
    "entity_to",
    "sector_from",
    "sector_to",
    "mass",
    "unit",
    "extra",
)
ROW_KEY_COLUMNS = (
    "entity_from",
    "entity_to",
    "sector_from",
    "sector_to",
    "extra",
)
BOUNDARY_SECTORS = frozenset({"P1", "P7", "Environment", "S1", "Phantom Flow"})
CORRECTION_LABELS = frozenset(
    {
        "Phantom Flow Correction",
        "Process Balance Adjustment",
        "ASU-FSP final demand floor for non-positive raw M8 balance",
    }
)
PROCESS_BALANCE_EDGES = frozenset(
    {
        ("M1", "P2"),
        ("M2", "P3"),
        ("M2", "P4"),
        ("M3", "P5"),
        ("M6", "P6"),
        ("P2", "M2"),
        ("P3", "M3"),
        ("P4", "M4"),
        ("P5", "M5"),
        ("P6", "M7"),
    }
)
NODE_BALANCE_TOLERANCE_KG = 1e-3
_TRADE_LABEL_PATTERN = re.compile(r"TRADE_SITC_\d+")
_CORRECTION_MARKERS = ("Phantom", "Correction", "Adjustment", "ASU-FSP")


def _empty_schema_result(missing_columns: list[str]) -> dict[str, object]:
    checks = {
        "required_columns": False,
        "non_null_required_values": False,
        "unique_row_keys": False,
        "finite_mass": False,
        "nonnegative_mass": False,
        "unit_kg": False,
        "trade_labels": False,
        "correction_labels": False,
        "phantom_edge_labels": False,
        "bilateral_trade_edges": False,
        "global_bilateral_balance_by_material": False,
    }
    return {
        "status": "FAIL",
        "row_count": 0,
        "missing_required_columns": missing_columns,
        "invalid_counts": {},
        "invalid_trade_labels": [],
        "invalid_correction_labels": [],
        "bilateral_balance": [],
        "checks": checks,
    }


def validate_mfa_schema(frame: pd.DataFrame) -> dict[str, object]:
    """Validate schema, row keys, units, labels, and bilateral trade invariants."""
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in frame]
    if missing_columns:
        result = _empty_schema_result(missing_columns)
        result["row_count"] = len(frame)
        return result

    required_non_null = [column for column in REQUIRED_COLUMNS if column != "extra"]
    null_required = int(frame.loc[:, required_non_null].isna().sum().sum())
    duplicate_row_keys = int(frame.duplicated(list(ROW_KEY_COLUMNS), keep=False).sum())

    numeric_mass = pd.to_numeric(frame["mass"], errors="coerce")
    finite_mass_mask = np.isfinite(numeric_mass.to_numpy(dtype="float64", na_value=np.nan))
    nonfinite_mass = int((~finite_mass_mask).sum())
    negative_mass = int(((numeric_mass < 0) & finite_mass_mask).sum())
    non_kg_unit = int(frame["unit"].ne("kg").sum())

    labels = frame["extra"].dropna().astype(str)
    invalid_trade_labels = sorted(
        {
            label
            for label in labels
            if label.startswith("TRADE_SITC_") and _TRADE_LABEL_PATTERN.fullmatch(label) is None
        }
    )
    invalid_correction_labels = sorted(
        {
            label
            for label in labels
            if any(marker in label for marker in _CORRECTION_MARKERS)
            and label not in CORRECTION_LABELS
        }
    )

    extra_as_string = frame["extra"].astype("string")
    phantom_edge = frame["sector_from"].eq("Phantom Flow") | frame["sector_to"].eq(
        "Phantom Flow"
    )
    phantom_label = extra_as_string.eq("Phantom Flow Correction").fillna(False)
    invalid_phantom_edge_label = int((phantom_edge ^ phantom_label).sum())

    process_label = extra_as_string.eq("Process Balance Adjustment").fillna(False)
    process_edges = pd.Series(
        list(zip(frame["sector_from"], frame["sector_to"])), index=frame.index
    ).isin(PROCESS_BALANCE_EDGES)
    invalid_process_edge_label = int(
        (process_label & (~process_edges | frame["entity_from"].ne(frame["entity_to"]))).sum()
    )

    asu_label = extra_as_string.eq(
        "ASU-FSP final demand floor for non-positive raw M8 balance"
    ).fillna(False)
    valid_asu_edge = (
        frame["sector_from"].eq("M8")
        & frame["sector_to"].eq("S1")
        & frame["entity_from"].eq(frame["entity_to"])
    )
    invalid_asu_edge_label = int((asu_label & ~valid_asu_edge).sum())

    trade_mask = extra_as_string.str.fullmatch(r"TRADE_SITC_\d+", na=False)
    trade_rows = frame.loc[trade_mask]
    valid_trade_edges = (
        trade_rows["entity_from"].ne(trade_rows["entity_to"])
        & trade_rows["sector_from"].eq(trade_rows["sector_to"])
        & trade_rows["sector_from"].astype(str).str.fullmatch(r"M\d+")
    )
    invalid_trade_edges = int((~valid_trade_edges).sum())

    exports = trade_rows.groupby("sector_from", sort=True, dropna=False)["mass"].sum()
    imports = trade_rows.groupby("sector_to", sort=True, dropna=False)["mass"].sum()
    materials = sorted(set(exports.index).union(imports.index), key=str)
    bilateral_balance = []
    for material in materials:
        export_mass = float(exports.get(material, 0.0))
        import_mass = float(imports.get(material, 0.0))
        bilateral_balance.append(
            {
                "material": str(material),
                "export_mass_kg": export_mass,
                "import_mass_kg": import_mass,
                "net_mass_kg": import_mass - export_mass,
            }
        )
    global_bilateral_balanced = all(
        row["net_mass_kg"] == 0.0 for row in bilateral_balance
    )

    invalid_counts = {
        "null_required_values": null_required,
        "duplicate_row_keys": duplicate_row_keys,
        "nonfinite_mass": nonfinite_mass,
        "negative_mass": negative_mass,
        "non_kg_unit": non_kg_unit,
        "invalid_trade_label": len(invalid_trade_labels),
        "invalid_correction_label": len(invalid_correction_labels),
        "phantom_edge_label": invalid_phantom_edge_label,
        "process_balance_edge_label": invalid_process_edge_label,
        "asu_fsp_edge_label": invalid_asu_edge_label,
        "bilateral_trade_edge": invalid_trade_edges,
    }
    checks = {
        "required_columns": True,
        "non_null_required_values": null_required == 0,
        "unique_row_keys": duplicate_row_keys == 0,
        "finite_mass": nonfinite_mass == 0,
        "nonnegative_mass": negative_mass == 0,
        "unit_kg": non_kg_unit == 0,
        "trade_labels": not invalid_trade_labels,
        "correction_labels": (
            not invalid_correction_labels
            and invalid_process_edge_label == 0
            and invalid_asu_edge_label == 0
        ),
        "phantom_edge_labels": invalid_phantom_edge_label == 0,
        "bilateral_trade_edges": invalid_trade_edges == 0,
        "global_bilateral_balance_by_material": global_bilateral_balanced,
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "row_count": len(frame),
        "missing_required_columns": [],
        "invalid_counts": invalid_counts,
        "invalid_trade_labels": invalid_trade_labels,
        "invalid_correction_labels": invalid_correction_labels,
        "bilateral_balance": bilateral_balance,
        "checks": checks,
    }


def calculate_node_balance(frame: pd.DataFrame) -> pd.DataFrame:
    """Return inflow, outflow, and residual mass for every country-sector node."""
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in frame]
    if missing_columns:
        raise ValueError(f"Missing required MFA columns: {', '.join(missing_columns)}")

    inflow = frame.groupby(["entity_to", "sector_to"], dropna=False)["mass"].sum()
    inflow.index = inflow.index.set_names(["entity", "sector"])
    outflow = frame.groupby(["entity_from", "sector_from"], dropna=False)["mass"].sum()
    outflow.index = outflow.index.set_names(["entity", "sector"])
    result = pd.concat(
        [inflow.rename("inflow_kg"), outflow.rename("outflow_kg")], axis=1
    ).fillna(0.0)
    result["net_inflow_kg"] = result["inflow_kg"] - result["outflow_kg"]
    result["abs_imbalance_kg"] = result["net_inflow_kg"].abs()
    result["is_boundary"] = result.index.get_level_values("sector").isin(
        BOUNDARY_SECTORS
    )
    result["is_balanced"] = result["abs_imbalance_kg"] < NODE_BALANCE_TOLERANCE_KG
    return result.reset_index().sort_values(["entity", "sector"]).reset_index(drop=True)


def _aggregate_comparison(
    reference: pd.DataFrame,
    candidate: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
    def aggregate(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
        return (
            frame.groupby(group_columns, dropna=False, sort=True)
            .agg(
                **{
                    f"{prefix}_rows": ("mass", "size"),
                    f"{prefix}_mass_kg": ("mass", "sum"),
                }
            )
            .reset_index()
        )

    merged = aggregate(reference, "reference").merge(
        aggregate(candidate, "candidate"), how="outer", on=group_columns
    )
    for column in (
        "reference_rows",
        "reference_mass_kg",
        "candidate_rows",
        "candidate_mass_kg",
    ):
        merged[column] = merged[column].fillna(0)
    merged["row_delta"] = merged["candidate_rows"] - merged["reference_rows"]
    merged["mass_delta_kg"] = (
        merged["candidate_mass_kg"] - merged["reference_mass_kg"]
    )
    return merged.sort_values(group_columns, kind="stable", na_position="first").reset_index(
        drop=True
    )


def compare_mfa(
    reference: pd.DataFrame, candidate: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    """Compare MFA mass by row key and independent scientific impact views."""
    for name, frame in (("reference", reference), ("candidate", candidate)):
        missing = [column for column in REQUIRED_COLUMNS if column not in frame]
        if missing:
            raise ValueError(f"{name} is missing required MFA columns: {', '.join(missing)}")

    row_key = _aggregate_comparison(reference, candidate, list(ROW_KEY_COLUMNS))
    edge = _aggregate_comparison(reference, candidate, ["sector_from", "sector_to"])

    def country_flows(frame: pd.DataFrame) -> pd.DataFrame:
        outflow = frame.rename(columns={"entity_from": "entity"}).assign(
            direction="outflow"
        )
        inflow = frame.rename(columns={"entity_to": "entity"}).assign(
            direction="inflow"
        )
        return pd.concat([inflow, outflow], ignore_index=True)

    country = _aggregate_comparison(
        country_flows(reference), country_flows(candidate), ["entity", "direction"]
    )

    reference_stage = reference.rename(columns={"sector_from": "stage"})
    candidate_stage = candidate.rename(columns={"sector_from": "stage"})
    stage = _aggregate_comparison(reference_stage, candidate_stage, ["stage"])

    def material_market(frame: pd.DataFrame) -> pd.DataFrame:
        mask = frame["extra"].astype("string").str.fullmatch(r"TRADE_SITC_\d+", na=False)
        return frame.loc[mask].assign(material_market=frame.loc[mask, "sector_from"])

    market = _aggregate_comparison(
        material_market(reference), material_market(candidate), ["material_market"]
    )

    def corrections(frame: pd.DataFrame) -> pd.DataFrame:
        return frame.loc[frame["extra"].isin(CORRECTION_LABELS)].rename(
            columns={"extra": "correction_label"}
        )

    correction_label = _aggregate_comparison(
        corrections(reference), corrections(candidate), ["correction_label"]
    )
    return {
        "row_key": row_key,
        "edge": edge,
        "country": country,
        "stage": stage,
        "material_market": market,
        "correction_label": correction_label,
    }

