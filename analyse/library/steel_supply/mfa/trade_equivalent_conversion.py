"""Shared deterministic trade-price filling and compatibility helpers."""

from __future__ import annotations
from steel_supply.pipeline_paths import science_path, bundle_root, data_path

import numpy as np
import pandas as pd


SEMANTIC_KEY_CANDIDATES = [
    "trade_year",
    "trade_flow",
    "flow",
    "reporter_iso_a3",
    "reporter_comtrade_id",
    "partner_iso_a3",
    "partner_comtrade_id",
    "commodity_code",
]
PRICE_COLUMNS = [
    "commodity_code",
    "trade_year",
    "weight_unified_kg",
    "monetary_usd",
    "unit_price_usd_per_kg_interpolated",
]
_SIGN_MASK = np.uint64(1 << 63)


def _deterministic_sort_values(series: pd.Series) -> np.ndarray:
    """Return exact sortable values, distinguishing float bit patterns."""
    if pd.api.types.is_float_dtype(series.dtype):
        return series.to_numpy(dtype=np.float64, copy=False).view(np.uint64)
    if pd.api.types.is_integer_dtype(series.dtype) or pd.api.types.is_bool_dtype(
        series.dtype
    ):
        return series.to_numpy(copy=False)
    codes, _ = pd.factorize(series, sort=True, use_na_sentinel=True)
    return codes


def deterministic_trade_order(frame: pd.DataFrame) -> pd.DataFrame:
    """Sort trade rows by semantic key and exact remaining-value tie-breakers."""
    semantic_columns = [
        column for column in SEMANTIC_KEY_CANDIDATES if column in frame.columns
    ]
    if not semantic_columns:
        raise ValueError("Trade table has no recognized semantic key columns")
    remaining_columns = sorted(set(frame.columns).difference(semantic_columns))
    ordered_columns = semantic_columns + remaining_columns

    reset = frame.reset_index(drop=True)
    sort_keys = [
        _deterministic_sort_values(reset[column]) for column in ordered_columns
    ]
    order = np.lexsort(tuple(reversed(sort_keys)))
    return reset.iloc[order].reset_index(drop=True)


def sum_weight_and_money(
    trade: pd.DataFrame,
    commodity_code: int,
    trade_year: int,
    small_value: float,
) -> tuple[float | None, float | None]:
    """Apply the legacy filters and sums to one commodity-year group."""
    filtered = trade.loc[
        trade["commodity_code"].eq(commodity_code)
        & trade["trade_year"].eq(trade_year),
        ["weight_unified_kg", "monetary_usd"],
    ].copy()
    filtered.loc[
        filtered["weight_unified_kg"].le(small_value), "weight_unified_kg"
    ] = np.nan
    filtered.loc[
        filtered["monetary_usd"].le(small_value), "monetary_usd"
    ] = np.nan
    filtered = filtered.dropna(subset=["weight_unified_kg", "monetary_usd"])
    if filtered.empty:
        return None, None
    return (
        float(filtered["weight_unified_kg"].sum()),
        float(filtered["monetary_usd"].sum()),
    )


def build_price_table(
    ordered_trade: pd.DataFrame, small_value: float
) -> pd.DataFrame:
    """Build the unchanged legacy commodity-year price aggregates."""
    commodity_codes = ordered_trade["commodity_code"].drop_duplicates().tolist()
    years = ordered_trade["trade_year"].drop_duplicates().tolist()
    rows = []
    for commodity_code in commodity_codes:
        for year in years:
            weight_sum, price_sum = sum_weight_and_money(
                ordered_trade, commodity_code, year, small_value
            )
            unit_price = (
                None
                if weight_sum is None or price_sum is None
                else price_sum / weight_sum
            )
            rows.append(
                {
                    "commodity_code": commodity_code,
                    "trade_year": year,
                    "weight_unified_kg": weight_sum,
                    "monetary_usd": price_sum,
                    "unit_price_usd_per_kg": unit_price,
                }
            )

    return pd.DataFrame(rows)


def interpolate_price_table(price_table: pd.DataFrame) -> pd.DataFrame:
    """Apply the unchanged legacy within-commodity interpolation rules."""
    interpolated = price_table.copy()
    interpolated["unit_price_usd_per_kg_interpolated"] = interpolated.groupby(
        "commodity_code", sort=False
    )["unit_price_usd_per_kg"].transform(
        lambda series: series.interpolate(method="linear")
    )
    interpolated["unit_price_usd_per_kg_interpolated"] = interpolated.groupby(
        "commodity_code", sort=False
    )["unit_price_usd_per_kg_interpolated"].transform(
        lambda series: series.ffill().bfill()
    )
    return interpolated.drop(columns="unit_price_usd_per_kg")


def build_interpolated_price_table(
    ordered_trade: pd.DataFrame, small_value: float
) -> pd.DataFrame:
    """Build and interpolate a price table from deterministically ordered rows."""
    return interpolate_price_table(build_price_table(ordered_trade, small_value))


def fill_trade_values(
    ordered_trade: pd.DataFrame,
    interpolated_price_table: pd.DataFrame,
    small_value: float,
) -> pd.DataFrame:
    """Apply the unchanged legacy weight/money fill formulas."""
    filled = pd.merge(
        ordered_trade,
        interpolated_price_table[
            [
                "commodity_code",
                "trade_year",
                "unit_price_usd_per_kg_interpolated",
            ]
        ],
        on=["commodity_code", "trade_year"],
        how="left",
        sort=False,
        validate="many_to_one",
    )
    weight_small = filled["weight_unified_kg"].lt(small_value)
    money_small = filled["monetary_usd"].lt(small_value)
    filled.loc[weight_small, "weight_unified_kg"] = (
        filled["monetary_usd"]
        / filled["unit_price_usd_per_kg_interpolated"]
    )
    filled.loc[money_small, "monetary_usd"] = (
        filled["weight_unified_kg"]
        * filled["unit_price_usd_per_kg_interpolated"]
    )
    return filled


def plot_unit_price_trends(
    price_table: pd.DataFrame,
    enabled: bool = False,
    pause_seconds: float = 0.05,
) -> int:
    """Optionally show non-blocking diagnostic plots and close every figure."""
    if not enabled:
        return 0

    import matplotlib.pyplot as plt

    plotted = 0
    for code in price_table["commodity_code"].drop_duplicates():
        commodity = price_table.loc[price_table["commodity_code"].eq(code)]
        figure, axis = plt.subplots()
        axis.plot(commodity["trade_year"], commodity["unit_price_usd_per_kg"])
        axis.set_title(f"Unit Price Trend for Commodity Code: {code}")
        axis.set_xlabel("Year")
        axis.set_ylabel("Unit Price (USD per KG)")
        axis.grid(True)
        plt.show(block=False)
        plt.pause(pause_seconds)
        plt.close(figure)
        plotted += 1
    return plotted


def _ordered_float_bits(values: np.ndarray) -> np.ndarray:
    numeric = np.asarray(values, dtype=np.float64).copy()
    numeric[numeric == 0] = 0.0
    bits = numeric.view(np.uint64)
    negative = (bits & _SIGN_MASK) != 0
    return np.where(negative, ~bits, bits | _SIGN_MASK).astype(np.uint64)


def ulp_distance_float64(reference, candidate) -> np.ndarray:
    """Return exact float64 representable-step distances for finite values."""
    reference_array, candidate_array = np.broadcast_arrays(
        np.asarray(reference, dtype=np.float64),
        np.asarray(candidate, dtype=np.float64),
    )
    if not (
        np.isfinite(reference_array).all() and np.isfinite(candidate_array).all()
    ):
        raise ValueError("ULP distance is defined here only for finite float64 values")
    reference_ordered = _ordered_float_bits(reference_array)
    candidate_ordered = _ordered_float_bits(candidate_array)
    high = np.maximum(reference_ordered, candidate_ordered)
    low = np.minimum(reference_ordered, candidate_ordered)
    return high - low


def ulp_compatibility(reference, candidate, max_ulps: int = 16) -> dict:
    """Summarize a cellwise legacy compatibility gate."""
    reference_array, candidate_array = np.broadcast_arrays(
        np.asarray(reference, dtype=np.float64),
        np.asarray(candidate, dtype=np.float64),
    )
    both_nan = np.isnan(reference_array) & np.isnan(candidate_array)
    equal = reference_array == candidate_array
    changed = ~(equal | both_nan)
    finite_changed = changed & np.isfinite(reference_array) & np.isfinite(
        candidate_array
    )
    nonfinite_changed = changed & ~finite_changed

    distances = ulp_distance_float64(
        reference_array[finite_changed], candidate_array[finite_changed]
    )
    return {
        "changed_cell_count": int(changed.sum()),
        "max_ulp_distance": int(distances.max()) if distances.size else None,
        "violation_count": int(nonfinite_changed.sum())
        + int((distances > max_ulps).sum()),
    }

