from __future__ import annotations

from typing import TYPE_CHECKING, Any

import polars as pl

from dataxid_profiling._analyzers import NumericStats
from dataxid_profiling._type_inference import ColumnType

if TYPE_CHECKING:
    from dataxid_profiling._config import ProfileConfig


def analyze_numeric(df: pl.DataFrame, col_name: str, config: ProfileConfig) -> NumericStats:
    col = pl.col(col_name)
    n_rows = df.height

    if n_rows == 0:
        return _empty_stats(col_name)

    row = df.select(
        col.count().alias("count"),
        col.null_count().alias("missing_count"),
        col.n_unique().alias("distinct_count"),
        col.mean().alias("mean"),
        col.std().alias("std"),
        col.min().alias("min"),
        col.max().alias("max"),
        col.quantile(0.25, interpolation="linear").alias("q25"),
        col.median().alias("median"),
        col.quantile(0.75, interpolation="linear").alias("q75"),
        col.skew().alias("skewness"),
        col.kurtosis().alias("kurtosis"),
        col.eq(0).sum().alias("zeros_count"),
        col.lt(0).sum().alias("negative_count"),
    ).row(0, named=True)

    missing_count: int = row["missing_count"]
    distinct_count: int = row["distinct_count"]
    min_val = row["min"]
    max_val = row["max"]
    q25 = row["q25"]
    q75 = row["q75"]

    range_val = (max_val - min_val) if min_val is not None and max_val is not None else None
    iqr = (q75 - q25) if q25 is not None and q75 is not None else None

    histogram = _compute_histogram(df, col_name, config.histogram_bins)

    return NumericStats(
        column_name=col_name,
        column_type=ColumnType.NUMERIC,
        count=n_rows,
        missing_count=missing_count,
        missing_pct=missing_count / n_rows if n_rows > 0 else 0.0,
        distinct_count=distinct_count,
        distinct_pct=distinct_count / n_rows if n_rows > 0 else 0.0,
        mean=_safe_float(row["mean"]),
        std=_safe_float(row["std"]),
        min=_safe_float(min_val),
        max=_safe_float(max_val),
        range=_safe_float(range_val),
        q25=_safe_float(q25),
        median=_safe_float(row["median"]),
        q75=_safe_float(q75),
        iqr=_safe_float(iqr),
        skewness=_safe_float(row["skewness"]),
        kurtosis=_safe_float(row["kurtosis"]),
        zeros_count=row["zeros_count"],
        zeros_pct=row["zeros_count"] / n_rows if n_rows > 0 else 0.0,
        negative_count=row["negative_count"],
        negative_pct=row["negative_count"] / n_rows if n_rows > 0 else 0.0,
        histogram=histogram,
    )


def _compute_histogram(
    df: pl.DataFrame, col_name: str, bin_count: int
) -> list[dict[str, Any]]:
    try:
        hist_df = df.select(
            pl.col(col_name).hist(bin_count=bin_count, include_breakpoint=True)
        ).unnest(col_name)

        return [
            {
                "breakpoint": row["breakpoint"],
                "count": row["count"],
            }
            for row in hist_df.iter_rows(named=True)
        ]
    except Exception:
        return []


def _safe_float(val: Any) -> float | None:
    if val is None:
        return None
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _empty_stats(col_name: str) -> NumericStats:
    return NumericStats(
        column_name=col_name,
        column_type=ColumnType.NUMERIC,
        count=0,
        missing_count=0,
        missing_pct=0.0,
    )
