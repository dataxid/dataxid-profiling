from __future__ import annotations

from typing import TYPE_CHECKING, Any

import polars as pl

from dataxid_profiling._analyzers import DatetimeStats
from dataxid_profiling._type_inference import ColumnType

if TYPE_CHECKING:
    from dataxid_profiling._config import ProfileConfig


def analyze_datetime(df: pl.DataFrame, col_name: str, config: ProfileConfig) -> DatetimeStats:  # noqa: ARG001
    col = pl.col(col_name)
    n_rows = df.height

    if n_rows == 0:
        return _empty_stats(col_name)

    row = df.select(
        col.null_count().alias("missing_count"),
        col.drop_nulls().n_unique().alias("distinct_count"),
        col.min().alias("min"),
        col.max().alias("max"),
    ).row(0, named=True)

    missing_count: int = row["missing_count"]
    distinct_count: int = row["distinct_count"]
    min_val = row["min"]
    max_val = row["max"]

    range_str = _compute_range(min_val, max_val)

    return DatetimeStats(
        column_name=col_name,
        column_type=ColumnType.DATETIME,
        count=n_rows,
        missing_count=missing_count,
        missing_pct=missing_count / n_rows,
        distinct_count=distinct_count,
        distinct_pct=distinct_count / n_rows,
        min=str(min_val) if min_val is not None else None,
        max=str(max_val) if max_val is not None else None,
        range=range_str,
    )


def _compute_range(min_val: Any, max_val: Any) -> str | None:
    if min_val is None or max_val is None:
        return None
    try:
        delta = max_val - min_val
        return str(delta)
    except TypeError:
        return None


def _empty_stats(col_name: str) -> DatetimeStats:
    return DatetimeStats(
        column_name=col_name,
        column_type=ColumnType.DATETIME,
        count=0,
        missing_count=0,
        missing_pct=0.0,
    )
