from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from dataxid_profiling._analyzers import TextStats
from dataxid_profiling._type_inference import ColumnType

if TYPE_CHECKING:
    from dataxid_profiling._config import ProfileConfig


def analyze_text(df: pl.DataFrame, col_name: str, config: ProfileConfig) -> TextStats:  # noqa: ARG001
    col = pl.col(col_name)
    n_rows = df.height

    if n_rows == 0:
        return _empty_stats(col_name)

    row = df.select(
        col.null_count().alias("missing_count"),
        col.drop_nulls().n_unique().alias("distinct_count"),
        col.str.len_chars().mean().alias("length_mean"),
        col.str.len_chars().min().alias("length_min"),
        col.str.len_chars().max().alias("length_max"),
    ).row(0, named=True)

    missing_count: int = row["missing_count"]
    distinct_count: int = row["distinct_count"]

    return TextStats(
        column_name=col_name,
        column_type=ColumnType.TEXT,
        count=n_rows,
        missing_count=missing_count,
        missing_pct=missing_count / n_rows,
        distinct_count=distinct_count,
        distinct_pct=distinct_count / n_rows,
        length_mean=float(row["length_mean"]) if row["length_mean"] is not None else None,
        length_min=row["length_min"],
        length_max=row["length_max"],
    )


def _empty_stats(col_name: str) -> TextStats:
    return TextStats(
        column_name=col_name,
        column_type=ColumnType.TEXT,
        count=0,
        missing_count=0,
        missing_pct=0.0,
    )
