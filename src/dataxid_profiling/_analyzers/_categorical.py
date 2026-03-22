from __future__ import annotations

from typing import TYPE_CHECKING, Any

import polars as pl

from dataxid_profiling._analyzers import CategoricalStats
from dataxid_profiling._type_inference import ColumnType

if TYPE_CHECKING:
    from dataxid_profiling._config import ProfileConfig


def analyze_categorical(
    df: pl.DataFrame, col_name: str, config: ProfileConfig
) -> CategoricalStats:
    col = pl.col(col_name)
    n_rows = df.height

    if n_rows == 0:
        return _empty_stats(col_name)

    row = df.select(
        col.null_count().alias("missing_count"),
        col.drop_nulls().n_unique().alias("distinct_count"),
    ).row(0, named=True)

    missing_count: int = row["missing_count"]
    distinct_count: int = row["distinct_count"]

    top_values = _compute_top_values(df, col_name, config.n_top_values)

    return CategoricalStats(
        column_name=col_name,
        column_type=ColumnType.CATEGORICAL,
        count=n_rows,
        missing_count=missing_count,
        missing_pct=missing_count / n_rows,
        distinct_count=distinct_count,
        distinct_pct=distinct_count / n_rows,
        top_values=top_values,
    )


def _compute_top_values(
    df: pl.DataFrame, col_name: str, n_top: int
) -> list[dict[str, Any]]:
    vc = (
        df.select(pl.col(col_name))
        .drop_nulls()
        .group_by(col_name)
        .len()
        .sort("len", descending=True)
        .head(n_top)
    )

    return [
        {"value": row[col_name], "count": row["len"]}
        for row in vc.iter_rows(named=True)
    ]


def _empty_stats(col_name: str) -> CategoricalStats:
    return CategoricalStats(
        column_name=col_name,
        column_type=ColumnType.CATEGORICAL,
        count=0,
        missing_count=0,
        missing_pct=0.0,
    )
