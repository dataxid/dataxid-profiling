"""Correlation matrices — Polars-native."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from dataxid_profiling._type_inference import ColumnType

if TYPE_CHECKING:
    from dataxid_profiling._config import ProfileConfig


def compute_correlations(
    df: pl.DataFrame,
    column_types: dict[str, ColumnType],
    config: ProfileConfig | None = None,
) -> dict[str, pl.DataFrame]:
    """Compute correlation matrices for the given DataFrame.

    Returns a dict keyed by method name (e.g. "pearson").
    Empty dict when fewer than 2 numeric columns or overview mode.
    """
    if config is not None and config.is_overview:
        return {}

    numeric_cols = sorted(
        col for col, ct in column_types.items() if ct is ColumnType.NUMERIC
    )

    if len(numeric_cols) < 2:
        return {}

    return {"pearson": _pearson_matrix(df, numeric_cols)}


def _pearson_matrix(df: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    """Build a Pearson correlation matrix using pl.corr()."""
    rows: list[dict[str, float | str]] = []
    for col_a in columns:
        row: dict[str, float | str] = {"column": col_a}
        for col_b in columns:
            if col_a == col_b:
                row[col_b] = 1.0
            else:
                val = df.select(pl.corr(col_a, col_b, method="pearson")).item()
                row[col_b] = float(val) if val is not None else 0.0
        rows.append(row)

    return pl.DataFrame(rows)
