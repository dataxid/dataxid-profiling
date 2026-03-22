from __future__ import annotations

from enum import Enum, auto

import polars as pl
import polars.selectors as cs

from dataxid_profiling._config import ProfileConfig


class ColumnType(Enum):
    NUMERIC = auto()
    CATEGORICAL = auto()
    BOOLEAN = auto()
    DATETIME = auto()
    TEXT = auto()
    UNSUPPORTED = auto()


_TEMPORAL_BASES: frozenset[type[pl.DataType]] = frozenset(
    {pl.Date, pl.Datetime, pl.Time, pl.Duration}
)


def infer_types(
    df: pl.DataFrame,
    config: ProfileConfig | None = None,
) -> dict[str, ColumnType]:
    """Infer semantic column types from a Polars DataFrame.

    Uses Polars dtype as the primary signal, then applies heuristics
    (e.g. unique ratio) to distinguish CATEGORICAL from TEXT.
    """
    if config is None:
        config = ProfileConfig()

    numeric_cols = set(df.select(cs.numeric()).columns)
    n_rows = df.height

    string_cols = [
        col for col, dtype in zip(df.columns, df.dtypes, strict=True)
        if type(dtype) in (pl.String, pl.Utf8)
    ]
    unique_ratios = _batch_unique_ratios(df, string_cols, n_rows)

    result: dict[str, ColumnType] = {}
    for col_name, dtype in zip(df.columns, df.dtypes, strict=True):
        result[col_name] = _infer_single(
            col_name, dtype, numeric_cols, unique_ratios, config,
        )

    return result


def _batch_unique_ratios(
    df: pl.DataFrame,
    string_cols: list[str],
    n_rows: int,
) -> dict[str, float]:
    """Compute unique ratio for all string columns in a single select."""
    if not string_cols or n_rows == 0:
        return {}

    exprs = []
    for col in string_cols:
        exprs.extend([
            pl.col(col).drop_nulls().n_unique().alias(f"{col}__n_unique"),
            (pl.lit(n_rows) - pl.col(col).null_count()).alias(f"{col}__non_null"),
        ])

    row = df.select(exprs).row(0, named=True)

    ratios: dict[str, float] = {}
    for col in string_cols:
        non_null = row[f"{col}__non_null"]
        if non_null == 0:
            ratios[col] = 0.0
        else:
            ratios[col] = row[f"{col}__n_unique"] / non_null

    return ratios


def _infer_single(
    col_name: str,
    dtype: pl.DataType,
    numeric_cols: set[str],
    unique_ratios: dict[str, float],
    config: ProfileConfig,
) -> ColumnType:
    dtype_class = type(dtype)

    if dtype_class == pl.Boolean:
        return ColumnType.BOOLEAN

    if dtype_class in _TEMPORAL_BASES:
        return ColumnType.DATETIME

    if col_name in numeric_cols:
        return ColumnType.NUMERIC

    if dtype_class == pl.Categorical:
        return ColumnType.CATEGORICAL

    if dtype_class in (pl.String, pl.Utf8):
        ratio = unique_ratios.get(col_name, 0.0)
        if ratio > config.text_unique_ratio:
            return ColumnType.TEXT
        return ColumnType.CATEGORICAL

    return ColumnType.UNSUPPORTED
