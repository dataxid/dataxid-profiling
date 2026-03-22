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

    return {
        col_name: _infer_single(df, col_name, dtype, n_rows, config, numeric_cols)
        for col_name, dtype in zip(df.columns, df.dtypes, strict=True)
    }


def _infer_single(
    df: pl.DataFrame,
    col_name: str,
    dtype: pl.DataType,
    n_rows: int,
    config: ProfileConfig,
    numeric_cols: set[str],
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
        return _classify_string(df, col_name, n_rows, config)

    return ColumnType.UNSUPPORTED


def _classify_string(
    df: pl.DataFrame,
    col_name: str,
    n_rows: int,
    config: ProfileConfig,
) -> ColumnType:
    if n_rows == 0:
        return ColumnType.CATEGORICAL

    # null'ları hariç tut — unique ratio sadece gerçek değerler üzerinden
    non_null_count = n_rows - df.select(pl.col(col_name).null_count()).item()
    if non_null_count == 0:
        return ColumnType.CATEGORICAL

    n_unique = df.select(pl.col(col_name).drop_nulls().n_unique()).item()
    unique_ratio = n_unique / non_null_count

    if unique_ratio > config.text_unique_ratio:
        return ColumnType.TEXT

    return ColumnType.CATEGORICAL
