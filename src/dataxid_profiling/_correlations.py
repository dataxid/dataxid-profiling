"""Correlation matrices — Polars-native.

Pearson and Spearman use polars-statistics (Rust plugin).
Kendall tau-b uses scipy's battle-tested C merge-sort — O(n log n)
vs the O(n²) naive pairwise approach.
Cramér's V uses Polars group_by for contingency tables + ps.cramers_v (Rust).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import polars as pl
import polars_statistics as ps
from scipy.stats import kendalltau

from dataxid_profiling._type_inference import ColumnType

if TYPE_CHECKING:
    from dataxid_profiling._config import ProfileConfig

PairFn = Callable[[pl.DataFrame, str, str], tuple[float, float | None]]


@dataclass(frozen=True, slots=True)
class CorrelationResult:
    """A correlation matrix with optional p-values."""

    matrix: pl.DataFrame
    pvalues: pl.DataFrame | None = None


def compute_correlations(
    df: pl.DataFrame,
    column_types: dict[str, ColumnType],
    config: ProfileConfig | None = None,
) -> dict[str, CorrelationResult]:
    """Compute correlation matrices.

    Returns a dict keyed by method name (e.g. "pearson").
    Empty dict when fewer than 2 numeric columns or overview mode.
    """
    if config is not None and config.is_overview:
        return {}

    numeric_cols = sorted(
        col for col, ct in column_types.items() if ct is ColumnType.NUMERIC
    )
    categorical_cols = sorted(
        col for col, ct in column_types.items() if ct is ColumnType.CATEGORICAL
    )

    result: dict[str, CorrelationResult] = {}

    if len(numeric_cols) >= 2:
        df_f64 = _ensure_f64(df, numeric_cols)
        result["pearson"] = _build_matrix(df_f64, numeric_cols, _pearson_pair)
        result["spearman"] = _build_matrix(df_f64, numeric_cols, _spearman_pair)
        result["kendall"] = _build_matrix(df_f64, numeric_cols, _kendall_pair)

    if len(categorical_cols) >= 2:
        result["cramers_v"] = _build_matrix(df, categorical_cols, _cramers_v_pair)

    return result


def _build_matrix(
    df: pl.DataFrame,
    columns: list[str],
    compute_fn: PairFn,
) -> CorrelationResult:
    """Generic N×N symmetric matrix builder.

    *compute_fn(df, col_a, col_b)* → (estimate, p_value | None).
    Diagonal is always (1.0, 0.0). Upper triangle is mirrored.
    """
    n = len(columns)
    est = [[0.0] * n for _ in range(n)]
    pval: list[list[float]] | None = None

    for i in range(n):
        est[i][i] = 1.0
        for j in range(i + 1, n):
            estimate, p_value = compute_fn(df, columns[i], columns[j])
            est[i][j] = estimate
            est[j][i] = estimate
            if p_value is not None:
                if pval is None:
                    pval = [[0.0] * n for _ in range(n)]
                pval[i][j] = p_value
                pval[j][i] = p_value

    matrix = _arrays_to_df(columns, est)
    pval_df = _arrays_to_df(columns, pval) if pval else None
    return CorrelationResult(matrix=matrix, pvalues=pval_df)


def _arrays_to_df(columns: list[str], data: list[list[float]]) -> pl.DataFrame:
    rows: list[dict[str, float | str]] = []
    for i, col_name in enumerate(columns):
        row: dict[str, float | str] = {"column": col_name}
        for j, other in enumerate(columns):
            row[other] = data[i][j]
        rows.append(row)
    return pl.DataFrame(rows)


# -- Pair functions --------------------------------------------------------


def _ensure_f64(df: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    """Cast numeric columns to Float64 — polars-statistics requires it."""
    casts = [
        pl.col(c).cast(pl.Float64) for c in columns if df[c].dtype != pl.Float64
    ]
    return df.with_columns(casts) if casts else df


def _extract_corr(df: pl.DataFrame, expr: pl.Expr) -> tuple[float, float]:
    """Extract (estimate, p_value) from a polars-statistics struct result.

    Zero-variance columns make correlation mathematically undefined (0/0);
    polars-statistics panics instead of returning NaN — we catch that and
    return the correct (NaN, NaN).
    """
    try:
        result = df.select(expr)
    except pl.exceptions.ComputeError:
        return float("nan"), float("nan")
    row = result.unnest(result.columns[0]).row(0, named=True)
    est = float(row["estimate"]) if row["estimate"] is not None else 0.0
    pval = float(row["p_value"]) if row["p_value"] is not None else 1.0
    return est, pval


def _pearson_pair(
    df: pl.DataFrame, col_a: str, col_b: str
) -> tuple[float, float | None]:
    return _extract_corr(df, ps.pearson(col_a, col_b))


def _spearman_pair(
    df: pl.DataFrame, col_a: str, col_b: str
) -> tuple[float, float | None]:
    return _extract_corr(df, ps.spearman(col_a, col_b))


def _kendall_pair(
    df: pl.DataFrame, col_a: str, col_b: str
) -> tuple[float, float | None]:
    """Kendall tau-b via scipy — O(n log n) C merge-sort, not ps.kendall O(n²)."""
    valid = df.select(col_a, col_b).drop_nulls()
    if valid.height < 2:
        return 0.0, 1.0
    tau, pval = kendalltau(valid[col_a].to_numpy(), valid[col_b].to_numpy())
    return float(tau), float(pval)


def _build_contingency_flat(
    df: pl.DataFrame, col_a: str, col_b: str
) -> tuple[list[int], int, int]:
    """Build a flattened contingency table (row-major) via Polars group_by + pivot."""
    ct = (
        df.select(col_a, col_b)
        .drop_nulls()
        .group_by(col_a, col_b)
        .len()
        .pivot(on=col_b, index=col_a, values="len")
        .fill_null(0)
    )
    row_labels = ct.sort(col_a)[col_a].to_list()
    value_cols = sorted(c for c in ct.columns if c != col_a)
    nr, nc = len(row_labels), len(value_cols)
    flat: list[int] = []
    for row in ct.sort(col_a).iter_rows(named=True):
        for vc in value_cols:
            flat.append(int(row[vc]))
    return flat, nr, nc


def _cramers_v_pair(
    df: pl.DataFrame, col_a: str, col_b: str
) -> tuple[float, float | None]:
    """Cramér's V via Polars contingency table + ps.cramers_v (Rust)."""
    flat, nr, nc = _build_contingency_flat(df, col_a, col_b)
    if nr < 2 or nc < 2:
        return float("nan"), None
    ct_df = pl.DataFrame({"ct": flat})
    try:
        result = ct_df.select(ps.cramers_v("ct", n_rows=nr, n_cols=nc))
    except pl.exceptions.ComputeError:
        return float("nan"), None
    row = result.unnest(result.columns[0]).row(0, named=True)
    v = float(row["estimate"]) if row["estimate"] is not None else 0.0
    return v, None
