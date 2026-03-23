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
        col.str.len_chars().min().alias("length_min"),
        col.str.len_chars().max().alias("length_max"),
        col.str.len_chars().mean().alias("length_mean"),
        col.str.len_chars().median().alias("length_median"),
        col.str.split(" ").list.len().mean().alias("word_count_mean"),
        col.str.contains(r"[^\x00-\x7F]").any().alias("has_non_ascii"),
    ).row(0, named=True)

    missing_count: int = row["missing_count"]
    distinct_count: int = row["distinct_count"]

    top_values = _compute_top_values(df, col_name, config.n_top_values)
    imbalance = _compute_imbalance(top_values, n_rows - missing_count)

    is_complete = not config.is_overview
    char_counts: list[dict[str, Any]] = []
    n_chars: int | None = None
    n_chars_distinct: int | None = None
    length_hist: list[dict[str, Any]] = []

    if is_complete:
        char_counts, n_chars, n_chars_distinct = _compute_character_stats(
            df, col_name
        )
        length_hist = _compute_length_histogram(df, col_name, config.histogram_bins)

    return CategoricalStats(
        column_name=col_name,
        column_type=ColumnType.CATEGORICAL,
        count=n_rows,
        missing_count=missing_count,
        missing_pct=missing_count / n_rows,
        distinct_count=distinct_count,
        distinct_pct=distinct_count / n_rows,
        imbalance=imbalance,
        length_min=row["length_min"],
        length_max=row["length_max"],
        length_mean=_safe_float(row["length_mean"]),
        length_median=_safe_float(row["length_median"]),
        word_count_mean=_safe_float(row["word_count_mean"]),
        n_characters=n_chars,
        n_characters_distinct=n_chars_distinct,
        has_non_ascii=bool(row["has_non_ascii"]),
        top_values=top_values,
        character_counts=char_counts,
        length_histogram=length_hist,
    )


def _compute_imbalance(
    top_values: list[dict[str, Any]], non_null_count: int
) -> float | None:
    """Imbalance = top value frequency. 0 = perfectly balanced, 1 = single value."""
    if non_null_count == 0 or not top_values:
        return None
    return top_values[0]["count"] / non_null_count


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


def _compute_character_stats(
    df: pl.DataFrame, col_name: str
) -> tuple[list[dict[str, Any]], int | None, int | None]:
    """Character frequency analysis. Returns (top_chars, total_chars, distinct_chars)."""
    try:
        exploded = (
            df.select(pl.col(col_name))
            .drop_nulls()
            .select(pl.col(col_name).str.split("").explode().alias("char"))
            .filter(pl.col("char") != "")
        )
        if exploded.height == 0:
            return [], None, None

        n_chars = exploded.height
        n_distinct = exploded.select(pl.col("char").n_unique()).item()

        top = (
            exploded.group_by("char")
            .len()
            .sort("len", descending=True)
            .head(20)
        )
        char_counts = [
            {"character": row["char"], "count": row["len"]}
            for row in top.iter_rows(named=True)
        ]
        return char_counts, n_chars, n_distinct
    except Exception:
        return [], None, None


def _compute_length_histogram(
    df: pl.DataFrame, col_name: str, bin_count: int
) -> list[dict[str, Any]]:
    try:
        hist_df = df.select(
            pl.col(col_name).str.len_chars().alias("_len")
        ).drop_nulls().select(
            pl.col("_len").hist(bin_count=bin_count, include_breakpoint=True)
        ).unnest("_len")

        return [
            {"breakpoint": row["breakpoint"], "count": row["count"]}
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


def _empty_stats(col_name: str) -> CategoricalStats:
    return CategoricalStats(
        column_name=col_name,
        column_type=ColumnType.CATEGORICAL,
        count=0,
        missing_count=0,
        missing_pct=0.0,
    )
