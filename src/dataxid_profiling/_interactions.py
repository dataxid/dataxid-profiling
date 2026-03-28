"""Interaction data for scatter and box plots — Polars-native.

Scatter: numeric × numeric raw data (columnar).
Box plot: categorical × numeric five-number summary + outliers via IQR × 1.5.

Sampling at config.interaction_sample_size threshold; seed for reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import polars as pl

from dataxid_profiling._type_inference import ColumnType

if TYPE_CHECKING:
    from dataxid_profiling._config import ProfileConfig


@dataclass(frozen=True, slots=True)
class BoxPlotGroup:
    """Five-number summary + outliers for one category."""

    category: str
    min: float
    q1: float
    median: float
    q3: float
    max: float
    outliers: list[float] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class InteractionData:
    """Pre-computed data for the Interactions report section."""

    numeric_columns: list[str]
    categorical_columns: list[str]
    numeric_data: dict[str, list[float]]
    boxplot_stats: dict[str, dict[str, list[BoxPlotGroup]]]
    sampled: bool
    total_rows: int
    sample_size: int


def compute_interactions(
    df: pl.DataFrame,
    column_types: dict[str, ColumnType],
    config: ProfileConfig | None = None,
) -> InteractionData | None:
    """Compute interaction data for scatter and box plots.

    Returns None when there is nothing useful to show (overview mode,
    too few rows, or insufficient column variety).
    """
    if config is not None and config.is_overview:
        return None

    if df.height < 2:
        return None

    numeric_cols = _filter_numeric(df, column_types)
    categorical_cols = _filter_categorical(df, column_types, config)

    has_scatter = len(numeric_cols) >= 2
    has_boxplot = len(numeric_cols) >= 1 and len(categorical_cols) >= 1

    if not has_scatter and not has_boxplot:
        return None

    total_rows = df.height
    sample_limit = config.interaction_sample_size if config else 100_000

    sampled = total_rows > sample_limit
    if sampled:
        seed = config.interaction_sample_seed if config else 42
        df_work = df.sample(n=sample_limit, seed=seed)
    else:
        df_work = df

    numeric_data = _extract_numeric_data(df_work, numeric_cols)
    boxplot_stats = _compute_boxplot_stats(df_work, categorical_cols, numeric_cols)

    return InteractionData(
        numeric_columns=numeric_cols,
        categorical_columns=categorical_cols,
        numeric_data=numeric_data,
        boxplot_stats=boxplot_stats,
        sampled=sampled,
        total_rows=total_rows,
        sample_size=df_work.height,
    )


# -- Column filtering --------------------------------------------------------


def _filter_numeric(
    df: pl.DataFrame,
    column_types: dict[str, ColumnType],
) -> list[str]:
    """Keep numeric columns that are non-constant and have enough non-null values."""
    result: list[str] = []
    for col, ct in sorted(column_types.items()):
        if ct is not ColumnType.NUMERIC:
            continue
        series = df[col].drop_nulls()
        if series.len() < 2 or series.n_unique() < 2:
            continue
        result.append(col)
    return result


def _filter_categorical(
    df: pl.DataFrame,
    column_types: dict[str, ColumnType],
    config: ProfileConfig | None,
) -> list[str]:
    """Keep categorical columns under the cardinality limit."""
    limit = config.interaction_cardinality_limit if config else 50
    result: list[str] = []
    for col, ct in sorted(column_types.items()):
        if ct is not ColumnType.CATEGORICAL:
            continue
        if df[col].drop_nulls().n_unique() > limit:
            continue
        result.append(col)
    return result


# -- Numeric data extraction --------------------------------------------------


def _extract_numeric_data(
    df: pl.DataFrame,
    numeric_cols: list[str],
) -> dict[str, list[float]]:
    """Extract numeric column values as lists, rounded to 4 decimals."""
    data: dict[str, list[float]] = {}
    for col in numeric_cols:
        values = df[col].drop_nulls().cast(pl.Float64)
        data[col] = [round(v, 4) for v in values.to_list()]
    return data


# -- Box plot computation -----------------------------------------------------


def _compute_boxplot_stats(
    df: pl.DataFrame,
    categorical_cols: list[str],
    numeric_cols: list[str],
) -> dict[str, dict[str, list[BoxPlotGroup]]]:
    """Compute five-number summary + outliers for each categorical × numeric pair."""
    if not categorical_cols or not numeric_cols:
        return {}

    stats: dict[str, dict[str, list[BoxPlotGroup]]] = {}
    for cat_col in categorical_cols:
        stats[cat_col] = {}
        for num_col in numeric_cols:
            groups = _boxplot_for_pair(df, cat_col, num_col)
            if groups:
                stats[cat_col][num_col] = groups

    return stats


def _boxplot_for_pair(
    df: pl.DataFrame,
    cat_col: str,
    num_col: str,
) -> list[BoxPlotGroup]:
    """Compute BoxPlotGroup for each category in cat_col vs num_col."""
    sub = df.select(cat_col, num_col).drop_nulls()
    if sub.height < 2:
        return []

    summary = sub.group_by(cat_col).agg(
        pl.col(num_col).quantile(0.0).alias("min"),
        pl.col(num_col).quantile(0.25).alias("q1"),
        pl.col(num_col).quantile(0.5).alias("median"),
        pl.col(num_col).quantile(0.75).alias("q3"),
        pl.col(num_col).quantile(1.0).alias("max"),
    ).sort(cat_col)

    groups: list[BoxPlotGroup] = []
    for row in summary.iter_rows(named=True):
        q1 = float(row["q1"])
        q3 = float(row["q3"])
        iqr = q3 - q1
        lower_fence = q1 - 1.5 * iqr
        upper_fence = q3 + 1.5 * iqr

        cat_value = str(row[cat_col])
        cat_data = sub.filter(pl.col(cat_col) == row[cat_col])[num_col]
        outliers = (
            cat_data.filter(
                (cat_data < lower_fence) | (cat_data > upper_fence)
            )
            .cast(pl.Float64)
            .to_list()
        )

        actual_min = float(row["min"])
        actual_max = float(row["max"])
        lower_whisker = max(actual_min, lower_fence)
        upper_whisker = min(actual_max, upper_fence)

        groups.append(
            BoxPlotGroup(
                category=cat_value,
                min=round(lower_whisker, 2),
                q1=round(q1, 2),
                median=round(float(row["median"]), 2),
                q3=round(q3, 2),
                max=round(upper_whisker, 2),
                outliers=[round(float(v), 2) for v in outliers],
            )
        )

    return groups
