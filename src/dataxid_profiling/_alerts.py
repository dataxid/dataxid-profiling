from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any

import polars as pl
import polars_statistics as ps

from dataxid_profiling._analyzers import (
    BooleanStats,
    CategoricalStats,
    NumericStats,
)
from dataxid_profiling._dataset_overview import DatasetOverview  # noqa: TC001 — used at runtime

if TYPE_CHECKING:
    from dataxid_profiling._analyzers import ColumnStats
    from dataxid_profiling._config import ProfileConfig
    from dataxid_profiling._correlations import CorrelationResult


class AlertType(Enum):
    HIGH_MISSING = auto()
    CONSTANT = auto()
    HIGH_CARDINALITY = auto()
    DUPLICATES = auto()
    HIGH_ZEROS = auto()
    SKEWED = auto()
    IMBALANCED = auto()
    HIGH_CORRELATION = auto()
    UNIFORM = auto()


@dataclass(frozen=True)
class Alert:
    column: str | None
    alert_type: AlertType
    value: float
    details: dict[str, Any] = field(default_factory=dict)


def check_quality(
    column_stats: dict[str, ColumnStats],
    overview: DatasetOverview,
    config: ProfileConfig,
    correlations: dict[str, CorrelationResult] | None = None,
) -> list[Alert]:
    alerts: list[Alert] = []

    if overview.duplicate_rows_pct > config.duplicate_threshold:
        alerts.append(Alert(
            column=None,
            alert_type=AlertType.DUPLICATES,
            value=overview.duplicate_rows_pct,
        ))

    for col_name, stats in column_stats.items():
        alerts.extend(_check_column(col_name, stats, config))

    if correlations:
        alerts.extend(_check_correlations(correlations, config))

    return alerts


def _check_column(
    col_name: str,
    stats: ColumnStats,
    config: ProfileConfig,
) -> list[Alert]:
    alerts: list[Alert] = []

    if stats.missing_pct > config.missing_threshold:
        alerts.append(Alert(col_name, AlertType.HIGH_MISSING, stats.missing_pct))

    if (
        hasattr(stats, "distinct_count")
        and stats.distinct_count <= config.constant_threshold
        and stats.count > 0
    ):
        alerts.append(Alert(col_name, AlertType.CONSTANT, float(stats.distinct_count)))

    if isinstance(stats, NumericStats):
        alerts.extend(_check_numeric(col_name, stats, config))

    if isinstance(stats, CategoricalStats):
        alerts.extend(_check_categorical(col_name, stats, config))

    if isinstance(stats, BooleanStats):
        alerts.extend(_check_boolean(col_name, stats, config))

    return alerts


def _check_numeric(
    col_name: str,
    stats: NumericStats,
    config: ProfileConfig,
) -> list[Alert]:
    alerts: list[Alert] = []

    if stats.distinct_pct > config.cardinality_threshold:
        alerts.append(Alert(col_name, AlertType.HIGH_CARDINALITY, stats.distinct_pct))

    if stats.zeros_pct > config.zero_threshold:
        alerts.append(Alert(col_name, AlertType.HIGH_ZEROS, stats.zeros_pct))

    if stats.skewness is not None and abs(stats.skewness) > config.skewness_threshold:
        alerts.append(Alert(col_name, AlertType.SKEWED, abs(stats.skewness)))

    return alerts


def _check_categorical(
    col_name: str,
    stats: CategoricalStats,
    config: ProfileConfig,
) -> list[Alert]:
    alerts: list[Alert] = []

    if stats.distinct_pct > config.cardinality_threshold:
        alerts.append(Alert(col_name, AlertType.HIGH_CARDINALITY, stats.distinct_pct))

    if stats.top_values and stats.count > 0:
        non_null = stats.count - stats.missing_count
        if non_null > 0:
            top_pct = stats.top_values[0]["count"] / non_null
            if top_pct > config.imbalance_threshold:
                alerts.append(Alert(col_name, AlertType.IMBALANCED, top_pct))

    if stats.top_values and len(stats.top_values) >= 2:
        uniform_alert = _check_uniform(col_name, stats, config)
        if uniform_alert is not None:
            alerts.append(uniform_alert)

    return alerts


def _check_uniform(
    col_name: str,
    stats: CategoricalStats,
    config: ProfileConfig,
) -> Alert | None:
    """Chi-squared goodness-of-fit test against uniform distribution (Rust)."""
    counts = [v["count"] for v in stats.top_values if v["count"] > 0]
    if len(counts) < 2:
        return None

    df = pl.DataFrame({"observed": counts})
    try:
        result = df.select(ps.chisq_goodness_of_fit("observed"))
    except pl.exceptions.ComputeError:
        return None

    row = result.unnest(result.columns[0]).row(0, named=True)
    p_value = float(row["p_value"]) if row["p_value"] is not None else 0.0

    if p_value > config.uniform_pvalue_threshold:
        return Alert(
            column=col_name,
            alert_type=AlertType.UNIFORM,
            value=p_value,
            details={"p_value": p_value, "test": "chi2_gof"},
        )
    return None


def _check_boolean(
    col_name: str,
    stats: BooleanStats,
    config: ProfileConfig,
) -> list[Alert]:
    alerts: list[Alert] = []

    dominant_pct = max(stats.true_pct, stats.false_pct)
    if dominant_pct > config.imbalance_threshold:
        alerts.append(Alert(col_name, AlertType.IMBALANCED, dominant_pct))

    return alerts


def _check_correlations(
    correlations: dict[str, CorrelationResult],
    config: ProfileConfig,
) -> list[Alert]:
    """Flag column pairs with |correlation| above threshold.

    Prefers Phi K (universal, covers all column types).
    Falls back to Pearson if Phi K is absent.
    """
    matrix_key = "phik" if "phik" in correlations else "pearson"
    if matrix_key not in correlations:
        return []

    matrix = correlations[matrix_key].matrix
    columns = [c for c in matrix.columns if c != "column"]
    seen: set[tuple[str, str]] = set()
    alerts: list[Alert] = []

    for row in matrix.iter_rows(named=True):
        col_a = str(row["column"])
        for col_b in columns:
            if col_a == col_b:
                continue
            pair = tuple(sorted((col_a, col_b)))
            if pair in seen:
                continue
            seen.add(pair)

            val = row[col_b]
            if val is None:
                continue
            corr = abs(float(val))
            if corr > config.correlation_threshold:
                alerts.append(Alert(
                    column=col_a,
                    alert_type=AlertType.HIGH_CORRELATION,
                    value=corr,
                    details={"column_b": col_b, "method": matrix_key},
                ))

    return alerts
