from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import TYPE_CHECKING

from dataxid_profiling._analyzers import (
    BooleanStats,
    CategoricalStats,
    NumericStats,
)
from dataxid_profiling._dataset_overview import DatasetOverview  # noqa: TC001 — used at runtime

if TYPE_CHECKING:
    from dataxid_profiling._analyzers import ColumnStats
    from dataxid_profiling._config import ProfileConfig


class AlertType(Enum):
    HIGH_MISSING = auto()
    CONSTANT = auto()
    HIGH_CARDINALITY = auto()
    DUPLICATES = auto()
    HIGH_ZEROS = auto()
    SKEWED = auto()
    IMBALANCED = auto()


@dataclass(frozen=True)
class Alert:
    column: str | None
    alert_type: AlertType
    value: float


def check_quality(
    column_stats: dict[str, ColumnStats],
    overview: DatasetOverview,
    config: ProfileConfig,
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

    return alerts


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
