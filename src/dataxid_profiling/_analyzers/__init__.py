from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from dataxid_profiling._config import ProfileConfig
from dataxid_profiling._type_inference import ColumnType

if TYPE_CHECKING:
    import polars as pl


# ---------------------------------------------------------------------------
# Base stats — shared across all column types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class BaseStats:
    column_name: str
    column_type: ColumnType
    count: int
    missing_count: int
    missing_pct: float


# ---------------------------------------------------------------------------
# Type-specific stats
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NumericStats(BaseStats):
    distinct_count: int = 0
    distinct_pct: float = 0.0
    mean: float | None = None
    std: float | None = None
    min: float | None = None
    max: float | None = None
    range: float | None = None
    q25: float | None = None
    median: float | None = None
    q75: float | None = None
    iqr: float | None = None
    skewness: float | None = None
    kurtosis: float | None = None
    zeros_count: int = 0
    zeros_pct: float = 0.0
    negative_count: int = 0
    negative_pct: float = 0.0
    histogram: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class CategoricalStats(BaseStats):
    distinct_count: int = 0
    distinct_pct: float = 0.0
    top_values: list[dict[str, Any]] = field(default_factory=list)


@dataclass(frozen=True)
class BooleanStats(BaseStats):
    true_count: int = 0
    true_pct: float = 0.0
    false_count: int = 0
    false_pct: float = 0.0


@dataclass(frozen=True)
class DatetimeStats(BaseStats):
    distinct_count: int = 0
    distinct_pct: float = 0.0
    min: str | None = None
    max: str | None = None
    range: str | None = None


@dataclass(frozen=True)
class TextStats(BaseStats):
    distinct_count: int = 0
    distinct_pct: float = 0.0
    length_mean: float | None = None
    length_min: int | None = None
    length_max: int | None = None


ColumnStats = NumericStats | CategoricalStats | BooleanStats | DatetimeStats | TextStats


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def analyze(
    df: pl.DataFrame,
    column_types: dict[str, ColumnType],
    config: ProfileConfig | None = None,
) -> dict[str, ColumnStats]:
    """Run type-specific analyzers for each column."""
    from dataxid_profiling._analyzers._boolean import analyze_boolean
    from dataxid_profiling._analyzers._categorical import analyze_categorical
    from dataxid_profiling._analyzers._datetime import analyze_datetime
    from dataxid_profiling._analyzers._numeric import analyze_numeric
    from dataxid_profiling._analyzers._text import analyze_text

    if config is None:
        config = ProfileConfig()

    dispatchers: dict[ColumnType, Any] = {
        ColumnType.NUMERIC: analyze_numeric,
        ColumnType.CATEGORICAL: analyze_categorical,
        ColumnType.BOOLEAN: analyze_boolean,
        ColumnType.DATETIME: analyze_datetime,
        ColumnType.TEXT: analyze_text,
    }

    results: dict[str, ColumnStats] = {}
    for col_name, col_type in column_types.items():
        analyzer = dispatchers.get(col_type)
        if analyzer is not None:
            results[col_name] = analyzer(df, col_name, config)

    return results
