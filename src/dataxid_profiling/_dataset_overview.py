from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING, Any

import polars as pl

from dataxid_profiling._type_inference import ColumnType  # noqa: TC001 — used at runtime

if TYPE_CHECKING:
    from dataxid_profiling._config import ProfileConfig

_SAMPLE_SIZE = 10


@dataclass(frozen=True)
class DatasetOverview:
    n_rows: int
    n_columns: int
    missing_cells: int
    missing_cells_pct: float
    duplicate_rows: int
    duplicate_rows_pct: float
    memory_bytes: int
    memory_human: str
    type_distribution: dict[str, int] = field(default_factory=dict)
    missing_per_column: dict[str, dict[str, Any]] = field(default_factory=dict)
    sample_head: list[dict[str, Any]] = field(default_factory=list)
    sample_tail: list[dict[str, Any]] = field(default_factory=list)
    duplicate_rows_sample: list[dict[str, Any]] = field(default_factory=list)
    reproduction: dict[str, str] = field(default_factory=dict)


def compute_overview(
    df: pl.DataFrame,
    column_types: dict[str, ColumnType],
    config: ProfileConfig | None = None,
) -> DatasetOverview:
    from dataxid_profiling._config import ProfileConfig as _PC

    config = config or _PC()
    n_rows = df.height
    n_columns = df.width
    total_cells = n_rows * n_columns

    missing_cells = _total_missing(df)
    duplicate_rows = _duplicate_count(df, n_rows)
    memory_bytes = df.estimated_size()

    type_dist: dict[str, int] = {}
    for ct in column_types.values():
        type_dist[ct.name] = type_dist.get(ct.name, 0) + 1

    missing_per_col = _missing_per_column(df, n_rows)
    sample_head = _sample_rows(df.head(_SAMPLE_SIZE))
    sample_tail = _sample_rows(df.tail(_SAMPLE_SIZE))

    dup_sample: list[dict[str, Any]] = []
    if not config.is_overview and duplicate_rows > 0:
        dup_sample = _duplicate_rows_sample(df)

    reproduction = _reproduction_info(config)

    return DatasetOverview(
        n_rows=n_rows,
        n_columns=n_columns,
        missing_cells=missing_cells,
        missing_cells_pct=missing_cells / total_cells if total_cells > 0 else 0.0,
        duplicate_rows=duplicate_rows,
        duplicate_rows_pct=duplicate_rows / n_rows if n_rows > 0 else 0.0,
        memory_bytes=memory_bytes,
        memory_human=_format_bytes(memory_bytes),
        type_distribution=type_dist,
        missing_per_column=missing_per_col,
        sample_head=sample_head,
        sample_tail=sample_tail,
        duplicate_rows_sample=dup_sample,
        reproduction=reproduction,
    )


def _total_missing(df: pl.DataFrame) -> int:
    if df.height == 0 or df.width == 0:
        return 0
    return sum(df.select(pl.all().null_count()).row(0))


def _missing_per_column(
    df: pl.DataFrame, n_rows: int
) -> dict[str, dict[str, Any]]:
    if df.height == 0 or df.width == 0:
        return {}
    null_counts = df.select(pl.all().null_count()).row(0, named=True)
    result: dict[str, dict[str, Any]] = {}
    for col_name, null_count in null_counts.items():
        result[col_name] = {
            "count": null_count,
            "pct": null_count / n_rows if n_rows > 0 else 0.0,
        }
    return result


def _duplicate_count(df: pl.DataFrame, n_rows: int) -> int:
    if n_rows == 0:
        return 0
    n_unique = df.select(pl.struct(pl.all()).n_unique()).item()
    return n_rows - n_unique


def _duplicate_rows_sample(
    df: pl.DataFrame, max_rows: int = _SAMPLE_SIZE
) -> list[dict[str, Any]]:
    try:
        dupes = df.filter(df.is_duplicated()).head(max_rows)
        return _sample_rows(dupes)
    except Exception:
        return []


def _sample_rows(df: pl.DataFrame) -> list[dict[str, Any]]:
    if df.height == 0:
        return []
    return [
        {col: str(val) if val is not None else None for col, val in row.items()}
        for row in df.iter_rows(named=True)
    ]


def _reproduction_info(config: ProfileConfig) -> dict[str, str]:
    import dataxid_profiling

    return {
        "dataxid_profiling_version": dataxid_profiling.__version__,
        "polars_version": pl.__version__,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "mode": config.mode,
    }


def _format_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024  # type: ignore[assignment]
    return f"{n:.1f} TB"
