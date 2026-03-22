from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import polars as pl

from dataxid_profiling._type_inference import ColumnType  # noqa: TC001 — used at runtime

if TYPE_CHECKING:
    from dataxid_profiling._config import ProfileConfig


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


def compute_overview(
    df: pl.DataFrame,
    column_types: dict[str, ColumnType],
    config: ProfileConfig | None = None,
) -> DatasetOverview:
    n_rows = df.height
    n_columns = df.width
    total_cells = n_rows * n_columns

    missing_cells = _total_missing(df)
    duplicate_rows = _duplicate_count(df, n_rows)
    memory_bytes = df.estimated_size()

    type_dist: dict[str, int] = {}
    for ct in column_types.values():
        type_dist[ct.name] = type_dist.get(ct.name, 0) + 1

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
    )


def _total_missing(df: pl.DataFrame) -> int:
    if df.height == 0 or df.width == 0:
        return 0
    return sum(df.select(pl.all().null_count()).row(0))


def _duplicate_count(df: pl.DataFrame, n_rows: int) -> int:
    if n_rows == 0:
        return 0
    n_unique = df.select(pl.struct(pl.all()).n_unique()).item()
    return n_rows - n_unique


def _format_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}" if unit != "B" else f"{n} B"
        n /= 1024  # type: ignore[assignment]
    return f"{n:.1f} TB"
