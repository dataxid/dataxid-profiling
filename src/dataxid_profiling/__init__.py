"""Fast, Polars-native data profiling."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from dataxid_profiling._config import ProfileConfig
from dataxid_profiling._ingest import ingest
from dataxid_profiling._type_inference import ColumnType, infer_types

if TYPE_CHECKING:
    import polars as pl

__version__ = "0.1.0"

__all__ = ["ProfileReport", "ProfileConfig", "ColumnType"]


class ProfileReport:
    """Main entry point for data profiling.

    Usage:
        report = ProfileReport(df)
        report = ProfileReport(df, title="My Data")
        report = ProfileReport("data.csv", minimal=True)
    """

    def __init__(
        self,
        source: Any,
        *,
        title: str = "Dataset Report",
        minimal: bool = False,
        config: ProfileConfig | None = None,
        **kwargs: Any,
    ) -> None:
        if config is not None:
            self._config = config
        else:
            self._config = ProfileConfig(title=title, minimal=minimal, **kwargs)

        self._df: pl.DataFrame = ingest(source)
        self._column_types: dict[str, ColumnType] = infer_types(self._df, self._config)

    @property
    def config(self) -> ProfileConfig:
        return self._config

    @property
    def df(self) -> pl.DataFrame:
        return self._df

    @property
    def column_types(self) -> dict[str, ColumnType]:
        return self._column_types

    def __repr__(self) -> str:
        return (
            f"ProfileReport("
            f"title='{self._config.title}', "
            f"rows={self._df.height}, "
            f"columns={self._df.width})"
        )
