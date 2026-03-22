"""Fast, Polars-native data profiling."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import TYPE_CHECKING, Any

import polars as pl  # noqa: TC002 — used at runtime

from dataxid_profiling._alerts import Alert, AlertType, check_quality
from dataxid_profiling._analyzers import ColumnStats, analyze
from dataxid_profiling._config import ProfileConfig
from dataxid_profiling._correlations import compute_correlations
from dataxid_profiling._dataset_overview import DatasetOverview, compute_overview
from dataxid_profiling._ingest import ingest
from dataxid_profiling._report._html import render_html
from dataxid_profiling._type_inference import ColumnType, infer_types

if TYPE_CHECKING:
    from pathlib import Path

__version__ = "0.1.0"

__all__ = ["ProfileReport", "ProfileConfig", "ColumnType", "AlertType", "Alert"]


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
        self._column_stats: dict[str, ColumnStats] = analyze(
            self._df, self._column_types, self._config
        )
        self._overview: DatasetOverview = compute_overview(
            self._df, self._column_types, self._config
        )
        self._alerts: list[Alert] = check_quality(
            self._column_stats, self._overview, self._config
        )
        self._correlations: dict[str, pl.DataFrame] = compute_correlations(
            self._df, self._column_types, self._config
        )

    @property
    def config(self) -> ProfileConfig:
        return self._config

    @property
    def df(self) -> pl.DataFrame:
        return self._df

    @property
    def column_types(self) -> dict[str, ColumnType]:
        return self._column_types

    @property
    def overview(self) -> dict[str, Any]:
        return asdict(self._overview)

    @property
    def alerts(self) -> list[Alert]:
        return self._alerts

    @property
    def stats(self) -> dict[str, dict[str, Any]]:
        return {name: asdict(s) for name, s in self._column_stats.items()}

    @property
    def correlations(self) -> dict[str, pl.DataFrame]:
        return self._correlations

    def to_dict(self) -> dict[str, Any]:
        corr_dict = {}
        for method, matrix in self._correlations.items():
            corr_dict[method] = matrix.to_dicts()

        return {
            "title": self._config.title,
            "overview": self.overview,
            "columns": self.stats,
            "alerts": [
                {
                    "column": a.column,
                    "alert_type": a.alert_type.name,
                    "value": a.value,
                }
                for a in self._alerts
            ],
            "correlations": corr_dict,
        }

    def to_html(self, path: str | Path | None = None) -> str:
        html = render_html(
            title=self._config.title,
            version=__version__,
            overview=self._overview,
            column_stats=self._column_stats,
            alerts=self._alerts,
            correlations=self._correlations,
        )
        if path is not None:
            from pathlib import Path as P

            P(path).write_text(html, encoding="utf-8")

        return html

    def to_json(self, path: str | Path | None = None, indent: int = 2) -> str:
        data = self.to_dict()
        # ColumnType enum → string for JSON serialization
        for col_data in data["columns"].values():
            if "column_type" in col_data:
                col_data["column_type"] = col_data["column_type"].name

        output = json.dumps(data, indent=indent, default=str, ensure_ascii=False)

        if path is not None:
            from pathlib import Path as P

            P(path).write_text(output, encoding="utf-8")

        return output

    def __repr__(self) -> str:
        return (
            f"ProfileReport("
            f"title='{self._config.title}', "
            f"rows={self._df.height}, "
            f"columns={self._df.width})"
        )
