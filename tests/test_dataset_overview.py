from __future__ import annotations

import polars as pl
import pytest

from dataxid_profiling._config import ProfileConfig
from dataxid_profiling._dataset_overview import DatasetOverview, compute_overview
from dataxid_profiling._type_inference import infer_types


@pytest.fixture
def config() -> ProfileConfig:
    return ProfileConfig()


def _overview(df: pl.DataFrame, config: ProfileConfig | None = None) -> DatasetOverview:
    config = config or ProfileConfig()
    column_types = infer_types(df, config)
    return compute_overview(df, column_types, config)


class TestOverviewBasic:
    def test_row_column_count(self, mixed_df: pl.DataFrame):
        ov = _overview(mixed_df)
        assert ov.n_rows == 10
        assert ov.n_columns == 7

    def test_missing_cells(self, mixed_df: pl.DataFrame):
        ov = _overview(mixed_df)
        assert ov.missing_cells > 0
        assert 0.0 < ov.missing_cells_pct < 1.0

    def test_missing_cells_exact(self):
        df = pl.DataFrame({"a": [1, None, 3], "b": [None, None, "x"]})
        ov = _overview(df)
        assert ov.missing_cells == 3
        assert ov.missing_cells_pct == pytest.approx(3 / 6)


class TestOverviewDuplicates:
    def test_no_duplicates(self, mixed_df: pl.DataFrame):
        ov = _overview(mixed_df)
        assert ov.duplicate_rows == 0

    def test_with_duplicates(self):
        df = pl.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
        ov = _overview(df)
        assert ov.duplicate_rows == 1
        assert ov.duplicate_rows_pct == pytest.approx(1 / 3)

    def test_all_duplicates(self):
        df = pl.DataFrame({"a": [1, 1, 1], "b": ["x", "x", "x"]})
        ov = _overview(df)
        assert ov.duplicate_rows == 2


class TestOverviewMemory:
    def test_memory_positive(self, mixed_df: pl.DataFrame):
        ov = _overview(mixed_df)
        assert ov.memory_bytes > 0

    def test_memory_human_readable(self, mixed_df: pl.DataFrame):
        ov = _overview(mixed_df)
        assert any(unit in ov.memory_human for unit in ("B", "KB", "MB", "GB"))


class TestOverviewTypeDistribution:
    def test_type_distribution(self, mixed_df: pl.DataFrame):
        ov = _overview(mixed_df)
        assert "NUMERIC" in ov.type_distribution
        assert "BOOLEAN" in ov.type_distribution
        assert "DATETIME" in ov.type_distribution
        assert sum(ov.type_distribution.values()) == 7

    def test_all_numeric(self, numeric_df: pl.DataFrame):
        ov = _overview(numeric_df)
        assert ov.type_distribution.get("NUMERIC", 0) == 3


class TestOverviewEdgeCases:
    def test_empty_dataframe(self, empty_df: pl.DataFrame):
        ov = _overview(empty_df)
        assert ov.n_rows == 0
        assert ov.n_columns == 2
        assert ov.missing_cells == 0
        assert ov.duplicate_rows == 0
        assert ov.missing_cells_pct == 0.0

    def test_no_missing(self):
        df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        ov = _overview(df)
        assert ov.missing_cells == 0
        assert ov.missing_cells_pct == 0.0

    def test_single_column(self):
        df = pl.DataFrame({"a": [1, 2, None]})
        ov = _overview(df)
        assert ov.n_columns == 1
        assert ov.missing_cells == 1
