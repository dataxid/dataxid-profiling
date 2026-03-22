from __future__ import annotations

import polars as pl
import pytest

from dataxid_profiling._analyzers._categorical import analyze_categorical
from dataxid_profiling._config import ProfileConfig
from dataxid_profiling._type_inference import ColumnType


@pytest.fixture
def config() -> ProfileConfig:
    return ProfileConfig()


class TestCategoricalBasicStats:
    def test_count_and_missing(self, categorical_df: pl.DataFrame, config: ProfileConfig):
        stats = analyze_categorical(categorical_df, "city", config)
        assert stats.count == 10
        assert stats.missing_count == 1
        assert stats.missing_pct == pytest.approx(0.1)

    def test_column_metadata(self, categorical_df: pl.DataFrame, config: ProfileConfig):
        stats = analyze_categorical(categorical_df, "city", config)
        assert stats.column_name == "city"
        assert stats.column_type == ColumnType.CATEGORICAL

    def test_distinct(self, categorical_df: pl.DataFrame, config: ProfileConfig):
        stats = analyze_categorical(categorical_df, "city", config)
        assert stats.distinct_count == 3  # Istanbul, Ankara, Izmir
        assert stats.distinct_pct == pytest.approx(3 / 10)


class TestCategoricalTopValues:
    def test_top_values_order(self, categorical_df: pl.DataFrame, config: ProfileConfig):
        stats = analyze_categorical(categorical_df, "city", config)
        assert len(stats.top_values) > 0
        counts = [v["count"] for v in stats.top_values]
        assert counts == sorted(counts, reverse=True)

    def test_top_values_content(self, categorical_df: pl.DataFrame, config: ProfileConfig):
        stats = analyze_categorical(categorical_df, "city", config)
        values = {v["value"] for v in stats.top_values}
        assert "Istanbul" in values

    def test_top_values_excludes_null(self, categorical_df: pl.DataFrame, config: ProfileConfig):
        stats = analyze_categorical(categorical_df, "city", config)
        values = [v["value"] for v in stats.top_values]
        assert None not in values

    def test_custom_n_top(self):
        df = pl.DataFrame({"cat": ["a", "b", "c", "d", "e", "a", "b", "c", "a", "b"]})
        config = ProfileConfig(n_top_values=2)
        stats = analyze_categorical(df, "cat", config)
        assert len(stats.top_values) == 2

    def test_top_values_has_value_and_count(
        self, categorical_df: pl.DataFrame, config: ProfileConfig
    ):
        stats = analyze_categorical(categorical_df, "city", config)
        for entry in stats.top_values:
            assert "value" in entry
            assert "count" in entry
            assert isinstance(entry["count"], int)


class TestCategoricalEdgeCases:
    def test_all_null(self, config: ProfileConfig):
        df = pl.DataFrame({"cat": pl.Series([None, None, None], dtype=pl.Utf8)})
        stats = analyze_categorical(df, "cat", config)
        assert stats.count == 3
        assert stats.missing_count == 3
        assert stats.distinct_count == 0
        assert stats.top_values == []

    def test_single_value(self, config: ProfileConfig):
        df = pl.DataFrame({"cat": ["only"] * 5})
        stats = analyze_categorical(df, "cat", config)
        assert stats.distinct_count == 1
        assert stats.top_values[0]["value"] == "only"
        assert stats.top_values[0]["count"] == 5

    def test_empty_dataframe(self, config: ProfileConfig):
        df = pl.DataFrame({"cat": pl.Series([], dtype=pl.Utf8)})
        stats = analyze_categorical(df, "cat", config)
        assert stats.count == 0
        assert stats.missing_count == 0

    def test_constant_column(self, constant_df: pl.DataFrame, config: ProfileConfig):
        stats = analyze_categorical(constant_df, "status", config)
        assert stats.distinct_count == 1
        assert stats.top_values[0]["value"] == "active"
