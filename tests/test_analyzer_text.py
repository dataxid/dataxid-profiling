from __future__ import annotations

import polars as pl
import pytest

from dataxid_profiling._analyzers._text import analyze_text
from dataxid_profiling._config import ProfileConfig
from dataxid_profiling._type_inference import ColumnType


@pytest.fixture
def config() -> ProfileConfig:
    return ProfileConfig()


class TestTextBasicStats:
    def test_count_and_missing(self, config: ProfileConfig):
        df = pl.DataFrame({"name": ["Alice", "Bob", None, "Diana", "Eve"]})
        stats = analyze_text(df, "name", config)
        assert stats.count == 5
        assert stats.missing_count == 1
        assert stats.missing_pct == pytest.approx(0.2)

    def test_column_metadata(self, config: ProfileConfig):
        df = pl.DataFrame({"name": ["Alice", "Bob"]})
        stats = analyze_text(df, "name", config)
        assert stats.column_name == "name"
        assert stats.column_type == ColumnType.TEXT

    def test_distinct(self, high_cardinality_df: pl.DataFrame, config: ProfileConfig):
        stats = analyze_text(high_cardinality_df, "email", config)
        assert stats.distinct_count == 100
        assert stats.distinct_pct == pytest.approx(1.0)


class TestTextLengthStats:
    def test_length_mean(self, config: ProfileConfig):
        df = pl.DataFrame({"name": ["ab", "abcd", "abcdef"]})
        stats = analyze_text(df, "name", config)
        assert stats.length_mean == pytest.approx(4.0)

    def test_length_min_max(self, config: ProfileConfig):
        df = pl.DataFrame({"name": ["ab", "abcd", "abcdef"]})
        stats = analyze_text(df, "name", config)
        assert stats.length_min == 2
        assert stats.length_max == 6

    def test_length_with_nulls(self, config: ProfileConfig):
        df = pl.DataFrame({"name": ["hello", None, "hi"]})
        stats = analyze_text(df, "name", config)
        assert stats.length_min == 2
        assert stats.length_max == 5
        assert stats.length_mean == pytest.approx(3.5)

    def test_uniform_length(self, config: ProfileConfig):
        df = pl.DataFrame({"code": ["abc", "def", "ghi"]})
        stats = analyze_text(df, "code", config)
        assert stats.length_min == stats.length_max == 3
        assert stats.length_mean == pytest.approx(3.0)


class TestTextEdgeCases:
    def test_all_null(self, config: ProfileConfig):
        df = pl.DataFrame({"name": pl.Series([None, None], dtype=pl.Utf8)})
        stats = analyze_text(df, "name", config)
        assert stats.count == 2
        assert stats.missing_count == 2
        assert stats.distinct_count == 0
        assert stats.length_mean is None
        assert stats.length_min is None
        assert stats.length_max is None

    def test_empty_dataframe(self, config: ProfileConfig):
        df = pl.DataFrame({"name": pl.Series([], dtype=pl.Utf8)})
        stats = analyze_text(df, "name", config)
        assert stats.count == 0

    def test_empty_strings(self, config: ProfileConfig):
        df = pl.DataFrame({"name": ["", "", "abc"]})
        stats = analyze_text(df, "name", config)
        assert stats.length_min == 0
        assert stats.length_max == 3

    def test_single_value(self, config: ProfileConfig):
        df = pl.DataFrame({"name": ["hello"]})
        stats = analyze_text(df, "name", config)
        assert stats.distinct_count == 1
        assert stats.length_mean == pytest.approx(5.0)
