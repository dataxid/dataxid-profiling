from __future__ import annotations

import polars as pl
import pytest

from dataxid_profiling._analyzers._boolean import analyze_boolean
from dataxid_profiling._config import ProfileConfig
from dataxid_profiling._type_inference import ColumnType


@pytest.fixture
def config() -> ProfileConfig:
    return ProfileConfig()


class TestBooleanBasicStats:
    def test_count_and_missing(self, boolean_df: pl.DataFrame, config: ProfileConfig):
        stats = analyze_boolean(boolean_df, "active", config)
        assert stats.count == 6
        assert stats.missing_count == 1
        assert stats.missing_pct == pytest.approx(1 / 6)

    def test_column_metadata(self, boolean_df: pl.DataFrame, config: ProfileConfig):
        stats = analyze_boolean(boolean_df, "active", config)
        assert stats.column_name == "active"
        assert stats.column_type == ColumnType.BOOLEAN


class TestBooleanDistribution:
    def test_true_false_counts(self, config: ProfileConfig):
        df = pl.DataFrame({"flag": [True, True, True, False, False]})
        stats = analyze_boolean(df, "flag", config)
        assert stats.true_count == 3
        assert stats.false_count == 2

    def test_true_false_pct(self, config: ProfileConfig):
        df = pl.DataFrame({"flag": [True, True, True, False, False]})
        stats = analyze_boolean(df, "flag", config)
        assert stats.true_pct == pytest.approx(0.6)
        assert stats.false_pct == pytest.approx(0.4)

    def test_pct_excludes_null(self, config: ProfileConfig):
        df = pl.DataFrame({"flag": [True, False, None, None]})
        stats = analyze_boolean(df, "flag", config)
        assert stats.true_pct == pytest.approx(0.5)
        assert stats.false_pct == pytest.approx(0.5)
        assert stats.true_count + stats.false_count == 2

    def test_all_true(self, config: ProfileConfig):
        df = pl.DataFrame({"flag": [True, True, True]})
        stats = analyze_boolean(df, "flag", config)
        assert stats.true_count == 3
        assert stats.false_count == 0
        assert stats.true_pct == pytest.approx(1.0)
        assert stats.false_pct == pytest.approx(0.0)

    def test_all_false(self, config: ProfileConfig):
        df = pl.DataFrame({"flag": [False, False, False]})
        stats = analyze_boolean(df, "flag", config)
        assert stats.true_count == 0
        assert stats.false_count == 3


class TestBooleanEdgeCases:
    def test_all_null(self, config: ProfileConfig):
        df = pl.DataFrame({"flag": pl.Series([None, None, None], dtype=pl.Boolean)})
        stats = analyze_boolean(df, "flag", config)
        assert stats.count == 3
        assert stats.missing_count == 3
        assert stats.true_count == 0
        assert stats.false_count == 0
        assert stats.true_pct == pytest.approx(0.0)

    def test_empty_dataframe(self, config: ProfileConfig):
        df = pl.DataFrame({"flag": pl.Series([], dtype=pl.Boolean)})
        stats = analyze_boolean(df, "flag", config)
        assert stats.count == 0
        assert stats.missing_count == 0

    def test_single_true(self, config: ProfileConfig):
        df = pl.DataFrame({"flag": [True]})
        stats = analyze_boolean(df, "flag", config)
        assert stats.true_count == 1
        assert stats.true_pct == pytest.approx(1.0)
