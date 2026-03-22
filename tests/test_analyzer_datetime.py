from __future__ import annotations

from datetime import date, datetime, time, timedelta

import polars as pl
import pytest

from dataxid_profiling._analyzers._datetime import analyze_datetime
from dataxid_profiling._config import ProfileConfig
from dataxid_profiling._type_inference import ColumnType


@pytest.fixture
def config() -> ProfileConfig:
    return ProfileConfig()


class TestDatetimeBasicStats:
    def test_count_and_missing(self, datetime_df: pl.DataFrame, config: ProfileConfig):
        stats = analyze_datetime(datetime_df, "created_at", config)
        assert stats.count == 6
        assert stats.missing_count == 1
        assert stats.missing_pct == pytest.approx(1 / 6)

    def test_column_metadata(self, datetime_df: pl.DataFrame, config: ProfileConfig):
        stats = analyze_datetime(datetime_df, "created_at", config)
        assert stats.column_name == "created_at"
        assert stats.column_type == ColumnType.DATETIME

    def test_distinct(self, datetime_df: pl.DataFrame, config: ProfileConfig):
        stats = analyze_datetime(datetime_df, "created_at", config)
        assert stats.distinct_count == 5
        assert stats.distinct_pct == pytest.approx(5 / 6)


class TestDatetimeMinMaxRange:
    def test_datetime_min_max(self, config: ProfileConfig):
        df = pl.DataFrame({
            "ts": [datetime(2024, 1, 1), datetime(2024, 6, 15), datetime(2024, 12, 31)],
        })
        stats = analyze_datetime(df, "ts", config)
        assert stats.min is not None
        assert stats.max is not None
        assert "2024-01-01" in stats.min
        assert "2024-12-31" in stats.max

    def test_date_min_max(self, config: ProfileConfig):
        df = pl.DataFrame({
            "d": [date(2023, 1, 1), date(2023, 6, 15), date(2023, 12, 31)],
        })
        stats = analyze_datetime(df, "d", config)
        assert "2023-01-01" in stats.min
        assert "2023-12-31" in stats.max

    def test_range_not_none(self, datetime_df: pl.DataFrame, config: ProfileConfig):
        stats = analyze_datetime(datetime_df, "created_at", config)
        assert stats.range is not None

    def test_date_range_value(self, config: ProfileConfig):
        df = pl.DataFrame({
            "d": [date(2024, 1, 1), date(2024, 1, 11)],
        })
        stats = analyze_datetime(df, "d", config)
        assert stats.range is not None
        assert "10" in stats.range


class TestDatetimeTimeAndDuration:
    def test_time_column(self, config: ProfileConfig):
        df = pl.DataFrame({"t": [time(8, 0), time(12, 0), time(18, 0)]})
        stats = analyze_datetime(df, "t", config)
        assert stats.count == 3
        assert stats.min is not None
        assert stats.max is not None

    def test_duration_column(self, config: ProfileConfig):
        df = pl.DataFrame({"dur": [timedelta(hours=1), timedelta(hours=5), timedelta(hours=10)]})
        stats = analyze_datetime(df, "dur", config)
        assert stats.count == 3
        assert stats.min is not None


class TestDatetimeEdgeCases:
    def test_all_null(self, config: ProfileConfig):
        df = pl.DataFrame({"ts": pl.Series([None, None], dtype=pl.Datetime)})
        stats = analyze_datetime(df, "ts", config)
        assert stats.count == 2
        assert stats.missing_count == 2
        assert stats.min is None
        assert stats.max is None
        assert stats.range is None

    def test_empty_dataframe(self, config: ProfileConfig):
        df = pl.DataFrame({"ts": pl.Series([], dtype=pl.Datetime)})
        stats = analyze_datetime(df, "ts", config)
        assert stats.count == 0

    def test_single_value(self, config: ProfileConfig):
        df = pl.DataFrame({"d": [date(2024, 6, 15)]})
        stats = analyze_datetime(df, "d", config)
        assert stats.distinct_count == 1
        assert stats.min == stats.max
