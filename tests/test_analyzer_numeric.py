from __future__ import annotations

import polars as pl
import pytest

from dataxid_profiling._analyzers._numeric import analyze_numeric
from dataxid_profiling._config import ProfileConfig
from dataxid_profiling._type_inference import ColumnType


@pytest.fixture
def config() -> ProfileConfig:
    return ProfileConfig()


class TestNumericBasicStats:
    def test_count_and_missing(self, numeric_df: pl.DataFrame, config: ProfileConfig):
        stats = analyze_numeric(numeric_df, "age", config)
        assert stats.count == 6
        assert stats.missing_count == 1
        assert stats.missing_pct == pytest.approx(1 / 6)

    def test_column_metadata(self, numeric_df: pl.DataFrame, config: ProfileConfig):
        stats = analyze_numeric(numeric_df, "age", config)
        assert stats.column_name == "age"
        assert stats.column_type == ColumnType.NUMERIC

    def test_distinct(self, numeric_df: pl.DataFrame, config: ProfileConfig):
        stats = analyze_numeric(numeric_df, "age", config)
        assert stats.distinct_count >= 1
        assert 0.0 <= stats.distinct_pct <= 1.0


class TestNumericDescriptiveStats:
    def test_mean(self, config: ProfileConfig):
        df = pl.DataFrame({"val": [10.0, 20.0, 30.0, 40.0, 50.0]})
        stats = analyze_numeric(df, "val", config)
        assert stats.mean == pytest.approx(30.0)

    def test_std(self, config: ProfileConfig):
        df = pl.DataFrame({"val": [10.0, 20.0, 30.0, 40.0, 50.0]})
        stats = analyze_numeric(df, "val", config)
        assert stats.std is not None
        assert stats.std > 0

    def test_min_max_range(self, config: ProfileConfig):
        df = pl.DataFrame({"val": [5, 10, 15, 20, 25]})
        stats = analyze_numeric(df, "val", config)
        assert stats.min == pytest.approx(5.0)
        assert stats.max == pytest.approx(25.0)
        assert stats.range == pytest.approx(20.0)

    def test_quantiles(self, config: ProfileConfig):
        df = pl.DataFrame({"val": list(range(1, 101))})
        stats = analyze_numeric(df, "val", config)
        assert stats.q25 is not None
        assert stats.median is not None
        assert stats.q75 is not None
        assert stats.q25 < stats.median < stats.q75

    def test_iqr(self, config: ProfileConfig):
        df = pl.DataFrame({"val": list(range(1, 101))})
        stats = analyze_numeric(df, "val", config)
        assert stats.iqr is not None
        assert stats.iqr == pytest.approx(stats.q75 - stats.q25)

    def test_skewness_symmetric(self, config: ProfileConfig):
        df = pl.DataFrame({"val": list(range(1, 101))})
        stats = analyze_numeric(df, "val", config)
        assert stats.skewness is not None
        assert abs(stats.skewness) < 0.5

    def test_kurtosis(self, config: ProfileConfig):
        df = pl.DataFrame({"val": list(range(1, 101))})
        stats = analyze_numeric(df, "val", config)
        assert stats.kurtosis is not None


class TestNumericZerosAndNegatives:
    def test_zeros(self, config: ProfileConfig):
        df = pl.DataFrame({"val": [0, 0, 1, 2, 3]})
        stats = analyze_numeric(df, "val", config)
        assert stats.zeros_count == 2
        assert stats.zeros_pct == pytest.approx(0.4)

    def test_negatives(self, config: ProfileConfig):
        df = pl.DataFrame({"val": [-5, -3, 0, 2, 4]})
        stats = analyze_numeric(df, "val", config)
        assert stats.negative_count == 2
        assert stats.negative_pct == pytest.approx(0.4)

    def test_no_zeros_no_negatives(self, config: ProfileConfig):
        df = pl.DataFrame({"val": [1, 2, 3]})
        stats = analyze_numeric(df, "val", config)
        assert stats.zeros_count == 0
        assert stats.negative_count == 0


class TestNumericHistogram:
    def test_histogram_not_empty(self, numeric_df: pl.DataFrame, config: ProfileConfig):
        stats = analyze_numeric(numeric_df, "salary", config)
        assert len(stats.histogram) > 0

    def test_histogram_has_breakpoint_and_count(self, config: ProfileConfig):
        df = pl.DataFrame({"val": list(range(100))})
        stats = analyze_numeric(df, "val", config)
        for bucket in stats.histogram:
            assert "breakpoint" in bucket
            assert "count" in bucket

    def test_histogram_custom_bins(self):
        df = pl.DataFrame({"val": list(range(100))})
        config = ProfileConfig(histogram_bins=10)
        stats = analyze_numeric(df, "val", config)
        assert len(stats.histogram) == 10


class TestNumericEdgeCases:
    def test_all_null(self, config: ProfileConfig):
        df = pl.DataFrame({"val": pl.Series([None, None, None], dtype=pl.Float64)})
        stats = analyze_numeric(df, "val", config)
        assert stats.count == 3
        assert stats.missing_count == 3
        assert stats.mean is None

    def test_single_value(self, config: ProfileConfig):
        df = pl.DataFrame({"val": [42]})
        stats = analyze_numeric(df, "val", config)
        assert stats.mean == pytest.approx(42.0)
        assert stats.min == pytest.approx(42.0)
        assert stats.max == pytest.approx(42.0)
        assert stats.range == pytest.approx(0.0)

    def test_empty_dataframe(self, config: ProfileConfig):
        df = pl.DataFrame({"val": pl.Series([], dtype=pl.Int64)})
        stats = analyze_numeric(df, "val", config)
        assert stats.count == 0
        assert stats.mean is None

    def test_float_types(self, config: ProfileConfig):
        df = pl.DataFrame({"val": pl.Series([1.5, 2.5, 3.5], dtype=pl.Float32)})
        stats = analyze_numeric(df, "val", config)
        assert stats.mean == pytest.approx(2.5, abs=0.01)


class TestNumericNewMetrics:
    def test_sum(self, config: ProfileConfig):
        df = pl.DataFrame({"val": [10, 20, 30]})
        stats = analyze_numeric(df, "val", config)
        assert stats.sum == pytest.approx(60.0)

    def test_variance(self, config: ProfileConfig):
        df = pl.DataFrame({"val": [10.0, 20.0, 30.0, 40.0, 50.0]})
        stats = analyze_numeric(df, "val", config)
        assert stats.variance is not None
        assert stats.variance > 0
        assert stats.std is not None
        assert stats.variance == pytest.approx(stats.std**2, rel=1e-3)

    def test_cv(self, config: ProfileConfig):
        df = pl.DataFrame({"val": [10.0, 20.0, 30.0, 40.0, 50.0]})
        stats = analyze_numeric(df, "val", config)
        assert stats.cv is not None
        assert stats.cv == pytest.approx(stats.std / abs(stats.mean), rel=1e-3)

    def test_cv_zero_mean(self, config: ProfileConfig):
        df = pl.DataFrame({"val": [-1.0, 0.0, 1.0]})
        stats = analyze_numeric(df, "val", config)
        assert stats.cv is None

    def test_mad(self, config: ProfileConfig):
        df = pl.DataFrame({"val": [1.0, 2.0, 3.0, 4.0, 5.0]})
        stats = analyze_numeric(df, "val", config)
        assert stats.mad is not None
        assert stats.mad == pytest.approx(1.0)

    def test_p5_p95(self, config: ProfileConfig):
        df = pl.DataFrame({"val": list(range(1, 101))})
        stats = analyze_numeric(df, "val", config)
        assert stats.p5 is not None
        assert stats.p95 is not None
        assert stats.p5 < stats.q25
        assert stats.p95 > stats.q75

    def test_n_infinite(self, config: ProfileConfig):
        df = pl.DataFrame({"val": [1.0, float("inf"), -float("inf"), 4.0]})
        stats = analyze_numeric(df, "val", config)
        assert stats.n_infinite == 2

    def test_n_infinite_none_for_int(self, config: ProfileConfig):
        df = pl.DataFrame({"val": [1, 2, 3, 4, 5]})
        stats = analyze_numeric(df, "val", config)
        assert stats.n_infinite == 0

    def test_monotonic_increase(self, config: ProfileConfig):
        df = pl.DataFrame({"val": [1, 2, 3, 4, 5]})
        stats = analyze_numeric(df, "val", config)
        assert stats.monotonic_increase is True
        assert stats.monotonic_decrease is False

    def test_monotonic_decrease(self, config: ProfileConfig):
        df = pl.DataFrame({"val": [5, 4, 3, 2, 1]})
        stats = analyze_numeric(df, "val", config)
        assert stats.monotonic_increase is False
        assert stats.monotonic_decrease is True

    def test_monotonic_constant(self, config: ProfileConfig):
        df = pl.DataFrame({"val": [3, 3, 3, 3]})
        stats = analyze_numeric(df, "val", config)
        assert stats.monotonic_increase is True
        assert stats.monotonic_decrease is True

    def test_monotonic_non_monotonic(self, config: ProfileConfig):
        df = pl.DataFrame({"val": [1, 3, 2, 4]})
        stats = analyze_numeric(df, "val", config)
        assert stats.monotonic_increase is False
        assert stats.monotonic_decrease is False

    def test_monotonic_with_nulls(self, config: ProfileConfig):
        df = pl.DataFrame({"val": [1, None, 3, None, 5]})
        stats = analyze_numeric(df, "val", config)
        assert stats.monotonic_increase is True

    def test_value_counts(self, config: ProfileConfig):
        df = pl.DataFrame({"val": [1, 1, 1, 2, 2, 3]})
        stats = analyze_numeric(df, "val", config)
        assert len(stats.value_counts) > 0
        assert stats.value_counts[0]["value"] == 1
        assert stats.value_counts[0]["count"] == 3

    def test_value_counts_respects_n_top(self):
        df = pl.DataFrame({"val": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5]})
        config = ProfileConfig(n_top_values=3)
        stats = analyze_numeric(df, "val", config)
        assert len(stats.value_counts) == 3
