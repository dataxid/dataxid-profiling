from __future__ import annotations

import polars as pl
import pytest

from dataxid_profiling._config import ProfileConfig
from dataxid_profiling._correlations import CorrelationResult, compute_correlations
from dataxid_profiling._type_inference import infer_types


def _corr(
    df: pl.DataFrame, config: ProfileConfig | None = None
) -> dict[str, CorrelationResult]:
    config = config or ProfileConfig()
    column_types = infer_types(df, config)
    return compute_correlations(df, column_types, config)


class TestPearsonBasic:
    def test_returns_pearson_key(self, numeric_df: pl.DataFrame):
        result = _corr(numeric_df)
        assert "pearson" in result
        assert isinstance(result["pearson"], CorrelationResult)

    def test_matrix_shape(self, numeric_df: pl.DataFrame):
        matrix = _corr(numeric_df)["pearson"].matrix
        n_numeric = 3  # age, salary, score
        assert matrix.height == n_numeric
        assert matrix.width == n_numeric + 1  # +1 for "column" label

    def test_diagonal_is_one(self, numeric_df: pl.DataFrame):
        matrix = _corr(numeric_df)["pearson"].matrix
        for row in matrix.iter_rows(named=True):
            col_name = row["column"]
            assert row[col_name] == pytest.approx(1.0)

    def test_symmetric(self, numeric_df: pl.DataFrame):
        matrix = _corr(numeric_df)["pearson"].matrix
        rows = {r["column"]: r for r in matrix.iter_rows(named=True)}
        cols = [c for c in matrix.columns if c != "column"]
        for a in cols:
            for b in cols:
                assert rows[a][b] == pytest.approx(rows[b][a], abs=1e-10)

    def test_values_in_range(self, numeric_df: pl.DataFrame):
        matrix = _corr(numeric_df)["pearson"].matrix
        for row in matrix.iter_rows(named=True):
            for col in matrix.columns:
                if col == "column":
                    continue
                assert -1.0 <= row[col] <= 1.0

    def test_perfect_correlation(self):
        df = pl.DataFrame({"a": [1, 2, 3, 4, 5], "b": [2, 4, 6, 8, 10]})
        matrix = _corr(df)["pearson"].matrix
        rows = {r["column"]: r for r in matrix.iter_rows(named=True)}
        assert rows["a"]["b"] == pytest.approx(1.0)

    def test_negative_correlation(self):
        df = pl.DataFrame({"a": [1, 2, 3, 4, 5], "b": [10, 8, 6, 4, 2]})
        matrix = _corr(df)["pearson"].matrix
        rows = {r["column"]: r for r in matrix.iter_rows(named=True)}
        assert rows["a"]["b"] == pytest.approx(-1.0)

    def test_columns_sorted(self, numeric_df: pl.DataFrame):
        matrix = _corr(numeric_df)["pearson"].matrix
        col_labels = matrix["column"].to_list()
        assert col_labels == sorted(col_labels)

    def test_pvalues_present(self, numeric_df: pl.DataFrame):
        result = _corr(numeric_df)["pearson"]
        assert result.pvalues is not None
        assert result.pvalues.height == result.matrix.height
        for row in result.pvalues.iter_rows(named=True):
            for col in result.pvalues.columns:
                if col == "column":
                    continue
                assert 0.0 <= row[col] <= 1.0


class TestSpearmanBasic:
    def test_returns_spearman_key(self, numeric_df: pl.DataFrame):
        result = _corr(numeric_df)
        assert "spearman" in result
        assert isinstance(result["spearman"], CorrelationResult)

    def test_matrix_shape(self, numeric_df: pl.DataFrame):
        matrix = _corr(numeric_df)["spearman"].matrix
        n_numeric = 3
        assert matrix.height == n_numeric
        assert matrix.width == n_numeric + 1

    def test_diagonal_is_one(self, numeric_df: pl.DataFrame):
        matrix = _corr(numeric_df)["spearman"].matrix
        for row in matrix.iter_rows(named=True):
            assert row[row["column"]] == pytest.approx(1.0)

    def test_symmetric(self, numeric_df: pl.DataFrame):
        matrix = _corr(numeric_df)["spearman"].matrix
        rows = {r["column"]: r for r in matrix.iter_rows(named=True)}
        cols = [c for c in matrix.columns if c != "column"]
        for a in cols:
            for b in cols:
                assert rows[a][b] == pytest.approx(rows[b][a], abs=1e-10)

    def test_values_in_range(self, numeric_df: pl.DataFrame):
        matrix = _corr(numeric_df)["spearman"].matrix
        for row in matrix.iter_rows(named=True):
            for col in matrix.columns:
                if col == "column":
                    continue
                assert -1.0 <= row[col] <= 1.0

    def test_perfect_monotonic(self):
        df = pl.DataFrame({"a": [1, 2, 3, 4, 5], "b": [2, 4, 6, 8, 10]})
        matrix = _corr(df)["spearman"].matrix
        rows = {r["column"]: r for r in matrix.iter_rows(named=True)}
        assert rows["a"]["b"] == pytest.approx(1.0)

    def test_negative_monotonic(self):
        df = pl.DataFrame({"a": [1, 2, 3, 4, 5], "b": [10, 8, 6, 4, 2]})
        matrix = _corr(df)["spearman"].matrix
        rows = {r["column"]: r for r in matrix.iter_rows(named=True)}
        assert rows["a"]["b"] == pytest.approx(-1.0)

    def test_nonlinear_monotonic(self):
        """Spearman captures monotonic non-linear relationships better than Pearson."""
        df = pl.DataFrame({"a": [1, 2, 3, 4, 5], "b": [1, 4, 9, 16, 25]})
        result = _corr(df)
        spearman = result["spearman"].matrix
        pearson = result["pearson"].matrix
        sp_rows = {r["column"]: r for r in spearman.iter_rows(named=True)}
        pe_rows = {r["column"]: r for r in pearson.iter_rows(named=True)}
        assert sp_rows["a"]["b"] == pytest.approx(1.0)
        assert pe_rows["a"]["b"] < 1.0

    def test_pvalues_present(self, numeric_df: pl.DataFrame):
        result = _corr(numeric_df)["spearman"]
        assert result.pvalues is not None
        assert result.pvalues.height == result.matrix.height
        for row in result.pvalues.iter_rows(named=True):
            for col in result.pvalues.columns:
                if col == "column":
                    continue
                assert 0.0 <= row[col] <= 1.0

    def test_perfect_correlation_pvalue_zero(self):
        df = pl.DataFrame({"a": [1, 2, 3, 4, 5], "b": [2, 4, 6, 8, 10]})
        result = _corr(df)["spearman"]
        rows = {r["column"]: r for r in result.pvalues.iter_rows(named=True)}
        assert rows["a"]["b"] == pytest.approx(0.0, abs=1e-6)

    def test_columns_sorted(self, numeric_df: pl.DataFrame):
        matrix = _corr(numeric_df)["spearman"].matrix
        col_labels = matrix["column"].to_list()
        assert col_labels == sorted(col_labels)


class TestCorrelationEdgeCases:
    def test_single_numeric_column(self):
        df = pl.DataFrame({"a": [1, 2, 3], "city": ["x", "y", "z"]})
        result = _corr(df)
        assert result == {}

    def test_no_numeric_columns(self, categorical_df: pl.DataFrame):
        result = _corr(categorical_df)
        assert result == {}

    def test_overview_mode_skips(self, numeric_df: pl.DataFrame):
        config = ProfileConfig(mode="overview")
        result = _corr(numeric_df, config)
        assert result == {}

    def test_with_nulls(self):
        df = pl.DataFrame({
            "a": [1.0, 2.0, None, 4.0, 5.0],
            "b": [None, 4.0, 6.0, 8.0, 10.0],
        })
        result = _corr(df)
        assert "pearson" in result
        matrix = result["pearson"].matrix
        rows = {r["column"]: r for r in matrix.iter_rows(named=True)}
        assert -1.0 <= rows["a"]["b"] <= 1.0

    def test_mixed_df_only_numeric(self, mixed_df: pl.DataFrame):
        result = _corr(mixed_df)
        assert "pearson" in result
        matrix = result["pearson"].matrix
        col_labels = matrix["column"].to_list()
        for label in col_labels:
            assert label in ("age", "id", "salary")

    def test_empty_dataframe(self, empty_df: pl.DataFrame):
        result = _corr(empty_df)
        assert result == {}
