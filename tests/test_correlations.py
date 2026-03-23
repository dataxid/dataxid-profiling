from __future__ import annotations

import polars as pl
import pytest

from dataxid_profiling._config import ProfileConfig
from dataxid_profiling._correlations import compute_correlations
from dataxid_profiling._type_inference import infer_types


def _corr(df: pl.DataFrame, config: ProfileConfig | None = None) -> dict[str, pl.DataFrame]:
    config = config or ProfileConfig()
    column_types = infer_types(df, config)
    return compute_correlations(df, column_types, config)


class TestPearsonBasic:
    def test_returns_pearson_key(self, numeric_df: pl.DataFrame):
        result = _corr(numeric_df)
        assert "pearson" in result

    def test_matrix_shape(self, numeric_df: pl.DataFrame):
        matrix = _corr(numeric_df)["pearson"]
        n_numeric = 3  # age, salary, score
        assert matrix.height == n_numeric
        assert matrix.width == n_numeric + 1  # +1 for "column" label

    def test_diagonal_is_one(self, numeric_df: pl.DataFrame):
        matrix = _corr(numeric_df)["pearson"]
        for row in matrix.iter_rows(named=True):
            col_name = row["column"]
            assert row[col_name] == pytest.approx(1.0)

    def test_symmetric(self, numeric_df: pl.DataFrame):
        matrix = _corr(numeric_df)["pearson"]
        rows = {r["column"]: r for r in matrix.iter_rows(named=True)}
        cols = [c for c in matrix.columns if c != "column"]
        for a in cols:
            for b in cols:
                assert rows[a][b] == pytest.approx(rows[b][a], abs=1e-10)

    def test_values_in_range(self, numeric_df: pl.DataFrame):
        matrix = _corr(numeric_df)["pearson"]
        for row in matrix.iter_rows(named=True):
            for col in matrix.columns:
                if col == "column":
                    continue
                assert -1.0 <= row[col] <= 1.0

    def test_perfect_correlation(self):
        df = pl.DataFrame({"a": [1, 2, 3, 4, 5], "b": [2, 4, 6, 8, 10]})
        matrix = _corr(df)["pearson"]
        rows = {r["column"]: r for r in matrix.iter_rows(named=True)}
        assert rows["a"]["b"] == pytest.approx(1.0)

    def test_negative_correlation(self):
        df = pl.DataFrame({"a": [1, 2, 3, 4, 5], "b": [10, 8, 6, 4, 2]})
        matrix = _corr(df)["pearson"]
        rows = {r["column"]: r for r in matrix.iter_rows(named=True)}
        assert rows["a"]["b"] == pytest.approx(-1.0)

    def test_columns_sorted(self, numeric_df: pl.DataFrame):
        matrix = _corr(numeric_df)["pearson"]
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
        matrix = result["pearson"]
        rows = {r["column"]: r for r in matrix.iter_rows(named=True)}
        assert -1.0 <= rows["a"]["b"] <= 1.0

    def test_mixed_df_only_numeric(self, mixed_df: pl.DataFrame):
        result = _corr(mixed_df)
        assert "pearson" in result
        matrix = result["pearson"]
        col_labels = matrix["column"].to_list()
        for label in col_labels:
            assert label in ("age", "id", "salary")

    def test_empty_dataframe(self, empty_df: pl.DataFrame):
        result = _corr(empty_df)
        assert result == {}
