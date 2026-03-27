from __future__ import annotations

import pandas as pd
import polars as pl
import pytest
from scipy.stats import kendalltau

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


class TestKendallBasic:
    def test_returns_kendall_key(self, numeric_df: pl.DataFrame):
        result = _corr(numeric_df)
        assert "kendall" in result
        assert isinstance(result["kendall"], CorrelationResult)

    def test_matrix_shape(self, numeric_df: pl.DataFrame):
        matrix = _corr(numeric_df)["kendall"].matrix
        n_numeric = 3
        assert matrix.height == n_numeric
        assert matrix.width == n_numeric + 1

    def test_diagonal_is_one(self, numeric_df: pl.DataFrame):
        matrix = _corr(numeric_df)["kendall"].matrix
        for row in matrix.iter_rows(named=True):
            assert row[row["column"]] == pytest.approx(1.0)

    def test_symmetric(self, numeric_df: pl.DataFrame):
        matrix = _corr(numeric_df)["kendall"].matrix
        rows = {r["column"]: r for r in matrix.iter_rows(named=True)}
        cols = [c for c in matrix.columns if c != "column"]
        for a in cols:
            for b in cols:
                assert rows[a][b] == pytest.approx(rows[b][a], abs=1e-10)

    def test_values_in_range(self, numeric_df: pl.DataFrame):
        matrix = _corr(numeric_df)["kendall"].matrix
        for row in matrix.iter_rows(named=True):
            for col in matrix.columns:
                if col == "column":
                    continue
                assert -1.0 <= row[col] <= 1.0

    def test_perfect_concordance(self):
        df = pl.DataFrame({"a": [1, 2, 3, 4, 5], "b": [2, 4, 6, 8, 10]})
        matrix = _corr(df)["kendall"].matrix
        rows = {r["column"]: r for r in matrix.iter_rows(named=True)}
        assert rows["a"]["b"] == pytest.approx(1.0)

    def test_perfect_discordance(self):
        df = pl.DataFrame({"a": [1, 2, 3, 4, 5], "b": [10, 8, 6, 4, 2]})
        matrix = _corr(df)["kendall"].matrix
        rows = {r["column"]: r for r in matrix.iter_rows(named=True)}
        assert rows["a"]["b"] == pytest.approx(-1.0)

    def test_with_ties(self):
        """Kendall tau-b handles tied values correctly."""
        df = pl.DataFrame({"a": [1, 2, 2, 3, 4], "b": [1, 2, 2, 3, 5]})
        matrix = _corr(df)["kendall"].matrix
        rows = {r["column"]: r for r in matrix.iter_rows(named=True)}
        assert 0.0 < rows["a"]["b"] <= 1.0

    def test_pvalues_present(self, numeric_df: pl.DataFrame):
        result = _corr(numeric_df)["kendall"]
        assert result.pvalues is not None
        assert result.pvalues.height == result.matrix.height
        for row in result.pvalues.iter_rows(named=True):
            for col in result.pvalues.columns:
                if col == "column":
                    continue
                assert 0.0 <= row[col] <= 1.0

    def test_columns_sorted(self, numeric_df: pl.DataFrame):
        matrix = _corr(numeric_df)["kendall"].matrix
        col_labels = matrix["column"].to_list()
        assert col_labels == sorted(col_labels)

    def test_matches_scipy_reference(self):
        """Our matrix values must match scipy.stats.kendalltau directly."""
        df = pl.DataFrame({"a": [1, 2, 2, 3, 4], "b": [1, 3, 2, 4, 5]})
        matrix = _corr(df)["kendall"].matrix
        rows = {r["column"]: r for r in matrix.iter_rows(named=True)}
        expected_tau, _ = kendalltau(df["a"].to_list(), df["b"].to_list())
        assert rows["a"]["b"] == pytest.approx(expected_tau, abs=1e-10)

    def test_perfect_concordance_pvalue_near_zero(self):
        df = pl.DataFrame({"a": list(range(20)), "b": list(range(20))})
        result = _corr(df)["kendall"]
        rows = {r["column"]: r for r in result.pvalues.iter_rows(named=True)}
        assert rows["a"]["b"] == pytest.approx(0.0, abs=1e-6)

    def test_constant_column_nan(self):
        """Kendall of a constant vs variable should be NaN (zero variance)."""
        import math

        df = pl.DataFrame({
            "a": [1, 1, 1, 1, 1],
            "b": [1, 2, 3, 4, 5],
            "c": [5, 4, 3, 2, 1],
        })
        result = _corr(df)["kendall"]
        rows = {r["column"]: r for r in result.matrix.iter_rows(named=True)}
        assert math.isnan(rows["a"]["b"])
        assert math.isnan(rows["a"]["c"])


class TestCramersVBasic:
    @pytest.fixture()
    def cat_df(self) -> pl.DataFrame:
        return pl.DataFrame({
            "color": ["red", "blue", "red", "green", "blue"] * 20,
            "shape": ["circle", "square", "circle", "triangle", "square"] * 20,
            "size": ["S", "M", "L", "S", "M"] * 20,
        })

    def test_returns_cramers_v_key(self, cat_df: pl.DataFrame):
        result = _corr(cat_df)
        assert "cramers_v" in result
        assert isinstance(result["cramers_v"], CorrelationResult)

    def test_matrix_shape(self, cat_df: pl.DataFrame):
        matrix = _corr(cat_df)["cramers_v"].matrix
        assert matrix.height == 3
        assert matrix.width == 3 + 1

    def test_diagonal_is_one(self, cat_df: pl.DataFrame):
        matrix = _corr(cat_df)["cramers_v"].matrix
        for row in matrix.iter_rows(named=True):
            assert row[row["column"]] == pytest.approx(1.0)

    def test_symmetric(self, cat_df: pl.DataFrame):
        matrix = _corr(cat_df)["cramers_v"].matrix
        rows = {r["column"]: r for r in matrix.iter_rows(named=True)}
        cols = [c for c in matrix.columns if c != "column"]
        for a in cols:
            for b in cols:
                assert rows[a][b] == pytest.approx(rows[b][a], abs=1e-10)

    def test_values_in_range(self, cat_df: pl.DataFrame):
        matrix = _corr(cat_df)["cramers_v"].matrix
        for row in matrix.iter_rows(named=True):
            for col in matrix.columns:
                if col == "column":
                    continue
                assert 0.0 <= row[col] <= 1.0

    def test_no_pvalues(self, cat_df: pl.DataFrame):
        result = _corr(cat_df)["cramers_v"]
        assert result.pvalues is None

    def test_perfect_association(self):
        """Identical columns → V = 1.0."""
        df = pl.DataFrame({
            "a": ["x", "y", "z"] * 10,
            "b": ["x", "y", "z"] * 10,
        })
        matrix = _corr(df)["cramers_v"].matrix
        rows = {r["column"]: r for r in matrix.iter_rows(named=True)}
        assert rows["a"]["b"] == pytest.approx(1.0)

    def test_no_association(self):
        """Independent columns → V ≈ 0."""
        import random
        random.seed(42)
        a = [random.choice(["x", "y", "z"]) for _ in range(1000)]
        b = [random.choice(["p", "q", "r"]) for _ in range(1000)]
        df = pl.DataFrame({"a": a, "b": b})
        matrix = _corr(df)["cramers_v"].matrix
        rows = {r["column"]: r for r in matrix.iter_rows(named=True)}
        assert rows["a"]["b"] < 0.1

    def test_columns_sorted(self, cat_df: pl.DataFrame):
        matrix = _corr(cat_df)["cramers_v"].matrix
        col_labels = matrix["column"].to_list()
        assert col_labels == sorted(col_labels)

    def test_single_categorical_column_skips(self):
        df = pl.DataFrame({"a": [1, 2, 3], "cat": ["x", "y", "z"]})
        result = _corr(df)
        assert "cramers_v" not in result

    def test_mixed_df_cramers_only_categorical(self):
        df = pl.DataFrame({
            "num1": [1, 2, 3, 4, 5],
            "num2": [5, 4, 3, 2, 1],
            "cat1": ["a", "b", "a", "b", "a"],
            "cat2": ["x", "y", "x", "y", "x"],
        })
        result = _corr(df)
        assert "pearson" in result
        assert "cramers_v" in result
        cv_cols = result["cramers_v"].matrix["column"].to_list()
        assert "num1" not in cv_cols
        assert "cat1" in cv_cols

    def test_matches_scipy_reference(self):
        """Our Cramér's V must match scipy chi2_contingency calculation."""
        import numpy as np
        from scipy.stats import chi2_contingency

        df = pl.DataFrame({
            "a": ["x", "x", "y", "y", "z", "z"] * 10,
            "b": ["p", "q", "p", "r", "q", "r"] * 10,
        })
        our_v = _corr(df)["cramers_v"].matrix
        rows = {r["column"]: r for r in our_v.iter_rows(named=True)}

        ct = pd.crosstab(
            pd.Series(df["a"].to_list()),
            pd.Series(df["b"].to_list()),
        )
        chi2, _, _, _ = chi2_contingency(ct)
        n_obs = ct.values.sum()
        min_dim = min(ct.shape) - 1
        expected_v = np.sqrt(chi2 / (n_obs * min_dim))

        assert rows["a"]["b"] == pytest.approx(expected_v, abs=1e-4)


class TestCorrelationEdgeCases:
    def test_single_numeric_column(self):
        df = pl.DataFrame({"a": [1, 2, 3], "city": ["x", "y", "z"]})
        result = _corr(df)
        assert result == {}

    def test_no_numeric_columns_has_cramers_v(self, categorical_df: pl.DataFrame):
        result = _corr(categorical_df)
        assert "pearson" not in result
        assert "cramers_v" in result

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
