from __future__ import annotations

import polars as pl
import pytest

from dataxid_profiling._config import ProfileConfig
from dataxid_profiling._interactions import (
    BoxPlotGroup,
    InteractionData,
    compute_interactions,
)
from dataxid_profiling._type_inference import ColumnType, infer_types


def _interact(
    df: pl.DataFrame, config: ProfileConfig | None = None
) -> InteractionData | None:
    config = config or ProfileConfig()
    column_types = infer_types(df, config)
    return compute_interactions(df, column_types, config)


# -- Basic scatter data -------------------------------------------------------


class TestScatterData:
    def test_returns_interaction_data(self, numeric_df: pl.DataFrame):
        result = _interact(numeric_df)
        assert result is not None
        assert isinstance(result, InteractionData)

    def test_numeric_columns_detected(self, numeric_df: pl.DataFrame):
        result = _interact(numeric_df)
        assert len(result.numeric_columns) >= 2
        for col in result.numeric_columns:
            assert col in numeric_df.columns

    def test_numeric_data_columnar(self, numeric_df: pl.DataFrame):
        result = _interact(numeric_df)
        for col in result.numeric_columns:
            assert col in result.numeric_data
            assert isinstance(result.numeric_data[col], list)
            assert len(result.numeric_data[col]) > 0

    def test_float_precision_4_decimals(self):
        df = pl.DataFrame({"a": [1.123456789, 2.987654321], "b": [3.0, 4.0]})
        result = _interact(df)
        for val in result.numeric_data["a"]:
            s = str(val)
            if "." in s:
                decimals = len(s.split(".")[1])
                assert decimals <= 4

    def test_nulls_dropped(self):
        df = pl.DataFrame({"a": [1.0, None, 3.0, 4.0, 5.0], "b": [2.0, 3.0, None, 5.0, 6.0]})
        result = _interact(df)
        for col in result.numeric_columns:
            assert None not in result.numeric_data[col]

    def test_columns_sorted(self, numeric_df: pl.DataFrame):
        result = _interact(numeric_df)
        assert result.numeric_columns == sorted(result.numeric_columns)


# -- Box plot data ------------------------------------------------------------


class TestBoxPlotData:
    @pytest.fixture()
    def mixed_interact_df(self) -> pl.DataFrame:
        return pl.DataFrame({
            "age": list(range(20, 70)),
            "salary": [50000 + i * 1000 for i in range(50)],
            "dept": (["Eng", "Sales"] * 25),
        })

    def test_boxplot_stats_present(self, mixed_interact_df: pl.DataFrame):
        result = _interact(mixed_interact_df)
        assert result is not None
        assert len(result.boxplot_stats) > 0

    def test_boxplot_structure(self, mixed_interact_df: pl.DataFrame):
        result = _interact(mixed_interact_df)
        for cat_col, num_map in result.boxplot_stats.items():
            assert cat_col in result.categorical_columns
            for num_col, groups in num_map.items():
                assert num_col in result.numeric_columns
                for g in groups:
                    assert isinstance(g, BoxPlotGroup)

    def test_five_number_summary_order(self, mixed_interact_df: pl.DataFrame):
        result = _interact(mixed_interact_df)
        for num_map in result.boxplot_stats.values():
            for groups in num_map.values():
                for g in groups:
                    assert g.min <= g.q1 <= g.median <= g.q3 <= g.max

    def test_whiskers_at_fences(self):
        df = pl.DataFrame({
            "cat": ["A"] * 20,
            "val": [10, 20, 30, 35, 40, 42, 44, 46, 48, 50,
                    52, 54, 56, 58, 60, 65, 70, 80, 150, 200],
        })
        types = {"cat": ColumnType.CATEGORICAL, "val": ColumnType.NUMERIC}
        result = compute_interactions(df, types)
        g = result.boxplot_stats["cat"]["val"][0]
        iqr = g.q3 - g.q1
        assert g.max <= g.q3 + 1.5 * iqr
        assert g.min >= g.q1 - 1.5 * iqr

    def test_outliers_outside_whiskers(self):
        df = pl.DataFrame({
            "cat": ["A"] * 20,
            "val": [10, 20, 30, 35, 40, 42, 44, 46, 48, 50,
                    52, 54, 56, 58, 60, 65, 70, 80, 150, 200],
        })
        types = {"cat": ColumnType.CATEGORICAL, "val": ColumnType.NUMERIC}
        result = compute_interactions(df, types)
        g = result.boxplot_stats["cat"]["val"][0]
        for o in g.outliers:
            assert o < g.min or o > g.max

    def test_boxplot_precision_2_decimals(self, mixed_interact_df: pl.DataFrame):
        result = _interact(mixed_interact_df)
        for num_map in result.boxplot_stats.values():
            for groups in num_map.values():
                for g in groups:
                    for val in [g.min, g.q1, g.median, g.q3, g.max]:
                        s = str(val)
                        if "." in s:
                            assert len(s.split(".")[1]) <= 2

    def test_categorical_columns_sorted(self, mixed_interact_df: pl.DataFrame):
        result = _interact(mixed_interact_df)
        assert result.categorical_columns == sorted(result.categorical_columns)


# -- Edge cases ---------------------------------------------------------------


class TestInteractionEdgeCases:
    def test_overview_mode_returns_none(self, numeric_df: pl.DataFrame):
        result = _interact(numeric_df, ProfileConfig(mode="overview"))
        assert result is None

    def test_single_row_returns_none(self):
        df = pl.DataFrame({"a": [1], "b": [2]})
        assert _interact(df) is None

    def test_empty_df_returns_none(self, empty_df: pl.DataFrame):
        assert _interact(empty_df) is None

    def test_single_numeric_no_categorical_returns_none(self):
        df = pl.DataFrame({"a": [1, 2, 3]})
        assert _interact(df) is None

    def test_constant_numeric_filtered(self):
        df = pl.DataFrame({"a": [5, 5, 5, 5], "b": [1, 2, 3, 4]})
        result = _interact(df)
        assert result is None

    def test_all_null_numeric_filtered(self):
        df = pl.DataFrame({
            "a": [None, None, None, None],
            "b": [1, 2, 3, 4],
            "c": [5, 6, 7, 8],
        }, schema={"a": pl.Float64, "b": pl.Int64, "c": pl.Int64})
        result = _interact(df)
        if result is not None:
            assert "a" not in result.numeric_columns

    def test_high_cardinality_categorical_filtered(self):
        df = pl.DataFrame({
            "n1": list(range(100)),
            "n2": list(range(100, 200)),
            "cat": [f"val_{i}" for i in range(100)],
        })
        config = ProfileConfig(interaction_cardinality_limit=50)
        types = infer_types(df, config)
        result = compute_interactions(df, types, config)
        assert result is not None
        assert "cat" not in result.categorical_columns

    def test_only_boxplot_no_scatter(self):
        df = pl.DataFrame({
            "num": [10, 20, 30, 40, 50],
            "cat": ["A", "B", "A", "B", "A"],
        })
        types = {"num": ColumnType.NUMERIC, "cat": ColumnType.CATEGORICAL}
        result = compute_interactions(df, types)
        assert result is not None
        assert len(result.numeric_columns) == 1
        assert len(result.categorical_columns) == 1
        assert len(result.boxplot_stats) > 0


# -- Sampling -----------------------------------------------------------------


class TestSampling:
    def test_no_sampling_under_threshold(self):
        df = pl.DataFrame({"a": list(range(100)), "b": list(range(100, 200))})
        result = _interact(df)
        assert result.sampled is False
        assert result.total_rows == 100
        assert result.sample_size == 100

    def test_sampling_over_threshold(self):
        n = 2000
        df = pl.DataFrame({"a": list(range(n)), "b": list(range(n, 2 * n))})
        config = ProfileConfig(interaction_sample_size=1000)
        result = _interact(df, config)
        assert result.sampled is True
        assert result.total_rows == n
        assert result.sample_size == 1000

    def test_sampling_reproducible(self):
        n = 2000
        df = pl.DataFrame({"a": list(range(n)), "b": list(range(n, 2 * n))})
        config = ProfileConfig(interaction_sample_size=1000, interaction_sample_seed=42)
        r1 = _interact(df, config)
        r2 = _interact(df, config)
        assert r1.numeric_data["a"] == r2.numeric_data["a"]


# -- Integration with ProfileReport ------------------------------------------


class TestProfileReportInteractions:
    def test_interactions_property(self, mixed_df: pl.DataFrame):
        from dataxid_profiling import ProfileReport
        report = ProfileReport(mixed_df)
        assert report.interactions is not None

    def test_interactions_in_to_dict(self, mixed_df: pl.DataFrame):
        from dataxid_profiling import ProfileReport
        report = ProfileReport(mixed_df)
        d = report.to_dict()
        assert "interactions" in d
        assert d["interactions"] is not None
        assert "numeric_columns" in d["interactions"]

    def test_no_interactions_overview(self, mixed_df: pl.DataFrame):
        from dataxid_profiling import ProfileReport
        report = ProfileReport(mixed_df, mode="overview")
        assert report.interactions is None
        assert report.to_dict()["interactions"] is None

    def test_html_contains_interactions(self, mixed_df: pl.DataFrame):
        from dataxid_profiling import ProfileReport
        report = ProfileReport(mixed_df)
        html = report.to_html()
        assert "Interactions" in html
        assert "interact-x" in html

    def test_html_no_interactions_overview(self, mixed_df: pl.DataFrame):
        from dataxid_profiling import ProfileReport
        report = ProfileReport(mixed_df, mode="overview")
        html = report.to_html()
        assert "interact-x" not in html
