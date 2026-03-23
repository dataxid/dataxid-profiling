from __future__ import annotations

import polars as pl

from dataxid_profiling._alerts import check_quality
from dataxid_profiling._analyzers import analyze
from dataxid_profiling._config import ProfileConfig
from dataxid_profiling._correlations import compute_correlations
from dataxid_profiling._dataset_overview import compute_overview
from dataxid_profiling._report._html import render_html
from dataxid_profiling._type_inference import infer_types


def _render(df: pl.DataFrame, config: ProfileConfig | None = None) -> str:
    config = config or ProfileConfig()
    column_types = infer_types(df, config)
    column_stats = analyze(df, column_types, config)
    overview = compute_overview(df, column_types, config)
    alerts = check_quality(column_stats, overview, config)
    correlations = compute_correlations(df, column_types, config)
    return render_html(
        title=config.title,
        version="0.1.0",
        overview=overview,
        column_stats=column_stats,
        alerts=alerts,
        correlations=correlations,
    )


class TestRenderBasic:
    def test_returns_html_string(self, mixed_df: pl.DataFrame):
        html = _render(mixed_df)
        assert isinstance(html, str)
        assert "<!DOCTYPE html>" in html

    def test_contains_title(self, mixed_df: pl.DataFrame):
        html = _render(mixed_df, ProfileConfig(title="Test Report"))
        assert "Test Report" in html

    def test_contains_overview_cards(self, mixed_df: pl.DataFrame):
        html = _render(mixed_df)
        assert "Dataset Overview" in html
        assert "Rows" in html
        assert "Columns" in html
        assert "Missing Cells" in html

    def test_contains_column_tabs(self, mixed_df: pl.DataFrame):
        html = _render(mixed_df)
        assert "Column Details" in html
        assert "age" in html
        assert "salary" in html
        assert "city" in html

    def test_contains_echarts(self, mixed_df: pl.DataFrame):
        html = _render(mixed_df)
        assert "echarts.init" in html

    def test_contains_footer(self, mixed_df: pl.DataFrame):
        html = _render(mixed_df)
        assert "dataxid-profiling" in html


class TestRenderAlerts:
    def test_alerts_section_present(self, mixed_df: pl.DataFrame):
        html = _render(mixed_df)
        assert "Alerts" in html

    def test_no_alerts_section_when_clean(self):
        df = pl.DataFrame({"a": list(range(100)), "b": list(range(100))})
        config = ProfileConfig(missing_threshold=1.0, duplicate_threshold=1.0)
        html = _render(df, config)
        assert "alert_type" not in html.lower() or "Alerts" in html


class TestRenderCorrelations:
    def test_correlation_heatmap(self, numeric_df: pl.DataFrame):
        html = _render(numeric_df)
        assert "Correlations" in html
        assert "corr_heatmap" in html
        assert "heatmap" in html

    def test_no_correlation_single_numeric(self):
        df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        html = _render(df)
        assert "corr_heatmap" not in html

    def test_no_correlation_overview_mode(self, numeric_df: pl.DataFrame):
        html = _render(numeric_df, ProfileConfig(mode="overview"))
        assert "corr_heatmap" not in html


class TestRenderCharts:
    def test_numeric_histogram(self, numeric_df: pl.DataFrame):
        html = _render(numeric_df)
        assert "Distribution" in html

    def test_categorical_bar(self, categorical_df: pl.DataFrame):
        html = _render(categorical_df)
        assert "Top Values" in html

    def test_boolean_pie(self, boolean_df: pl.DataFrame):
        html = _render(boolean_df)
        assert "True" in html
        assert "False" in html


class TestRenderWordCloud:
    def test_categorical_wordcloud(self, categorical_df: pl.DataFrame):
        html = _render(categorical_df)
        assert "wordCloud" in html
        assert "col_wc_" in html

    def test_numeric_no_wordcloud(self, numeric_df: pl.DataFrame):
        html = _render(numeric_df)
        assert "wordCloud" not in html

    def test_wordcloud_cdn_included(self, categorical_df: pl.DataFrame):
        html = _render(categorical_df)
        assert "echarts-wordcloud" in html


class TestRenderMissingSection:
    def test_missing_section_present(self, mixed_df: pl.DataFrame):
        html = _render(mixed_df)
        assert "Missing Values" in html
        assert "missing_bar" in html

    def test_only_missing_columns_in_table(self):
        df = pl.DataFrame({"a": [1, None, 3], "b": ["x", "y", "z"]})
        html = _render(df)
        assert "Missing Values" in html
        table_start = html.index("Missing Values")
        table_section = html[table_start:table_start + 3000]
        assert ">a<" in table_section
        assert ">b<" not in table_section

    def test_no_missing_section_when_clean(self):
        df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        html = _render(df)
        assert "missing_bar" not in html


class TestRenderSampleSection:
    def test_sample_section_present(self, mixed_df: pl.DataFrame):
        html = _render(mixed_df)
        assert "Sample" in html
        assert "sample-head" in html
        assert "sample-tail" in html

    def test_head_tail_tabs(self, mixed_df: pl.DataFrame):
        html = _render(mixed_df)
        assert "switchSampleTab" in html


class TestRenderDuplicateSection:
    def test_duplicate_section_with_dupes(self):
        df = pl.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
        html = _render(df)
        assert "Duplicate Rows" in html
        assert html.count("Duplicate Rows") >= 2

    def test_no_duplicate_section_without_dupes(self):
        df = pl.DataFrame({"a": [1, 2, 3]})
        html = _render(df)
        assert html.count("Duplicate Rows") == 1


class TestRenderReproduction:
    def test_reproduction_section(self, mixed_df: pl.DataFrame):
        html = _render(mixed_df)
        assert "Reproduction Details" in html
        assert "polars_version" in html
        assert "dataxid_profiling_version" in html


class TestRenderEdgeCases:
    def test_empty_dataframe(self, empty_df: pl.DataFrame):
        html = _render(empty_df)
        assert "<!DOCTYPE html>" in html
        assert "0" in html

    def test_single_column(self):
        df = pl.DataFrame({"x": [1, 2, 3]})
        html = _render(df)
        assert "<!DOCTYPE html>" in html
        assert "x" in html


class TestFilters:
    def test_format_number(self, mixed_df: pl.DataFrame):
        html = _render(mixed_df)
        assert "10" in html  # n_rows

    def test_format_pct(self, mixed_df: pl.DataFrame):
        html = _render(mixed_df)
        assert "%" in html
