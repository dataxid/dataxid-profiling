from __future__ import annotations

import json
from typing import TYPE_CHECKING

import polars as pl
import pytest

from dataxid_profiling import AlertType, ProfileConfig, ProfileReport

if TYPE_CHECKING:
    from pathlib import Path


class TestProfileReportCreation:
    def test_from_dataframe(self, mixed_df: pl.DataFrame):
        report = ProfileReport(mixed_df)
        assert report.df.shape == mixed_df.shape

    def test_from_csv(self, tmp_path: Path, mixed_df: pl.DataFrame):
        csv_path = tmp_path / "data.csv"
        mixed_df.write_csv(csv_path)
        report = ProfileReport(str(csv_path))
        assert report.df.shape == mixed_df.shape

    def test_custom_title(self, mixed_df: pl.DataFrame):
        report = ProfileReport(mixed_df, title="My Report")
        assert report.config.title == "My Report"

    def test_with_config(self, mixed_df: pl.DataFrame):
        cfg = ProfileConfig(title="Custom", missing_threshold=0.1)
        report = ProfileReport(mixed_df, config=cfg)
        assert report.config.missing_threshold == 0.1

    def test_repr(self, mixed_df: pl.DataFrame):
        report = ProfileReport(mixed_df, title="Test")
        assert "Test" in repr(report)
        assert "rows=10" in repr(report)


class TestProfileReportStats:
    def test_stats_keys_match_columns(self, mixed_df: pl.DataFrame):
        report = ProfileReport(mixed_df)
        for col in mixed_df.columns:
            assert col in report.stats

    def test_numeric_stats_have_mean(self, mixed_df: pl.DataFrame):
        report = ProfileReport(mixed_df)
        assert "mean" in report.stats["age"]
        assert report.stats["age"]["mean"] is not None

    def test_categorical_stats_have_top_values(self, mixed_df: pl.DataFrame):
        report = ProfileReport(mixed_df)
        assert "top_values" in report.stats["city"]

    def test_boolean_stats_have_true_pct(self, mixed_df: pl.DataFrame):
        report = ProfileReport(mixed_df)
        assert "true_pct" in report.stats["active"]

    def test_datetime_stats_have_min_max(self, mixed_df: pl.DataFrame):
        report = ProfileReport(mixed_df)
        assert "min" in report.stats["signup_date"]
        assert "max" in report.stats["signup_date"]


class TestProfileReportOverview:
    def test_overview_keys(self, mixed_df: pl.DataFrame):
        report = ProfileReport(mixed_df)
        ov = report.overview
        assert ov["n_rows"] == 10
        assert ov["n_columns"] == 7
        assert "missing_cells" in ov
        assert "duplicate_rows" in ov
        assert "memory_bytes" in ov
        assert "type_distribution" in ov


class TestProfileReportAlerts:
    def test_alerts_is_list(self, mixed_df: pl.DataFrame):
        report = ProfileReport(mixed_df)
        assert isinstance(report.alerts, list)

    def test_high_missing_detected(self):
        df = pl.DataFrame({"a": [1, None, None, None, None]})
        report = ProfileReport(df, missing_threshold=0.05)
        alert_types = {a.alert_type for a in report.alerts}
        assert AlertType.HIGH_MISSING in alert_types


class TestProfileReportToDict:
    def test_to_dict_structure(self, mixed_df: pl.DataFrame):
        report = ProfileReport(mixed_df)
        d = report.to_dict()
        assert "title" in d
        assert "overview" in d
        assert "columns" in d
        assert "alerts" in d

    def test_to_dict_columns_match(self, mixed_df: pl.DataFrame):
        report = ProfileReport(mixed_df)
        d = report.to_dict()
        assert set(d["columns"].keys()) == set(mixed_df.columns)

    def test_to_dict_alerts_serializable(self, mixed_df: pl.DataFrame):
        report = ProfileReport(mixed_df)
        d = report.to_dict()
        for alert in d["alerts"]:
            assert "column" in alert
            assert "alert_type" in alert
            assert isinstance(alert["alert_type"], str)
            assert "value" in alert


class TestProfileReportToJson:
    def test_to_json_returns_string(self, mixed_df: pl.DataFrame):
        report = ProfileReport(mixed_df)
        output = report.to_json()
        assert isinstance(output, str)

    def test_to_json_valid_json(self, mixed_df: pl.DataFrame):
        report = ProfileReport(mixed_df)
        output = report.to_json()
        parsed = json.loads(output)
        assert "title" in parsed
        assert "columns" in parsed

    def test_to_json_writes_file(self, tmp_path: Path, mixed_df: pl.DataFrame):
        report = ProfileReport(mixed_df)
        json_path = tmp_path / "report.json"
        report.to_json(path=json_path)
        assert json_path.exists()
        parsed = json.loads(json_path.read_text())
        assert parsed["overview"]["n_rows"] == 10

    def test_to_json_column_type_is_string(self, mixed_df: pl.DataFrame):
        report = ProfileReport(mixed_df)
        output = report.to_json()
        parsed = json.loads(output)
        for col_data in parsed["columns"].values():
            assert isinstance(col_data["column_type"], str)


class TestProfileReportCorrelations:
    def test_correlations_property(self, mixed_df: pl.DataFrame):
        report = ProfileReport(mixed_df)
        assert isinstance(report.correlations, dict)
        assert "pearson" in report.correlations
        assert "spearman" in report.correlations
        assert "kendall" in report.correlations

    def test_correlations_in_to_dict(self, mixed_df: pl.DataFrame):
        report = ProfileReport(mixed_df)
        d = report.to_dict()
        assert "correlations" in d
        assert "pearson" in d["correlations"]
        assert "matrix" in d["correlations"]["pearson"]

    def test_no_correlations_overview(self, mixed_df: pl.DataFrame):
        report = ProfileReport(mixed_df, mode="overview")
        assert report.correlations == {}

    def test_cramers_v_with_categoricals(self):
        df = pl.DataFrame({
            "a": ["x", "y", "z"] * 10,
            "b": ["p", "q", "r"] * 10,
        })
        report = ProfileReport(df)
        assert "cramers_v" in report.correlations

    def test_cramers_v_in_to_dict(self):
        df = pl.DataFrame({
            "a": ["x", "y", "z"] * 10,
            "b": ["p", "q", "r"] * 10,
        })
        report = ProfileReport(df)
        d = report.to_dict()
        assert "cramers_v" in d["correlations"]
        assert "matrix" in d["correlations"]["cramers_v"]

    def test_no_correlations_single_numeric(self):
        df = pl.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
        report = ProfileReport(df)
        assert "pearson" not in report.correlations
        assert "cramers_v" not in report.correlations


class TestProfileReportToHtml:
    def test_to_html_returns_string(self, mixed_df: pl.DataFrame):
        report = ProfileReport(mixed_df)
        html = report.to_html()
        assert isinstance(html, str)
        assert "<!DOCTYPE html>" in html

    def test_to_html_contains_title(self, mixed_df: pl.DataFrame):
        report = ProfileReport(mixed_df, title="HTML Test")
        html = report.to_html()
        assert "HTML Test" in html

    def test_to_html_writes_file(self, tmp_path: Path, mixed_df: pl.DataFrame):
        report = ProfileReport(mixed_df)
        html_path = tmp_path / "report.html"
        report.to_html(path=html_path)
        assert html_path.exists()
        content = html_path.read_text()
        assert "<!DOCTYPE html>" in content

    def test_to_html_contains_columns(self, mixed_df: pl.DataFrame):
        report = ProfileReport(mixed_df)
        html = report.to_html()
        assert "age" in html
        assert "salary" in html
        assert "city" in html

    def test_to_html_contains_charts(self, mixed_df: pl.DataFrame):
        report = ProfileReport(mixed_df)
        html = report.to_html()
        assert "echarts.init" in html

    def test_to_html_contains_correlations(self, mixed_df: pl.DataFrame):
        report = ProfileReport(mixed_df)
        html = report.to_html()
        assert "corr_pearson" in html
        assert "corr_spearman" in html
        assert "corr_kendall" in html

    def test_to_html_no_correlation_overview(self, mixed_df: pl.DataFrame):
        report = ProfileReport(mixed_df, mode="overview")
        html = report.to_html()
        assert "corr_pearson" not in html


class TestProfileReportEdgeCases:
    def test_empty_dataframe(self, empty_df: pl.DataFrame):
        report = ProfileReport(empty_df)
        assert report.overview["n_rows"] == 0
        assert len(report.alerts) == 0

    def test_single_row(self):
        df = pl.DataFrame({"a": [42], "b": ["hello"], "c": [True]})
        report = ProfileReport(df)
        assert report.overview["n_rows"] == 1
        assert report.stats["a"]["mean"] == pytest.approx(42.0)
