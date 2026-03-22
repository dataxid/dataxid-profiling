from __future__ import annotations

import polars as pl
import pytest

from dataxid_profiling._alerts import Alert, AlertType, check_quality
from dataxid_profiling._analyzers import analyze
from dataxid_profiling._config import ProfileConfig
from dataxid_profiling._dataset_overview import compute_overview
from dataxid_profiling._type_inference import infer_types


def _get_alerts(
    df: pl.DataFrame, config: ProfileConfig | None = None
) -> list[Alert]:
    config = config or ProfileConfig()
    column_types = infer_types(df, config)
    column_stats = analyze(df, column_types, config)
    overview = compute_overview(df, column_types, config)
    return check_quality(column_stats, overview, config)


def _alert_types(alerts: list[Alert]) -> set[AlertType]:
    return {a.alert_type for a in alerts}


def _alerts_for_column(alerts: list[Alert], col: str) -> list[Alert]:
    return [a for a in alerts if a.column == col]


class TestHighMissing:
    def test_triggers_above_threshold(self):
        df = pl.DataFrame({"a": [1, None, None, None, None]})
        alerts = _get_alerts(df, ProfileConfig(missing_threshold=0.05))
        col_alerts = _alerts_for_column(alerts, "a")
        assert any(a.alert_type == AlertType.HIGH_MISSING for a in col_alerts)

    def test_no_alert_below_threshold(self):
        df = pl.DataFrame({"a": [1, 2, 3, 4, None]})
        alerts = _get_alerts(df, ProfileConfig(missing_threshold=0.5))
        col_alerts = _alerts_for_column(alerts, "a")
        assert not any(a.alert_type == AlertType.HIGH_MISSING for a in col_alerts)

    def test_missing_value_correct(self):
        df = pl.DataFrame({"a": [1, None, None, 4, 5]})
        alerts = _get_alerts(df, ProfileConfig(missing_threshold=0.05))
        missing_alerts = [
            a for a in alerts
            if a.alert_type == AlertType.HIGH_MISSING and a.column == "a"
        ]
        assert len(missing_alerts) == 1
        assert missing_alerts[0].value == pytest.approx(0.4)


class TestConstant:
    def test_single_value_triggers(self):
        df = pl.DataFrame({"a": [42, 42, 42, 42, 42]})
        alerts = _get_alerts(df)
        col_alerts = _alerts_for_column(alerts, "a")
        assert any(a.alert_type == AlertType.CONSTANT for a in col_alerts)

    def test_multiple_values_no_alert(self):
        df = pl.DataFrame({"a": [1, 2, 3]})
        alerts = _get_alerts(df)
        col_alerts = _alerts_for_column(alerts, "a")
        assert not any(a.alert_type == AlertType.CONSTANT for a in col_alerts)


class TestHighCardinality:
    def test_numeric_high_cardinality(self):
        df = pl.DataFrame({"a": list(range(100))})
        alerts = _get_alerts(df, ProfileConfig(cardinality_threshold=0.95))
        col_alerts = _alerts_for_column(alerts, "a")
        assert any(a.alert_type == AlertType.HIGH_CARDINALITY for a in col_alerts)

    def test_categorical_high_cardinality(self):
        df = pl.DataFrame({"a": [f"val_{i}" for i in range(20)]})
        alerts = _get_alerts(df, ProfileConfig(cardinality_threshold=0.95, text_unique_ratio=1.0))
        col_alerts = _alerts_for_column(alerts, "a")
        assert any(a.alert_type == AlertType.HIGH_CARDINALITY for a in col_alerts)


class TestDuplicates:
    def test_duplicates_trigger(self):
        df = pl.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
        alerts = _get_alerts(df)
        assert any(a.alert_type == AlertType.DUPLICATES for a in alerts)

    def test_no_duplicates(self):
        df = pl.DataFrame({"a": [1, 2, 3]})
        alerts = _get_alerts(df)
        assert not any(a.alert_type == AlertType.DUPLICATES for a in alerts)

    def test_duplicate_is_dataset_level(self):
        df = pl.DataFrame({"a": [1, 1, 2]})
        dup_alerts = [a for a in _get_alerts(df) if a.alert_type == AlertType.DUPLICATES]
        assert all(a.column is None for a in dup_alerts)


class TestHighZeros:
    def test_zeros_trigger(self):
        df = pl.DataFrame({"a": [0, 0, 0, 1, 2]})
        alerts = _get_alerts(df, ProfileConfig(zero_threshold=0.05))
        col_alerts = _alerts_for_column(alerts, "a")
        assert any(a.alert_type == AlertType.HIGH_ZEROS for a in col_alerts)

    def test_no_zeros_no_alert(self):
        df = pl.DataFrame({"a": [1, 2, 3]})
        alerts = _get_alerts(df)
        col_alerts = _alerts_for_column(alerts, "a")
        assert not any(a.alert_type == AlertType.HIGH_ZEROS for a in col_alerts)


class TestSkewed:
    def test_skewed_triggers(self):
        # Highly right-skewed distribution
        df = pl.DataFrame({"a": [1] * 90 + [1000] * 10})
        alerts = _get_alerts(df, ProfileConfig(skewness_threshold=2.0))
        col_alerts = _alerts_for_column(alerts, "a")
        assert any(a.alert_type == AlertType.SKEWED for a in col_alerts)

    def test_symmetric_no_alert(self):
        df = pl.DataFrame({"a": list(range(1, 101))})
        alerts = _get_alerts(df, ProfileConfig(skewness_threshold=2.0))
        col_alerts = _alerts_for_column(alerts, "a")
        assert not any(a.alert_type == AlertType.SKEWED for a in col_alerts)


class TestImbalanced:
    def test_boolean_imbalanced(self):
        df = pl.DataFrame({"a": [True] * 95 + [False] * 5})
        alerts = _get_alerts(df, ProfileConfig(imbalance_threshold=0.9))
        col_alerts = _alerts_for_column(alerts, "a")
        assert any(a.alert_type == AlertType.IMBALANCED for a in col_alerts)

    def test_boolean_balanced_no_alert(self):
        df = pl.DataFrame({"a": [True, False, True, False]})
        alerts = _get_alerts(df, ProfileConfig(imbalance_threshold=0.9))
        col_alerts = _alerts_for_column(alerts, "a")
        assert not any(a.alert_type == AlertType.IMBALANCED for a in col_alerts)

    def test_categorical_imbalanced(self):
        df = pl.DataFrame({"a": ["x"] * 95 + ["y"] * 5})
        alerts = _get_alerts(df, ProfileConfig(imbalance_threshold=0.9, text_unique_ratio=1.0))
        col_alerts = _alerts_for_column(alerts, "a")
        assert any(a.alert_type == AlertType.IMBALANCED for a in col_alerts)


class TestCleanData:
    def test_no_alerts_on_clean_data(self):
        df = pl.DataFrame({
            "score": [10, 20, 30, 40, 50, 10, 20, 30, 40, 50],
            "city": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"],
            "active": [True, False, True, False, True, False, True, False, True, False],
        })
        alerts = _get_alerts(df)
        # duplicate rows exist but that's expected — only column-level alerts checked
        col_alerts = [a for a in alerts if a.column is not None]
        assert len(col_alerts) == 0
