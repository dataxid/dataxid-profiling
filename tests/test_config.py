from __future__ import annotations

import pytest

from dataxid_profiling._config import ProfileConfig


class TestProfileConfigDefaults:
    def test_default_title(self):
        cfg = ProfileConfig()
        assert cfg.title == "Dataset Report"

    def test_default_thresholds(self):
        cfg = ProfileConfig()
        assert cfg.missing_threshold == 0.05
        assert cfg.cardinality_threshold == 0.95
        assert cfg.correlation_threshold == 0.9
        assert cfg.text_unique_ratio == 0.5

    def test_default_mode_complete(self):
        cfg = ProfileConfig()
        assert cfg.mode == "complete"
        assert cfg.is_overview is False

    def test_default_display(self):
        cfg = ProfileConfig()
        assert cfg.n_top_values == 5
        assert cfg.histogram_bins == 50


class TestProfileConfigCustom:
    def test_custom_title(self):
        cfg = ProfileConfig(title="My Report")
        assert cfg.title == "My Report"

    def test_custom_thresholds(self):
        cfg = ProfileConfig(missing_threshold=0.1, cardinality_threshold=0.8)
        assert cfg.missing_threshold == 0.1
        assert cfg.cardinality_threshold == 0.8

    def test_overview_mode(self):
        cfg = ProfileConfig(mode="overview")
        assert cfg.mode == "overview"
        assert cfg.is_overview is True

    def test_column_overrides(self):
        overrides = {"age": {"missing_threshold": 0.2}}
        cfg = ProfileConfig(column_overrides=overrides)
        assert cfg.column_overrides["age"]["missing_threshold"] == 0.2


class TestProfileConfigImmutable:
    def test_frozen(self):
        cfg = ProfileConfig()
        with pytest.raises(AttributeError):
            cfg.title = "Changed"  # type: ignore[misc]


class TestProfileConfigValidation:
    def test_invalid_text_unique_ratio(self):
        with pytest.raises(ValueError, match="text_unique_ratio"):
            ProfileConfig(text_unique_ratio=1.5)

    def test_invalid_missing_threshold(self):
        with pytest.raises(ValueError, match="missing_threshold"):
            ProfileConfig(missing_threshold=-0.1)

    def test_invalid_cardinality_threshold(self):
        with pytest.raises(ValueError, match="cardinality_threshold"):
            ProfileConfig(cardinality_threshold=2.0)

    def test_invalid_n_top_values(self):
        with pytest.raises(ValueError, match="n_top_values"):
            ProfileConfig(n_top_values=0)

    def test_invalid_histogram_bins(self):
        with pytest.raises(ValueError, match="histogram_bins"):
            ProfileConfig(histogram_bins=1)

    def test_invalid_mode(self):
        with pytest.raises(ValueError, match="mode"):
            ProfileConfig(mode="invalid")
