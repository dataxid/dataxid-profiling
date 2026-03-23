from __future__ import annotations

import polars as pl
import pytest

from dataxid_profiling._config import ProfileConfig
from dataxid_profiling._dataset_overview import DatasetOverview, compute_overview
from dataxid_profiling._type_inference import infer_types


def _overview(
    df: pl.DataFrame, config: ProfileConfig | None = None
) -> DatasetOverview:
    config = config or ProfileConfig()
    column_types = infer_types(df, config)
    return compute_overview(df, column_types, config)


class TestMissingPerColumn:
    def test_missing_per_column_counts(self):
        df = pl.DataFrame({"a": [1, None, 3], "b": [None, None, "x"]})
        ov = _overview(df)
        assert ov.missing_per_column["a"]["count"] == 1
        assert ov.missing_per_column["b"]["count"] == 2

    def test_missing_per_column_pct(self):
        df = pl.DataFrame({"a": [1, None, 3], "b": [None, None, "x"]})
        ov = _overview(df)
        assert ov.missing_per_column["a"]["pct"] == pytest.approx(1 / 3)
        assert ov.missing_per_column["b"]["pct"] == pytest.approx(2 / 3)

    def test_no_missing(self):
        df = pl.DataFrame({"a": [1, 2, 3]})
        ov = _overview(df)
        assert ov.missing_per_column["a"]["count"] == 0
        assert ov.missing_per_column["a"]["pct"] == 0.0

    def test_empty_df(self, empty_df: pl.DataFrame):
        ov = _overview(empty_df)
        assert ov.missing_per_column == {}


class TestSampleHeadTail:
    def test_sample_head(self, mixed_df: pl.DataFrame):
        ov = _overview(mixed_df)
        assert len(ov.sample_head) == 10
        assert "id" in ov.sample_head[0]

    def test_sample_tail(self, mixed_df: pl.DataFrame):
        ov = _overview(mixed_df)
        assert len(ov.sample_tail) == 10

    def test_small_df(self):
        df = pl.DataFrame({"a": [1, 2]})
        ov = _overview(df)
        assert len(ov.sample_head) == 2
        assert len(ov.sample_tail) == 2

    def test_empty_df(self, empty_df: pl.DataFrame):
        ov = _overview(empty_df)
        assert ov.sample_head == []
        assert ov.sample_tail == []

    def test_values_are_strings(self):
        df = pl.DataFrame({"a": [1, 2], "b": [3.14, 2.71]})
        ov = _overview(df)
        assert all(isinstance(v, str) for v in ov.sample_head[0].values())

    def test_null_preserved(self):
        df = pl.DataFrame({"a": [1, None]})
        ov = _overview(df)
        values = [row["a"] for row in ov.sample_head]
        assert None in values


class TestDuplicateRowsSample:
    def test_with_duplicates(self):
        df = pl.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
        ov = _overview(df)
        assert len(ov.duplicate_rows_sample) > 0

    def test_no_duplicates(self):
        df = pl.DataFrame({"a": [1, 2, 3]})
        ov = _overview(df)
        assert ov.duplicate_rows_sample == []

    def test_overview_mode_skips_sample(self):
        df = pl.DataFrame({"a": [1, 1, 2], "b": ["x", "x", "y"]})
        config = ProfileConfig(mode="overview")
        ov = _overview(df, config)
        assert ov.duplicate_rows_sample == []

    def test_sample_max_rows(self):
        rows = [{"a": 1, "b": "x"}] * 50
        df = pl.DataFrame(rows)
        ov = _overview(df)
        assert len(ov.duplicate_rows_sample) <= 10


class TestReproduction:
    def test_has_keys(self, mixed_df: pl.DataFrame):
        ov = _overview(mixed_df)
        assert "dataxid_profiling_version" in ov.reproduction
        assert "polars_version" in ov.reproduction
        assert "generated_at" in ov.reproduction
        assert "mode" in ov.reproduction

    def test_mode_value(self):
        df = pl.DataFrame({"a": [1]})
        ov = _overview(df, ProfileConfig(mode="overview"))
        assert ov.reproduction["mode"] == "overview"

    def test_complete_mode(self):
        df = pl.DataFrame({"a": [1]})
        ov = _overview(df)
        assert ov.reproduction["mode"] == "complete"
