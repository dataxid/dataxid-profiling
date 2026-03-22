from __future__ import annotations

import polars as pl

from dataxid_profiling._config import ProfileConfig
from dataxid_profiling._type_inference import ColumnType, infer_types


class TestNumericInference:
    def test_int_columns(self, numeric_df: pl.DataFrame):
        types = infer_types(numeric_df)
        assert types["age"] == ColumnType.NUMERIC
        assert types["score"] == ColumnType.NUMERIC

    def test_float_columns(self, numeric_df: pl.DataFrame):
        types = infer_types(numeric_df)
        assert types["salary"] == ColumnType.NUMERIC

    def test_all_numeric_dtypes(self):
        df = pl.DataFrame(
            {
                "i8": pl.Series([1], dtype=pl.Int8),
                "i16": pl.Series([1], dtype=pl.Int16),
                "i32": pl.Series([1], dtype=pl.Int32),
                "i64": pl.Series([1], dtype=pl.Int64),
                "u8": pl.Series([1], dtype=pl.UInt8),
                "u16": pl.Series([1], dtype=pl.UInt16),
                "u32": pl.Series([1], dtype=pl.UInt32),
                "u64": pl.Series([1], dtype=pl.UInt64),
                "f32": pl.Series([1.0], dtype=pl.Float32),
                "f64": pl.Series([1.0], dtype=pl.Float64),
            }
        )
        types = infer_types(df)
        assert all(t == ColumnType.NUMERIC for t in types.values())


class TestCategoricalInference:
    def test_low_cardinality_string(self, categorical_df: pl.DataFrame):
        types = infer_types(categorical_df)
        assert types["city"] == ColumnType.CATEGORICAL
        assert types["color"] == ColumnType.CATEGORICAL

    def test_categorical_dtype(self):
        df = pl.DataFrame({"cat": pl.Series(["a", "b", "a"]).cast(pl.Categorical)})
        types = infer_types(df)
        assert types["cat"] == ColumnType.CATEGORICAL


class TestTextInference:
    def test_high_cardinality_string(self, high_cardinality_df: pl.DataFrame):
        types = infer_types(high_cardinality_df)
        assert types["user_id"] == ColumnType.TEXT
        assert types["email"] == ColumnType.TEXT

    def test_custom_threshold(self):
        df = pl.DataFrame({"name": ["Alice", "Bob", "Charlie", "Diana", "Eve"]})
        # 5 unique / 5 rows = 1.0 > 0.5 → TEXT
        assert infer_types(df)["name"] == ColumnType.TEXT
        # threshold 1.0 → unique_ratio must exceed 1.0, impossible → CATEGORICAL
        config = ProfileConfig(text_unique_ratio=1.0)
        assert infer_types(df, config)["name"] == ColumnType.CATEGORICAL


class TestBooleanInference:
    def test_boolean_columns(self, boolean_df: pl.DataFrame):
        types = infer_types(boolean_df)
        assert types["active"] == ColumnType.BOOLEAN
        assert types["verified"] == ColumnType.BOOLEAN


class TestDatetimeInference:
    def test_datetime_columns(self, datetime_df: pl.DataFrame):
        types = infer_types(datetime_df)
        assert types["created_at"] == ColumnType.DATETIME
        assert types["birth_date"] == ColumnType.DATETIME

    def test_time_and_duration(self):
        from datetime import time, timedelta

        df = pl.DataFrame(
            {
                "t": [time(10, 0), time(14, 30)],
                "dur": [timedelta(hours=1), timedelta(hours=2)],
            }
        )
        types = infer_types(df)
        assert types["t"] == ColumnType.DATETIME
        assert types["dur"] == ColumnType.DATETIME


class TestMixedDataFrame:
    def test_mixed_types(self, mixed_df: pl.DataFrame):
        types = infer_types(mixed_df)
        assert types["id"] == ColumnType.NUMERIC
        assert types["age"] == ColumnType.NUMERIC
        assert types["salary"] == ColumnType.NUMERIC
        assert types["city"] == ColumnType.CATEGORICAL
        assert types["active"] == ColumnType.BOOLEAN
        assert types["signup_date"] == ColumnType.DATETIME

    def test_name_is_text(self, mixed_df: pl.DataFrame):
        types = infer_types(mixed_df)
        # 10 unique names / 10 rows = 1.0 > 0.5 → TEXT
        assert types["name"] == ColumnType.TEXT


class TestEdgeCases:
    def test_empty_dataframe(self, empty_df: pl.DataFrame):
        types = infer_types(empty_df)
        assert types["a"] == ColumnType.NUMERIC
        assert types["b"] == ColumnType.CATEGORICAL

    def test_constant_columns(self, constant_df: pl.DataFrame):
        types = infer_types(constant_df)
        assert types["status"] == ColumnType.CATEGORICAL
        assert types["flag"] == ColumnType.BOOLEAN
        assert types["value"] == ColumnType.NUMERIC

    def test_unsupported_dtype(self):
        df = pl.DataFrame({"bin": [b"hello", b"world"]})
        types = infer_types(df)
        assert types["bin"] == ColumnType.UNSUPPORTED

    def test_default_config_when_none(self, numeric_df: pl.DataFrame):
        types = infer_types(numeric_df, config=None)
        assert types["age"] == ColumnType.NUMERIC
