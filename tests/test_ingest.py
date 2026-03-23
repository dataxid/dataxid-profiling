from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from dataxid_profiling._ingest import ingest


class TestPolarsInput:
    def test_dataframe_passthrough(self, numeric_df: pl.DataFrame):
        result = ingest(numeric_df)
        assert result is numeric_df

    def test_lazyframe_collect(self, numeric_df: pl.DataFrame):
        lazy = numeric_df.lazy()
        result = ingest(lazy)
        assert isinstance(result, pl.DataFrame)
        assert result.shape == numeric_df.shape

    def test_empty_dataframe(self, empty_df: pl.DataFrame):
        result = ingest(empty_df)
        assert result.shape == (0, 2)


class TestPandasInput:
    def test_pandas_auto_convert(self, numeric_df: pl.DataFrame):
        pytest.importorskip("pyarrow")
        pd = pytest.importorskip("pandas")
        pdf = numeric_df.to_pandas()
        assert isinstance(pdf, pd.DataFrame)

        result = ingest(pdf)
        assert isinstance(result, pl.DataFrame)
        assert result.shape == numeric_df.shape

    def test_pandas_column_names_preserved(self):
        pytest.importorskip("pyarrow")
        pd = pytest.importorskip("pandas")
        pdf = pd.DataFrame({"col_a": [1, 2], "col_b": ["x", "y"]})
        result = ingest(pdf)
        assert result.columns == ["col_a", "col_b"]


class TestFileInput:
    def test_csv_file(self, tmp_path: Path, mixed_df: pl.DataFrame):
        csv_path = tmp_path / "data.csv"
        mixed_df.write_csv(csv_path)

        result = ingest(csv_path)
        assert result.shape == mixed_df.shape
        assert result.columns == mixed_df.columns

    def test_csv_string_path(self, tmp_path: Path, numeric_df: pl.DataFrame):
        csv_path = tmp_path / "data.csv"
        numeric_df.write_csv(csv_path)

        result = ingest(str(csv_path))
        assert isinstance(result, pl.DataFrame)
        assert result.shape == numeric_df.shape

    def test_tsv_file(self, tmp_path: Path, numeric_df: pl.DataFrame):
        tsv_path = tmp_path / "data.tsv"
        numeric_df.write_csv(tsv_path, separator="\t")

        result = ingest(tsv_path)
        assert result.shape == numeric_df.shape

    def test_parquet_file(self, tmp_path: Path, mixed_df: pl.DataFrame):
        parquet_path = tmp_path / "data.parquet"
        mixed_df.write_parquet(parquet_path)

        result = ingest(parquet_path)
        assert result.shape == mixed_df.shape

    def test_file_not_found(self):
        with pytest.raises(FileNotFoundError, match="File not found"):
            ingest(Path("/nonexistent/data.csv"))

    def test_unsupported_format(self, tmp_path: Path):
        xlsx_path = tmp_path / "data.xlsx"
        xlsx_path.touch()
        with pytest.raises(ValueError, match="Unsupported file format"):
            ingest(xlsx_path)


class TestInvalidInput:
    def test_int_raises(self):
        with pytest.raises(TypeError, match="Unsupported input type"):
            ingest(42)

    def test_list_raises(self):
        with pytest.raises(TypeError, match="Unsupported input type"):
            ingest([1, 2, 3])

    def test_none_raises(self):
        with pytest.raises(TypeError, match="Unsupported input type"):
            ingest(None)

    def test_dict_raises(self):
        with pytest.raises(TypeError, match="Unsupported input type"):
            ingest({"a": [1, 2]})
