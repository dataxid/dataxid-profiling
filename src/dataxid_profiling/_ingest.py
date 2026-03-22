from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl


def ingest(source: Any) -> pl.DataFrame:
    """Accept various input types and return a Polars DataFrame.

    Supported:
        - pl.DataFrame (passthrough)
        - pl.LazyFrame (.collect())
        - pd.DataFrame (auto-convert via pl.from_pandas)
        - str / Path to CSV or Parquet file
    """
    if isinstance(source, pl.DataFrame):
        return source

    if isinstance(source, pl.LazyFrame):
        return source.collect()

    if _is_pandas_dataframe(source):
        return _from_pandas(source)

    if isinstance(source, (str, Path)):
        return _from_path(Path(source))

    type_name = type(source).__qualname__
    msg = (
        f"Unsupported input type: {type_name}. "
        "Expected pl.DataFrame, pl.LazyFrame, pd.DataFrame, or a file path (str/Path)."
    )
    raise TypeError(msg)


def _is_pandas_dataframe(obj: Any) -> bool:
    try:
        import pandas as pd
    except ImportError:
        return False
    return isinstance(obj, pd.DataFrame)


def _from_pandas(pdf: Any) -> pl.DataFrame:
    try:
        return pl.from_pandas(pdf)
    except Exception as exc:
        msg = f"Failed to convert Pandas DataFrame to Polars: {exc}"
        raise ValueError(msg) from exc


def _from_path(path: Path) -> pl.DataFrame:
    if not path.exists():
        msg = f"File not found: {path}"
        raise FileNotFoundError(msg)

    suffix = path.suffix.lower()

    readers: dict[str, Any] = {
        ".csv": pl.read_csv,
        ".tsv": lambda p: pl.read_csv(p, separator="\t"),
        ".parquet": pl.read_parquet,
    }

    reader = readers.get(suffix)
    if reader is None:
        supported = ", ".join(sorted(readers.keys()))
        msg = f"Unsupported file format: '{suffix}'. Supported: {supported}"
        raise ValueError(msg)

    return reader(path)
