from __future__ import annotations

import polars as pl
import pytest


@pytest.fixture
def numeric_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "age": [25, 30, 35, 40, None, 50],
            "salary": [50_000.0, 60_000.0, None, 80_000.0, 90_000.0, 100_000.0],
            "score": [85, 90, 78, 92, 88, None],
        }
    )


@pytest.fixture
def categorical_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "city": [
                "Istanbul", "Ankara", "Istanbul", "Izmir", "Ankara",
                "Istanbul", "Ankara", "Izmir", None, "Istanbul",
            ],
            "color": [
                "red", "blue", "red", "green", "blue",
                "red", "green", "blue", None, "red",
            ],
        }
    )


@pytest.fixture
def boolean_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "active": [True, False, True, True, None, False],
            "verified": [True, True, False, None, True, True],
        }
    )


@pytest.fixture
def datetime_df() -> pl.DataFrame:
    from datetime import date, datetime

    return pl.DataFrame(
        {
            "created_at": [
                datetime(2024, 1, 1, 10, 0),
                datetime(2024, 2, 15, 14, 30),
                datetime(2024, 3, 20, 9, 15),
                None,
                datetime(2024, 5, 10, 16, 45),
                datetime(2024, 6, 1, 8, 0),
            ],
            "birth_date": [
                date(1990, 5, 15),
                date(1985, 8, 22),
                None,
                date(1992, 12, 1),
                date(1988, 3, 10),
                date(1995, 7, 30),
            ],
        }
    )


@pytest.fixture
def mixed_df() -> pl.DataFrame:
    """Tüm kolon tiplerini içeren gerçekçi bir DataFrame."""
    from datetime import date

    return pl.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "name": [
                "Alice", "Bob", "Charlie", "Diana", "Eve",
                "Frank", "Grace", "Hank", "Ivy", "Jack",
            ],
            "age": [25, 30, None, 40, 35, 28, 45, None, 33, 27],
            "salary": [
                50_000.0, 60_000.0, 70_000.0, None, 90_000.0,
                55_000.0, 80_000.0, 65_000.0, 72_000.0, 48_000.0,
            ],
            "city": [
                "Istanbul", "Ankara", "Istanbul", "Izmir", "Ankara",
                "Istanbul", "Izmir", "Ankara", "Istanbul", None,
            ],
            "active": [True, False, True, True, None, False, True, True, False, True],
            "signup_date": [
                date(2023, 1, 15), date(2023, 2, 20), date(2023, 3, 10),
                date(2023, 4, 5), date(2023, 5, 12), date(2023, 6, 1),
                date(2023, 7, 18), None, date(2023, 9, 3), date(2023, 10, 25),
            ],
        }
    )


@pytest.fixture
def empty_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "a": pl.Series([], dtype=pl.Int64),
            "b": pl.Series([], dtype=pl.Utf8),
        }
    )


@pytest.fixture
def constant_df() -> pl.DataFrame:
    """Tek değer içeren kolonlar — CONSTANT alert testi için."""
    return pl.DataFrame(
        {
            "status": ["active"] * 10,
            "flag": [True] * 10,
            "value": [42] * 10,
        }
    )


@pytest.fixture
def high_cardinality_df() -> pl.DataFrame:
    """Her satır unique — HIGH_CARDINALITY alert testi için."""
    return pl.DataFrame(
        {
            "user_id": [f"user_{i}" for i in range(100)],
            "email": [f"user{i}@example.com" for i in range(100)],
        }
    )
