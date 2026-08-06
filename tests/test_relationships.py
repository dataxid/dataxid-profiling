import polars as pl

from dataxid_profiling._relationships import (
    compute_joint_histogram,
    compute_joint_scatter,
    compute_pair_boxplot,
)
from dataxid_profiling._type_inference import ColumnType


def _grid_count_total(result):
    return sum(c["count"] for c in result["cells"])


def test_numeric_numeric_grid_counts_rows():
    df = pl.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [10.0, 20.0, 30.0, 40.0]})
    types = {"a": ColumnType.NUMERIC, "b": ColumnType.NUMERIC}
    result = compute_joint_histogram(df, "a", "b", types, bins=2)
    assert result["kind"] == "grid"
    assert result["x"]["column"] == "a"
    assert result["x"]["type"] == "NUMERIC"
    assert result["y"]["type"] == "NUMERIC"
    assert len(result["x"]["bins"]) == 2
    assert len(result["y"]["bins"]) == 2
    assert _grid_count_total(result) == 4
    # positive relationship: mass on the diagonal, off-diagonal empty
    diag = {(c["xi"], c["yi"]) for c in result["cells"]}
    assert (0, 1) not in diag and (1, 0) not in diag


def test_categorical_categorical_contingency():
    df = pl.DataFrame({"c": ["x", "x", "y", "y"], "d": ["p", "q", "p", "p"]})
    types = {"c": ColumnType.CATEGORICAL, "d": ColumnType.CATEGORICAL}
    result = compute_joint_histogram(df, "c", "d", types)
    assert result["kind"] == "grid"
    assert result["x"]["type"] == "CATEGORICAL"
    assert set(result["x"]["bins"]) == {"x", "y"}
    assert _grid_count_total(result) == 4


def test_rows_with_nulls_dropped_pairwise():
    df = pl.DataFrame({"a": [1.0, None, 3.0], "b": [10.0, 20.0, None]})
    types = {"a": ColumnType.NUMERIC, "b": ColumnType.NUMERIC}
    result = compute_joint_histogram(df, "a", "b", types, bins=2)
    # only the (1.0, 10.0) row survives pairwise drop_nulls
    assert _grid_count_total(result) == 1


def test_empty_after_dropna_returns_empty_cells():
    df = pl.DataFrame(
        {"a": [None, None], "b": [1.0, 2.0]},
        schema={"a": pl.Float64, "b": pl.Float64},
    )
    types = {"a": ColumnType.NUMERIC, "b": ColumnType.NUMERIC}
    result = compute_joint_histogram(df, "a", "b", types, bins=2)
    assert result["cells"] == []


def test_pair_boxplot_per_category():
    df = pl.DataFrame({
        "city": ["A", "A", "A", "B", "B", "B"],
        "income": [10.0, 20.0, 30.0, 100.0, 110.0, 120.0],
    })
    result = compute_pair_boxplot(df, "city", "income")
    assert result["kind"] == "box"
    assert result["x"]["column"] == "city"
    assert result["y"]["column"] == "income"
    cats = {b["category"] for b in result["boxes"]}
    assert cats == {"A", "B"}
    box_a = next(b for b in result["boxes"] if b["category"] == "A")
    assert box_a["median"] == 20.0
    assert set(box_a.keys()) >= {"category", "min", "q1", "median", "q3", "max", "outliers"}


def test_pair_boxplot_empty_when_no_overlap():
    df = pl.DataFrame({"city": [None, None], "income": [1.0, 2.0]},
                      schema={"city": pl.String, "income": pl.Float64})
    result = compute_pair_boxplot(df, "city", "income")
    assert result["boxes"] == []


def test_scatter_raw_returns_all_points():
    df = pl.DataFrame({"a": [1.0, 2.0, 3.0], "b": [10.0, 20.0, 30.0]})
    result = compute_joint_scatter(df, "a", "b", max_points=5000)
    assert result["kind"] == "scatter"
    assert result["mode"] == "raw"
    assert result["x"]["column"] == "a"
    assert result["y"]["column"] == "b"
    assert result["sampled_from"] == 3
    assert len(result["points"]) == 3
    assert sorted(result["points"]) == [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]


def test_scatter_sample_caps_points():
    n = 100
    df = pl.DataFrame({"a": [float(i) for i in range(n)], "b": [float(i) for i in range(n)]})
    result = compute_joint_scatter(df, "a", "b", max_points=10)
    assert result["mode"] == "sample"
    assert result["sampled_from"] == n
    assert len(result["points"]) == 10


def test_scatter_sample_is_reproducible():
    n = 100
    df = pl.DataFrame({"a": [float(i) for i in range(n)], "b": [float(i) for i in range(n)]})
    r1 = compute_joint_scatter(df, "a", "b", max_points=10)
    r2 = compute_joint_scatter(df, "a", "b", max_points=10)
    assert r1["points"] == r2["points"]


def test_scatter_drops_nulls_pairwise():
    df = pl.DataFrame({"a": [1.0, None, 3.0], "b": [10.0, 20.0, None]})
    result = compute_joint_scatter(df, "a", "b", max_points=5000)
    assert result["sampled_from"] == 1
    assert result["points"] == [[1.0, 10.0]]


def test_scatter_z_none_regression():
    df = pl.DataFrame({"a": [1.0, 2.0, 3.0], "b": [10.0, 20.0, 30.0]})
    result = compute_joint_scatter(df, "a", "b")
    assert "z" not in result
    assert result["points"] == [[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]


def test_scatter_z_numeric_appends_value_and_range():
    df = pl.DataFrame({
        "a": [1.0, 2.0, 3.0],
        "b": [10.0, 20.0, 30.0],
        "c": [5.0, 15.0, 25.0],
    })
    result = compute_joint_scatter(df, "a", "b", z="c", z_type=ColumnType.NUMERIC)
    assert result["z"] == {"column": "c", "type": "NUMERIC", "min": 5.0, "max": 25.0}
    assert result["points"] == [[1.0, 10.0, 5.0], [2.0, 20.0, 15.0], [3.0, 30.0, 25.0]]


def test_scatter_z_categorical_appends_label_and_categories():
    df = pl.DataFrame({
        "a": [1.0, 2.0, 3.0, 4.0],
        "b": [10.0, 20.0, 30.0, 40.0],
        "g": ["x", "x", "y", "y"],
    })
    result = compute_joint_scatter(df, "a", "b", z="g", z_type=ColumnType.CATEGORICAL)
    assert result["z"]["column"] == "g"
    assert result["z"]["type"] == "CATEGORICAL"
    assert set(result["z"]["categories"]) == {"x", "y"}
    labels = [p[2] for p in result["points"]]
    assert set(labels) == {"x", "y"}


def test_scatter_z_categorical_top_n_others():
    df = pl.DataFrame({
        "a": [float(i) for i in range(12)],
        "b": [float(i) for i in range(12)],
        # 11 distinct labels: g0 most frequent (2 rows), g1..g10 one each
        "g": ["g0", "g0"] + [f"g{i}" for i in range(1, 11)],
    })
    result = compute_joint_scatter(df, "a", "b", z="g", z_type=ColumnType.CATEGORICAL, top_n=10)
    cats = result["z"]["categories"]
    assert len(cats) == 11  # top-10 + "Other"
    assert cats[-1] == "Other"
    assert "Other" in {p[2] for p in result["points"]}


def test_scatter_z_drops_nulls_triplewise():
    df = pl.DataFrame({
        "a": [1.0, 2.0, 3.0],
        "b": [10.0, 20.0, 30.0],
        "c": [5.0, None, 25.0],
    })
    result = compute_joint_scatter(df, "a", "b", z="c", z_type=ColumnType.NUMERIC)
    assert result["sampled_from"] == 2
    assert result["points"] == [[1.0, 10.0, 5.0], [3.0, 30.0, 25.0]]


def test_scatter_z_sample_reproducible():
    n = 100
    df = pl.DataFrame({
        "a": [float(i) for i in range(n)],
        "b": [float(i) for i in range(n)],
        "c": [float(i) for i in range(n)],
    })
    r1 = compute_joint_scatter(df, "a", "b", z="c", z_type=ColumnType.NUMERIC, max_points=10)
    r2 = compute_joint_scatter(df, "a", "b", z="c", z_type=ColumnType.NUMERIC, max_points=10)
    assert r1["points"] == r2["points"]
    assert len(r1["points"]) == 10


def test_scatter_z_datetime_treated_as_numeric():
    from datetime import datetime

    df = pl.DataFrame({
        "a": [1.0, 2.0, 3.0],
        "b": [10.0, 20.0, 30.0],
        "t": pl.Series(
            "t",
            [datetime(2024, 1, 1), datetime(2024, 6, 1), datetime(2024, 12, 1)],
            dtype=pl.Datetime,
        ),
    })
    result = compute_joint_scatter(df, "a", "b", z="t", z_type=ColumnType.DATETIME)
    assert result["z"]["type"] == "NUMERIC"
    assert all(isinstance(p[2], (int, float)) for p in result["points"])
    assert isinstance(result["z"]["min"], (int, float))
    assert isinstance(result["z"]["max"], (int, float))
    assert result["z"]["min"] <= result["z"]["max"]
