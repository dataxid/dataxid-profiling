from __future__ import annotations

from dataclasses import asdict
from typing import Any

import polars as pl

from dataxid_profiling._interactions import _boxplot_for_pair
from dataxid_profiling._type_inference import ColumnType

_NUMERIC_TYPES = {ColumnType.NUMERIC, ColumnType.DATETIME}


def _axis_is_numeric(t: ColumnType) -> bool:
    return t in _NUMERIC_TYPES


def _numeric_edges(series: pl.Series, bins: int) -> list[float]:
    """Equal-width upper-edge breakpoints, matching the 1D histogram convention."""
    lo = series.min()
    hi = series.max()
    if lo is None or hi is None:
        return []
    lo_f = float(lo)
    hi_f = float(hi)
    if lo_f == hi_f:
        return [hi_f]
    width = (hi_f - lo_f) / bins
    return [lo_f + width * (i + 1) for i in range(bins)]


def _numeric_index_expr(col: str, edges: list[float]) -> pl.Expr:
    """Bin index for numeric values given equal-width upper edges."""
    n = len(edges)
    if n <= 1:
        return pl.lit(0)
    # Assign the lowest matching bin: value <= edges[i] -> i (checked ascending).
    expr = pl.lit(n - 1)
    for i in range(n - 1, -1, -1):
        expr = pl.when(pl.col(col) <= edges[i]).then(pl.lit(i)).otherwise(expr)
    return expr


def _axis_descriptor_indexed(
    df: pl.DataFrame, col: str, ctype: ColumnType, bins: int, top_n: int, alias: str
) -> tuple[dict[str, Any], pl.Expr]:
    """Return (descriptor, bin-index expr aliased to `alias`)."""
    if _axis_is_numeric(ctype):
        edges = _numeric_edges(df[col], bins)
        return (
            {"column": col, "type": ctype.name, "bins": edges},
            _numeric_index_expr(col, edges).alias(alias),
        )

    labels = (
        df.select(col)
        .drop_nulls()
        .group_by(col)
        .len()
        .sort("len", descending=True)
        .head(top_n)
        .get_column(col)
        .to_list()
    )
    labels = [str(v) for v in labels]
    label_to_idx = {lab: i for i, lab in enumerate(labels)}
    other_idx = len(labels)

    distinct = df.select(col).drop_nulls().get_column(col).n_unique()
    has_overflow = distinct > len(labels)
    bins_out = labels + (["Other"] if has_overflow else [])

    idx_expr = (
        pl.col(col)
        .cast(pl.String)
        .replace_strict(label_to_idx, default=other_idx, return_dtype=pl.Int64)
        .alias(alias)
    )
    return ({"column": col, "type": ctype.name, "bins": bins_out}, idx_expr)


def compute_joint_histogram(
    df: pl.DataFrame,
    x: str,
    y: str,
    column_types: dict[str, ColumnType],
    bins: int = 20,
    top_n: int = 50,
) -> dict[str, Any]:
    xt = column_types[x]
    yt = column_types[y]

    sub = df.select(x, y).drop_nulls()
    if sub.height == 0:
        return {
            "kind": "grid",
            "x": {"column": x, "type": xt.name, "bins": []},
            "y": {"column": y, "type": yt.name, "bins": []},
            "cells": [],
        }

    x_desc, x_idx = _axis_descriptor_indexed(sub, x, xt, bins, top_n, "__xi")
    y_desc, y_idx = _axis_descriptor_indexed(sub, y, yt, bins, top_n, "__yi")

    counts = (
        sub.with_columns(x_idx, y_idx)
        .group_by("__xi", "__yi")
        .len()
        .sort("__xi", "__yi")
    )
    cells = [
        {"xi": int(r["__xi"]), "yi": int(r["__yi"]), "count": int(r["len"])}
        for r in counts.iter_rows(named=True)
    ]
    return {"kind": "grid", "x": x_desc, "y": y_desc, "cells": cells}


def compute_pair_boxplot(df: pl.DataFrame, cat_col: str, num_col: str) -> dict[str, Any]:
    groups = _boxplot_for_pair(df, cat_col, num_col)
    return {
        "kind": "box",
        "x": {"column": cat_col, "type": "CATEGORICAL"},
        "y": {"column": num_col, "type": "NUMERIC"},
        "boxes": [asdict(g) for g in groups],
    }


def compute_joint_scatter(
    df: pl.DataFrame,
    x: str,
    y: str,
    z: str | None = None,
    z_type: ColumnType | None = None,
    max_points: int = 5000,
    top_n: int = 10,
) -> dict[str, Any]:
    """Raw (or reproducibly sampled) numeric×numeric points for a scatter plot.

    Optional z encodes a third column: numeric z -> point magnitude (min/max
    provided), categorical z -> series label (top-N frequent categories, rest
    "Other"). z=None keeps the original 2-tuple point shape.
    """
    cols = [x, y] + ([z] if z is not None else [])
    sub = df.select(cols).drop_nulls()
    total = sub.height
    if total > max_points:
        sub = sub.sample(n=max_points, seed=42)
        mode = "sample"
    else:
        mode = "raw"

    result: dict[str, Any] = {
        "kind": "scatter",
        "x": {"column": x, "type": "NUMERIC"},
        "y": {"column": y, "type": "NUMERIC"},
        "mode": mode,
        "sampled_from": total,
    }

    xs = sub.get_column(x).to_list()
    ys = sub.get_column(y).to_list()

    if z is None:
        result["points"] = [[float(rx), float(ry)] for rx, ry in zip(xs, ys)]
        return result

    if z_type is not None and _axis_is_numeric(z_type):
        # DATETIME is treated as a numeric axis; cast temporal z to an integer
        # epoch so per-point values and min/max are plain numbers.
        z_series = sub.get_column(z)
        if z_series.dtype.is_temporal():
            z_series = z_series.dt.epoch("ms")
        zs = z_series.to_list()
        result["points"] = [
            [float(rx), float(ry), float(rz)] for rx, ry, rz in zip(xs, ys, zs)
        ]
        zmin = z_series.min()
        zmax = z_series.max()
        result["z"] = {
            "column": z,
            "type": "NUMERIC",
            "min": float(zmin) if zmin is not None else 0.0,
            "max": float(zmax) if zmax is not None else 0.0,
        }
        return result

    # categorical z: top-N labels by frequency, rest -> "Other"
    labels = (
        sub.select(z)
        .group_by(z)
        .len()
        .sort("len", descending=True)
        .head(top_n)
        .get_column(z)
        .to_list()
    )
    labels = [str(v) for v in labels]
    keep = set(labels)
    distinct = sub.select(z).get_column(z).n_unique()
    has_overflow = distinct > len(labels)
    zs = [str(v) for v in sub.get_column(z).to_list()]
    zlabels = [v if v in keep else "Other" for v in zs]
    result["points"] = [
        [float(rx), float(ry), zl] for rx, ry, zl in zip(xs, ys, zlabels)
    ]
    result["z"] = {
        "column": z,
        "type": "CATEGORICAL",
        "categories": labels + (["Other"] if has_overflow else []),
    }
    return result
