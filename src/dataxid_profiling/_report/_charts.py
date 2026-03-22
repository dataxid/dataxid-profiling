"""Chart rendering abstraction — pluggable backend.

ChartRenderer Protocol defines the contract.
EChartsRenderer is the default implementation.
"""

from __future__ import annotations

import json
from typing import Protocol, runtime_checkable


@runtime_checkable
class ChartRenderer(Protocol):
    """Contract for chart rendering backends.

    Each method returns an HTML string containing a <div> and <script>
    that renders the chart. The div_id must be unique per page.
    """

    def histogram(
        self,
        div_id: str,
        labels: list[str],
        values: list[int | float],
        title: str = "",
    ) -> str: ...

    def bar_horizontal(
        self,
        div_id: str,
        labels: list[str],
        values: list[int | float],
        title: str = "",
    ) -> str: ...

    def heatmap(
        self,
        div_id: str,
        x_labels: list[str],
        y_labels: list[str],
        data: list[list[float]],
        title: str = "",
    ) -> str: ...

    def pie(
        self,
        div_id: str,
        labels: list[str],
        values: list[int | float],
        title: str = "",
    ) -> str: ...

    def word_cloud(
        self,
        div_id: str,
        words: list[str],
        weights: list[int | float],
        title: str = "",
    ) -> str: ...


class EChartsRenderer:
    """ECharts-based chart renderer. Produces self-contained HTML snippets."""

    CHART_HEIGHT = "300px"
    HEATMAP_HEIGHT = "400px"
    BRAND_TEAL = "#0d3b3b"
    BRAND_CORAL = "#e8845c"
    BRAND_PURPLE = "#b06aed"
    BRAND_PEACH = "#f4a683"

    def histogram(
        self,
        div_id: str,
        labels: list[str],
        values: list[int | float],
        title: str = "",
    ) -> str:
        option = {
            "title": {"text": title, "left": "center", "textStyle": {"fontSize": 13}},
            "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
            "grid": {"left": "10%", "right": "5%", "bottom": "15%", "top": "15%"},
            "xAxis": {
                "type": "category",
                "data": labels,
                "axisLabel": {"rotate": 30, "fontSize": 10},
            },
            "yAxis": {"type": "value"},
            "series": [
                {
                    "type": "bar",
                    "data": values,
                    "itemStyle": {"color": self.BRAND_TEAL},
                    "barWidth": "90%",
                }
            ],
        }
        return self._wrap(div_id, option, self.CHART_HEIGHT)

    def bar_horizontal(
        self,
        div_id: str,
        labels: list[str],
        values: list[int | float],
        title: str = "",
    ) -> str:
        option = {
            "title": {"text": title, "left": "center", "textStyle": {"fontSize": 13}},
            "tooltip": {"trigger": "axis", "axisPointer": {"type": "shadow"}},
            "grid": {"left": "25%", "right": "10%", "bottom": "10%", "top": "15%"},
            "xAxis": {"type": "value"},
            "yAxis": {
                "type": "category",
                "data": list(reversed(labels)),
                "axisLabel": {"fontSize": 11},
            },
            "series": [
                {
                    "type": "bar",
                    "data": list(reversed(values)),
                    "itemStyle": {"color": self.BRAND_CORAL},
                }
            ],
        }
        return self._wrap(div_id, option, self.CHART_HEIGHT)

    def heatmap(
        self,
        div_id: str,
        x_labels: list[str],
        y_labels: list[str],
        data: list[list[float]],
        title: str = "",
    ) -> str:
        flat_data = []
        v_min, v_max = float("inf"), float("-inf")
        for row_idx, row in enumerate(data):
            for col_idx, val in enumerate(row):
                flat_data.append([col_idx, row_idx, round(val, 3)])
                if row_idx != col_idx:
                    v_min = min(v_min, val)
                    v_max = max(v_max, val)

        if v_min == float("inf"):
            v_min, v_max = -1.0, 1.0

        option = {
            "title": {"text": title, "left": "center", "textStyle": {"fontSize": 13}},
            "tooltip": {
                "position": "top",
                "formatter": None,  # replaced below
            },
            "grid": {"left": "15%", "right": "12%", "bottom": "15%", "top": "15%"},
            "xAxis": {
                "type": "category",
                "data": x_labels,
                "splitArea": {"show": True},
                "axisLabel": {"rotate": 30, "fontSize": 10},
            },
            "yAxis": {
                "type": "category",
                "data": y_labels,
                "splitArea": {"show": True},
                "axisLabel": {"fontSize": 10},
            },
            "visualMap": {
                "min": round(v_min, 3),
                "max": round(v_max, 3),
                "calculable": True,
                "orient": "vertical",
                "right": "2%",
                "top": "center",
                "inRange": {
                    "color": ["#0d3b3b", "#1a6b6b", "#4a9e9e", "#a8d5d5",
                              "#f4e8d0", "#f4a683", "#e8845c", "#b06aed"],
                },
            },
            "series": [
                {
                    "type": "heatmap",
                    "data": flat_data,
                    "label": {"show": True, "fontSize": 10},
                    "emphasis": {
                        "itemStyle": {"shadowBlur": 10, "shadowColor": "rgba(0,0,0,0.5)"}
                    },
                }
            ],
        }

        option_json = json.dumps(option, ensure_ascii=False)
        # Inject JS formatter function (can't be in JSON)
        tooltip_fn = (
            "function(p){return p.name + ' vs ' "
            "+ p.data[1] + ': ' + p.data[2];}"
        )
        option_json = option_json.replace('"formatter": null', f'"formatter": {tooltip_fn}')

        return (
            f'<div id="{div_id}" style="width:100%;height:{self.HEATMAP_HEIGHT}"></div>\n'
            f"<script>echarts.init(document.getElementById('{div_id}'))"
            f".setOption({option_json});</script>"
        )

    def pie(
        self,
        div_id: str,
        labels: list[str],
        values: list[int | float],
        title: str = "",
    ) -> str:
        pie_data = [{"name": lbl, "value": v} for lbl, v in zip(labels, values, strict=True)]
        option = {
            "title": {"text": title, "left": "center", "textStyle": {"fontSize": 13}},
            "tooltip": {"trigger": "item", "formatter": "{b}: {c} ({d}%)"},
            "legend": {"orient": "horizontal", "bottom": "0%", "left": "center"},
            "series": [
                {
                    "type": "pie",
                    "radius": ["35%", "60%"],
                    "center": ["50%", "50%"],
                    "data": pie_data,
                    "emphasis": {
                        "itemStyle": {
                            "shadowBlur": 10,
                            "shadowOffsetX": 0,
                            "shadowColor": "rgba(0,0,0,0.5)",
                        }
                    },
                    "label": {"formatter": "{b}: {d}%"},
                }
            ],
        }
        return self._wrap(div_id, option, self.CHART_HEIGHT)

    def word_cloud(
        self,
        div_id: str,
        words: list[str],
        weights: list[int | float],
        title: str = "",
    ) -> str:
        wc_data = [
            {"name": w, "value": v}
            for w, v in zip(words, weights, strict=True)
        ]
        option = {
            "title": {"text": title, "left": "center", "textStyle": {"fontSize": 13}},
            "tooltip": {"trigger": "item", "formatter": "{b}: {c}"},
            "series": [
                {
                    "type": "wordCloud",
                    "shape": "circle",
                    "sizeRange": [14, 60],
                    "rotationRange": [0, 0],
                    "rotationStep": 0,
                    "gridSize": 8,
                    "drawOutOfBound": False,
                    "layoutAnimation": True,
                    "textStyle": {
                        "fontFamily": "Inter, system-ui, sans-serif",
                        "fontWeight": "bold",
                        "color": None,  # replaced below
                    },
                    "emphasis": {
                        "textStyle": {"shadowBlur": 10, "shadowColor": "rgba(0,0,0,0.3)"}
                    },
                    "data": wc_data,
                }
            ],
        }

        option_json = json.dumps(option, ensure_ascii=False)
        color_fn = (
            "function(){var c=["
            f"'{self.BRAND_TEAL}','{self.BRAND_CORAL}',"
            f"'{self.BRAND_PURPLE}','{self.BRAND_PEACH}'"
            "];return c[Math.floor(Math.random()*c.length)];}"
        )
        option_json = option_json.replace('"color": null', f'"color": {color_fn}')

        return (
            f'<div id="{div_id}" style="width:100%;height:{self.CHART_HEIGHT}"></div>\n'
            f"<script>echarts.init(document.getElementById('{div_id}'))"
            f".setOption({option_json});</script>"
        )

    @staticmethod
    def _wrap(div_id: str, option: dict, height: str) -> str:
        option_json = json.dumps(option, ensure_ascii=False)
        return (
            f'<div id="{div_id}" style="width:100%;height:{height}"></div>\n'
            f"<script>echarts.init(document.getElementById('{div_id}'))"
            f".setOption({option_json});</script>"
        )
