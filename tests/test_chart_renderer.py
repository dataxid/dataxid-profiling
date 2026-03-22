from __future__ import annotations

import pytest

from dataxid_profiling._report._charts import ChartRenderer, EChartsRenderer


@pytest.fixture
def renderer() -> EChartsRenderer:
    return EChartsRenderer()


class TestProtocolCompliance:
    def test_echarts_implements_protocol(self, renderer: EChartsRenderer):
        assert isinstance(renderer, ChartRenderer)


class TestHistogram:
    def test_returns_html_string(self, renderer: EChartsRenderer):
        html = renderer.histogram("hist_1", ["0-10", "10-20", "20-30"], [5, 12, 8])
        assert isinstance(html, str)

    def test_contains_div_and_script(self, renderer: EChartsRenderer):
        html = renderer.histogram("hist_1", ["0-10", "10-20"], [5, 12])
        assert '<div id="hist_1"' in html
        assert "<script>" in html
        assert "echarts.init" in html

    def test_contains_data(self, renderer: EChartsRenderer):
        html = renderer.histogram("h1", ["a", "b"], [10, 20])
        assert "[10, 20]" in html or "[10,20]" in html

    def test_title_in_output(self, renderer: EChartsRenderer):
        html = renderer.histogram("h1", ["a"], [1], title="Age Distribution")
        assert "Age Distribution" in html

    def test_unique_div_ids(self, renderer: EChartsRenderer):
        h1 = renderer.histogram("chart_1", ["a"], [1])
        h2 = renderer.histogram("chart_2", ["a"], [1])
        assert 'id="chart_1"' in h1
        assert 'id="chart_2"' in h2


class TestBarHorizontal:
    def test_returns_html(self, renderer: EChartsRenderer):
        html = renderer.bar_horizontal("bar_1", ["Istanbul", "Ankara"], [45, 30])
        assert '<div id="bar_1"' in html
        assert "echarts.init" in html

    def test_reversed_order(self, renderer: EChartsRenderer):
        html = renderer.bar_horizontal("b1", ["A", "B", "C"], [10, 20, 30])
        assert "echarts.init" in html

    def test_title(self, renderer: EChartsRenderer):
        html = renderer.bar_horizontal("b1", ["A"], [1], title="Top Values")
        assert "Top Values" in html


class TestHeatmap:
    def test_returns_html(self, renderer: EChartsRenderer):
        html = renderer.heatmap(
            "hm_1",
            x_labels=["a", "b"],
            y_labels=["a", "b"],
            data=[[1.0, 0.5], [0.5, 1.0]],
        )
        assert '<div id="hm_1"' in html
        assert "echarts.init" in html

    def test_contains_heatmap_type(self, renderer: EChartsRenderer):
        html = renderer.heatmap("hm", ["x"], ["y"], data=[[1.0]])
        assert "heatmap" in html

    def test_correlation_values(self, renderer: EChartsRenderer):
        html = renderer.heatmap(
            "hm",
            ["a", "b"],
            ["a", "b"],
            data=[[1.0, -0.8], [-0.8, 1.0]],
        )
        assert "-0.8" in html

    def test_title(self, renderer: EChartsRenderer):
        html = renderer.heatmap("hm", ["a"], ["a"], [[1.0]], title="Pearson")
        assert "Pearson" in html

    def test_tooltip_formatter(self, renderer: EChartsRenderer):
        html = renderer.heatmap("hm", ["a", "b"], ["a", "b"], [[1.0, 0.5], [0.5, 1.0]])
        assert "function(p)" in html


class TestPie:
    def test_returns_html(self, renderer: EChartsRenderer):
        html = renderer.pie("pie_1", ["True", "False"], [70, 30])
        assert '<div id="pie_1"' in html
        assert "echarts.init" in html

    def test_contains_pie_type(self, renderer: EChartsRenderer):
        html = renderer.pie("p1", ["A", "B"], [1, 2])
        assert '"pie"' in html

    def test_data_names(self, renderer: EChartsRenderer):
        html = renderer.pie("p1", ["True", "False", "Null"], [60, 30, 10])
        assert "True" in html
        assert "False" in html
        assert "Null" in html

    def test_title(self, renderer: EChartsRenderer):
        html = renderer.pie("p1", ["A"], [1], title="Distribution")
        assert "Distribution" in html

    def test_label_value_mismatch_raises(self, renderer: EChartsRenderer):
        with pytest.raises(ValueError):
            renderer.pie("p1", ["A", "B"], [1])


class TestWordCloud:
    def test_returns_html(self, renderer: EChartsRenderer):
        html = renderer.word_cloud("wc_1", ["cat", "dog", "bird"], [50, 30, 20])
        assert '<div id="wc_1"' in html
        assert "echarts.init" in html

    def test_contains_wordcloud_type(self, renderer: EChartsRenderer):
        html = renderer.word_cloud("wc", ["a", "b"], [10, 5])
        assert "wordCloud" in html

    def test_data_words(self, renderer: EChartsRenderer):
        html = renderer.word_cloud("wc", ["Istanbul", "Ankara", "Izmir"], [100, 60, 40])
        assert "Istanbul" in html
        assert "Ankara" in html
        assert "Izmir" in html

    def test_title(self, renderer: EChartsRenderer):
        html = renderer.word_cloud("wc", ["a"], [1], title="Word Cloud")
        assert "Word Cloud" in html

    def test_word_weight_mismatch_raises(self, renderer: EChartsRenderer):
        with pytest.raises(ValueError):
            renderer.word_cloud("wc", ["a", "b"], [1])

    def test_empty_data(self, renderer: EChartsRenderer):
        html = renderer.word_cloud("wc_empty", [], [])
        assert '<div id="wc_empty"' in html


class TestEdgeCases:
    def test_empty_data(self, renderer: EChartsRenderer):
        html = renderer.histogram("e1", [], [])
        assert '<div id="e1"' in html

    def test_single_value(self, renderer: EChartsRenderer):
        html = renderer.histogram("e2", ["bin"], [42])
        assert "42" in html

    def test_special_chars_in_labels(self, renderer: EChartsRenderer):
        html = renderer.bar_horizontal("e3", ["<script>", "a&b"], [1, 2])
        assert "echarts.init" in html
