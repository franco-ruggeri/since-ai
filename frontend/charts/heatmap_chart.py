import altair as alt
from .base_chart import Chart

class HeatmapChart(Chart, chart_type="heatmap"):
    def get_chart(self, df):
        return alt.Chart(df).mark_rect().encode(
            **self.encoded
        ).properties(
            title=self.spec.get("title")
        )
