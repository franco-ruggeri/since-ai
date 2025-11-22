import altair as alt
from .base_chart import Chart

class BoxPlotChart(Chart, chart_type="boxplot"):
    def get_chart(self, df):
        return alt.Chart(df).mark_boxplot().encode(
            **self.encoded
        ).properties(
            title=self.spec.get("title")
        )
