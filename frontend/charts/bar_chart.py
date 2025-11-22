import altair as alt
from .base_chart import Chart

class BarChart(Chart, chart_type="bar"):
    def get_chart(self, df):
        return alt.Chart(df).mark_bar().encode(
            **self.encoded
        ).properties(
            title=self.spec.get("title")
        )
