import altair as alt
from .base_chart import Chart

class LineChart(Chart, chart_type="line"):
    def get_chart(self, df):
        return alt.Chart(df).mark_line().encode(
            **self.encoded
        ).properties(
            title=self.spec.get("title")
        )
