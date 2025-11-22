import altair as alt
from .base_chart import Chart

class HistogramChart(Chart, chart_type="histogram"):
    def get_chart(self, df):
        return alt.Chart(df).mark_bar().encode(
            x=alt.X(self.spec["channels"]["x"], bin=True),
            y="count()",
            **{k: v for k, v in self.encoded.items() if k not in ["x", "y"]}
        ).properties(
            title=self.spec.get("title")
        )
