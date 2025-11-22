import altair as alt
from .base_chart import Chart

class PieChart(Chart, chart_type="pie"):
    def get_chart(self, df):
        category = self.spec["channels"]["category"]
        value = self.spec["channels"]["value"]

        return alt.Chart(df).mark_arc().encode(
            theta=alt.Theta(value, stack=True),
            color=alt.Color(category)
        ).properties(
            title=self.spec.get("title")
        )
