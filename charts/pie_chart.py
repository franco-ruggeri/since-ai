import plotly.express as px
from .base_chart import Chart


class PieChart(Chart, chart_type="pie"):
    def get_chart(self, df):

        x_channel = self.channels.get("x")
        y_channel = self.channels.get("y")

        if not x_channel or not y_channel:
            raise ValueError(
                f"Pie chart requires both 'x' and 'y' channels. Got x_channel={x_channel}, y_channel={y_channel}"
            )

        category = x_channel["column"]
        value = y_channel["column"]

        fig = px.pie(
            df, names=category, values=value, title=self.spec.get("title", "Pie Chart")
        )

        return fig
