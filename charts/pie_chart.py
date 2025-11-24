import plotly.express as px
from .base_chart import Chart

class PieChart(Chart, chart_type="pie"):
    def get_chart(self, df):
        category = self.channels.get("x")
        value = self.channels.get("y")
        
        fig = px.pie(
            df,
            names=category,
            values=value,
            title=self.spec.get("title", "Pie Chart")
        )
        
        return fig
