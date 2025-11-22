import plotly.express as px
from .base_chart import Chart

class PieChart(Chart, chart_type="pie"):
    def get_chart(self, df):
        category = self.spec["channels"]["category"]
        value = self.spec["channels"]["value"]
        
        fig = px.pie(
            df,
            names=category,
            values=value,
            title=self.spec.get("title", "Pie Chart")
        )
        
        return fig
