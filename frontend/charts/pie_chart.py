import plotly.graph_objects as go
from .base_chart import Chart

class PieChart(Chart, chart_type="pie"):
    def get_chart(self, df):
        category = self.spec["channels"]["category"]
        value = self.spec["channels"]["value"]
        
        fig = go.Figure(data=[go.Pie(
            labels=df[category],
            values=df[value],
            name=value
        )])
        
        fig.update_layout(
            title=self.spec.get("title", "Pie Chart")
        )
        
        return fig
