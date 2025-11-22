import plotly.graph_objects as go
from .base_chart import Chart

class HistogramChart(Chart, chart_type="histogram"):
    def get_chart(self, df):
        x_col = self.channels.get("x")
        
        if not x_col:
            raise ValueError("Histogram chart requires 'x' channel")
        
        fig = go.Figure(data=[go.Histogram(
            x=df[x_col["column"]],
            nbinsx=30,
            name=x_col["column"]
        )])
        
        fig.update_layout(
            title=self.spec.get("title", "Histogram"),
            xaxis_title=x_col["column"],
            yaxis_title="Count",
            hovermode='x unified'
        )
        
        return fig
