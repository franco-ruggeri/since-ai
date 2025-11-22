import plotly.graph_objects as go
from .base_chart import Chart

class LineChart(Chart, chart_type="line"):
    def get_chart(self, df):
        x_col = self.channels.get("x")
        y_col = self.channels.get("y")
        
        if not x_col or not y_col:
            raise ValueError("Line chart requires 'x' and 'y' channels")
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=df[x_col["column"]],
            y=df[y_col["column"]],
            mode='lines',
            name=y_col["column"]
        ))
        
        fig.update_layout(
            title=self.spec.get("title", "Line Chart"),
            xaxis_title=x_col["column"],
            yaxis_title=y_col["column"],
            hovermode='x unified'
        )
        
        return fig
