import plotly.graph_objects as go
from .base_chart import Chart

class BarChart(Chart, chart_type="bar"):
    def get_chart(self, df):
        x_col = self.channels.get("x")
        y_col = self.channels.get("y")
        color_col = self.channels.get("color")
        
        if not x_col or not y_col:
            raise ValueError("Bar chart requires 'x' and 'y' channels")
        
        fig = go.Figure()
        
        if color_col:
            fig.add_trace(go.Bar(
                x=df[x_col["column"]],
                y=df[y_col["column"]],
                marker=dict(color=df[color_col["column"]]),
                name=y_col["column"]
            ))
        else:
            fig.add_trace(go.Bar(
                x=df[x_col["column"]],
                y=df[y_col["column"]],
                name=y_col["column"]
            ))
        
        fig.update_layout(
            title=self.spec.get("title", "Bar Chart"),
            xaxis_title=x_col["column"],
            yaxis_title=y_col["column"],
            hovermode='x unified'
        )
        
        return fig
