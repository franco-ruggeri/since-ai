import plotly.graph_objects as go
from .base_chart import Chart

class BoxPlotChart(Chart, chart_type="boxplot"):
    def get_chart(self, df):
        x_col = self.channels.get("x")
        y_col = self.channels.get("y")
        
        if not y_col:
            raise ValueError("Box plot chart requires 'y' channel")
        
        if x_col:
            # Box plot with categories
            fig = go.Figure()
            for category in df[x_col["column"]].unique():
                fig.add_trace(go.Box(
                    y=df[df[x_col["column"]] == category][y_col["column"]],
                    name=str(category)
                ))
        else:
            # Simple box plot
            fig = go.Figure(data=[go.Box(
                y=df[y_col["column"]],
                name=y_col["column"]
            )])
        
        fig.update_layout(
            title=self.spec.get("title", "Box Plot"),
            yaxis_title=y_col["column"],
            hovermode='y unified'
        )
        
        if x_col:
            fig.update_layout(xaxis_title=x_col["column"])
        
        return fig
