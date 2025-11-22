import plotly.graph_objects as go
from .base_chart import Chart

class HeatmapChart(Chart, chart_type="heatmap"):
    def get_chart(self, df):
        x_col = self.channels.get("x")
        y_col = self.channels.get("y")
        color_col = self.channels.get("color")
        
        if not x_col or not y_col or not color_col:
            raise ValueError("Heatmap chart requires 'x', 'y', and 'color' channels")
        
        # Pivot data for heatmap
        heatmap_data = df.pivot_table(
            index=y_col["column"],
            columns=x_col["column"],
            values=color_col["column"],
            aggfunc='first'
        )
        
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            colorscale='Viridis'
        ))
        
        fig.update_layout(
            title=self.spec.get("title", "Heatmap"),
            xaxis_title=x_col["column"],
            yaxis_title=y_col["column"]
        )
        
        return fig
