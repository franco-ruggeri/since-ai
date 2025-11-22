import plotly.express as px
from .base_chart import Chart

class HeatmapChart(Chart, chart_type="heatmap"):
    def get_chart(self, df):
        x_col = self.channels.get("x")
        y_col = self.channels.get("y")
        color_col = self.channels.get("color")
        
        if not x_col or not y_col or not color_col:
            raise ValueError("Heatmap chart requires 'x', 'y', and 'color' channels")
        
        fig = px.density_heatmap(
            df,
            x=x_col["column"],
            y=y_col["column"],
            nbinsx=10,
            nbinsy=10,
            color_continuous_scale='Viridis',
            title=self.spec.get("title", "Heatmap")
        )
        
        return fig
