import plotly.express as px
from .base_chart import Chart

class HistogramChart(Chart, chart_type="histogram"):
    def get_chart(self, df):
        x_col = self.channels.get("x")
        
        if not x_col:
            raise ValueError("Histogram chart requires 'x' channel")
        
        fig = px.histogram(
            df,
            x=x_col["column"],
            nbins=30,
            title=self.spec.get("title", "Histogram"),
            labels={x_col["column"]: x_col["column"]}
        )
        
        fig.update_layout(hovermode='x unified')
        
        return fig
