import plotly.express as px
from .base_chart import Chart

class BoxPlotChart(Chart, chart_type="box"):
    def get_chart(self, df):
        x_col = self.channels.get("x")
        y_col = self.channels.get("y")
        
        if not y_col:
            raise ValueError("Box plot chart requires 'y' channel")
        
        fig = px.box(
            df,
            x=x_col["column"] if x_col else None,
            y=y_col["column"],
            title=self.spec.get("title", "Box Plot"),
            labels={y_col["column"]: y_col["column"]}
        )
        
        fig.update_layout(hovermode='y unified')
        
        return fig
