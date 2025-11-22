import plotly.express as px
from .base_chart import Chart

class BarChart(Chart, chart_type="bar"):
    def get_chart(self, df):
        x_col = self.channels.get("x")
        y_col = self.channels.get("y")
        color_col = self.channels.get("color")
        
        if not x_col or not y_col:
            raise ValueError("Bar chart requires 'x' and 'y' channels")
        
        fig = px.bar(
            df,
            x=x_col["column"],
            y=y_col["column"],
            color=color_col["column"] if color_col else None,
            title=self.spec.get("title", "Bar Chart"),
            labels={x_col["column"]: x_col["column"], y_col["column"]: y_col["column"]}
        )
        
        fig.update_layout(hovermode='x unified')
        
        return fig
