import plotly.express as px
from .base_chart import Chart

class LineChart(Chart, chart_type="line"):
    def get_chart(self, df):
        x_col = self.channels.get("x")
        y_col = self.channels.get("y")
        
        if not x_col or not y_col:
            raise ValueError("Line chart requires 'x' and 'y' channels")
        
        fig = px.line(
            df,
            x=x_col["column"],
            y=y_col["column"],
            title=self.spec.get("title", "Line Chart"),
            labels={x_col["column"]: x_col["column"], y_col["column"]: y_col["column"]}
        )
        
        fig.update_layout(hovermode='x unified')
        
        return fig
