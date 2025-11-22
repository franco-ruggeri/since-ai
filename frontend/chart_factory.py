import altair as alt
import pandas as pd
from charts.chart_registry import ChartRegistry

def make_chart(df, spec):
    chart_type = spec["chart_type"]
    print(chart_type)
    chart_cls = ChartRegistry.get(chart_type)

    if chart_cls is None:
        raise ValueError(f"Unknown chart type: {chart_type}")

    chart = chart_cls(spec)
    return chart.build(df)
