import pandas as pd
from .chart_registry import ChartRegistry

class Chart:
    def __init_subclass__(cls, chart_type=None, **kwargs):
        super().__init_subclass__(**kwargs)
        if chart_type:
            ChartRegistry.register(chart_type, cls)

    def __init__(self, spec):
        self.spec = spec
        self.channels = None

    # -------------------------------
    # Infer data type from Pandas dtype
    # -------------------------------
    def _infer_type(self, series: pd.Series):
        if pd.api.types.is_datetime64_any_dtype(series):
            return "temporal"
        if pd.api.types.is_numeric_dtype(series):
            return "quantitative"
        return "nominal"

    # -------------------------------
    # Build channels dictionary
    # -------------------------------
    def _build_channels(self, df):
        channels = self.spec.get("channels", {})
        result = {}

        for channel, col in channels.items():
            col_type = self._infer_type(df[col])
            result[channel] = {
                "column": col,
                "type": col_type,
                "data": df[col]
            }

        return result

    # -------------------------------
    # FINAL TEMPLATE METHOD
    # -------------------------------
    def build(self, df):
        # compute channels once
        self.channels = self._build_channels(df)
        # subclass-specific drawing
        return self.get_chart(df)

    # -------------------------------
    # Subclasses must implement get_chart()
    # -------------------------------
    def get_chart(self, df):
        raise NotImplementedError("Subclasses must implement get_chart()")
