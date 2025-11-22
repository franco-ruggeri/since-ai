import altair as alt
import pandas as pd
from .chart_registry import ChartRegistry

class Chart:
    def __init_subclass__(cls, chart_type=None, **kwargs):
        super().__init_subclass__(**kwargs)
        if chart_type:
            ChartRegistry.register(chart_type, cls)

    def __init__(self, spec):
        self.spec = spec
        self.encoded = None

    # -------------------------------
    # Infer Vega type from Pandas dtype
    # -------------------------------
    def _infer_type(self, series: pd.Series):
        if pd.api.types.is_datetime64_any_dtype(series):
            return "temporal"
        if pd.api.types.is_numeric_dtype(series):
            return "quantitative"
        return "nominal"

    # -------------------------------
    # Build encoding dictionary
    # -------------------------------
    def _build_encodings(self, df):
        channels = self.spec.get("channels", {})
        encoding = {}

        for channel, col in channels.items():
            vega_type = self._infer_type(df[col])

            if channel == "x":
                encoding["x"] = alt.X(col, type=vega_type)
            elif channel == "y":
                encoding["y"] = alt.Y(col, type=vega_type)
            elif channel == "color":
                encoding["color"] = alt.Color(col, type=vega_type)
            else:
                encoding[channel] = alt.Value(col)

        return encoding

    # -------------------------------
    # FINAL TEMPLATE METHOD
    # -------------------------------
    def build(self, df):
        # compute encoding once
        self.encoded = self._build_encodings(df)
        # subclass-specific drawing
        return self.get_chart(df)

    # -------------------------------
    # Subclasses must implement get_chart()
    # -------------------------------
    def get_chart(self, df):
        raise NotImplementedError("Subclasses must implement get_chart()")
