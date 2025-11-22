from .chart_registry import ChartRegistry

class Chart:
    def __init_subclass__(cls, chart_type=None, **kwargs):
        super().__init_subclass__(**kwargs)
        if chart_type:
            ChartRegistry.register(chart_type, cls)

    def __init__(self, spec):
        self.spec = spec
        self.channels = None

    def _build_channels(self, df):
        channels = self.spec.get("channels", {})
        result = {}

        for channel, col in channels.items():
            result[channel] = {
                "column": col,
                "data": df[col]
            }

        return result

    def build(self, df):
        # compute channels once
        self.channels = self._build_channels(df)
        # subclass-specific drawing
        return self.get_chart(df)

    # Subclasses must implement get_chart()
    def get_chart(self, df):
        raise NotImplementedError("Subclasses must implement get_chart()")
