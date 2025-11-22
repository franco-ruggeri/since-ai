class ChartRegistry:
    _registry = {}

    @classmethod
    def register(cls, chart_type, chart_cls):
        cls._registry[chart_type] = chart_cls

    @classmethod
    def get(cls, chart_type):
        return cls._registry.get(chart_type)
