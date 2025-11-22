from typing import TypedDict


class PlotGenState(TypedDict):
    user_query: str
    data_table: str | dict
    execution_plan: str
    code: str
    figure_path: str
    numeric_feedback: str
    lexical_feedback: str
    visual_feedback: str
    iteration_count: int
    max_iterations: int
    status: str
    llm_model: str | None
    plot_recommendations: str | None
    suggestion_k: int | None
    # Local path to saved JSON file containing parsed recommendations
    plot_recommendations_path: str | None
    # Processed/extracted data ready for plotting (as dict or JSON string)
    processed_data: str | dict | None
