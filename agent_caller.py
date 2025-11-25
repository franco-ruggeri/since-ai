import json
import pandas as pd
from typing import Optional, Callable
from plot_type_generator.main import run_plot_generation_pipeline


def get_response(
    user_prompt: str,
    dataframe: pd.DataFrame,
    output_callback: Optional[Callable[[str], None]] = None,
):
    # Convert dataframe to JSON
    try:
        df_json = dataframe.to_json(orient="records")
    except Exception as e:
        raise ValueError(f"Failed to convert dataframe to JSON: {str(e)}")

    # Make API request to backend
    try:
        # Replace with your actual backend URL
        state = run_plot_generation_pipeline(
            user_query=user_prompt,
            data_table=df_json,
            max_iterations=2,
            output_callback=output_callback,
            verbose=False,
        )

        # json_plot_state = json.loads(state["plot_recommendations"])
        # return _get_plot_values(json_plot_state)
        return _get_plot_values_from_saved_json(state["plot_recommendations_path"])
    except Exception as e:
        raise e


def _get_plot_values(json_plot_state: dict):
    processed_data = json_plot_state["processed_data"]
    chart_metadata = {
        "chart_type": json_plot_state["chart_type"],
        "channels": json_plot_state["channels"],
    }
    preprocessing_steps = json_plot_state["preprocessing_steps"]
    rationale = json_plot_state["rationale"]
    df = pd.DataFrame(data=processed_data["data"], columns=processed_data["columns"])
    return df, chart_metadata, preprocessing_steps, rationale


def _get_plot_values_from_saved_json(path: str):
    with open(path, "r") as f:
        x = json.load(f)
        return _get_plot_values(x)
