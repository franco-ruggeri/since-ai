import json
import pandas as pd
from typing import Optional, Callable
from plot_type_generator.main import run_plot_generation_pipeline
from plot_type_generator.plot_gen_state import PlotGenState


def get_response(user_prompt: str, dataframe: pd.DataFrame, output_callback: Optional[Callable[[str], None]] = None) -> PlotGenState:
    """
    Converts a dataframe to JSON and sends it to the backend API along with user prompt.

    Args:
        user_prompt (str): The user's prompt/query
        dataframe (pd.DataFrame): The dataframe to visualize
        output_callback (Optional[Callable]): Optional callback to send output to (e.g., st.write)

    Returns:
        dict: The API response

    Raises:
        ValueError: If dataframe cannot be converted to JSON
        requests.RequestException: If API request fails
    """

    # Convert dataframe to JSON
    try:
        df_json = dataframe.to_json(orient='records')
    except Exception as e:
        raise ValueError(f"Failed to convert dataframe to JSON: {str(e)}")

    print(df_json)
    # Make API request to backend
    try:
        # Replace with your actual backend URL
        response = run_plot_generation_pipeline(
            user_query=user_prompt, data_table=df_json, max_iterations=2, output_callback=output_callback
        )

        return response.json()
    except Exception as e:
        raise e


def get_test_response():
    with open('recommendations/recommendations/plot_recommendations_Rappuset.json', 'r') as f:
        x = json.load(f)
        processed_data = x["processed_data"]
        chart_metadata = {
            "chart_type": x["chart_type"],
            "channels": x["channels"]
        }
        df = pd.DataFrame(
            data=processed_data["data"], columns=processed_data["columns"])
        return df, chart_metadata
