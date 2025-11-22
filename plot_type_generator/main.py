import os, sys

from pathlib import Path
from dotenv import load_dotenv

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from plot_type_generator.query_planning_agent import query_planning_agent
from plot_type_generator.plot_type_chooser_agent import plot_type_chooser_agent
from plot_type_generator.plot_gen_state import PlotGenState
import plot_type_generator.utils as utils


def main() -> None:
    """Orchestrate the pipeline: query planning -> plot type choosing.

    Ensure `FEATHERLESS_API_KEY` is set in the environment (or in a `.env` file).
    """
    load_dotenv()

    if not os.environ.get("FEATHERLESS_API_KEY"):
        print(
            "FEATHERLESS_API_KEY not set. Set it in the environment or plot_type_generator/.env"
        )
        return

    state: PlotGenState = {
        "user_query": "Show me monthly sales trends for product A in 2024",
        "data_table": {
            "columns": ["date", "product", "sales_amount", "region"],
            "dtypes": {
                "date": "datetime",
                "product": "category",
                "sales_amount": "float",
                "region": "category",
            },
            "sample_rows": [
                ["2024-01-01", "A", 123.45, "north"],
                ["2024-01-02", "B", 10.0, "south"],
            ],
        },
        "execution_plan": "",
        "code": "",
        "figure_path": "",
        "numeric_feedback": "",
        "lexical_feedback": "",
        "visual_feedback": "",
        "iteration_count": 0,
        "max_iterations": 3,
        "status": "pending",
        "llm_model": None,
        "suggestion_k": 3,
        "plot_recommendations": None,
        "plot_recommendations_path": "./plot_recommendations.json",
    }

    try:
        state = query_planning_agent(state)
    except Exception as e:
        print("Query planning agent failed:", e)
        return

    print("--- Execution Plan ---")
    print(state.get("execution_plan"))

    try:
        state = plot_type_chooser_agent(state, k=state.get("suggestion_k") or 3)
    except Exception as e:
        print("Plot type chooser failed:", e)
        return

    print("--- Plot Recommendations ---")
    print(state.get("plot_recommendations"))
    if state.get("plot_recommendations"):
        parsed_text = utils.extract_json_content(state.get("plot_recommendations"))
        utils.save_recommendations(parsed_text)


if __name__ == "__main__":
    main()
