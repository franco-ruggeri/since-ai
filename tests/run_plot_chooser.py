import sys
from pathlib import Path
from env_vars import *

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from plot_type_generator.plot_type_chooser_agent import plot_type_chooser_agent
from plot_type_generator.plot_gen_state import PlotGenState
import os


def _load_env_file(path: str) -> None:
    """Lightweight .env loader (no extra dependency).

    Supports lines like KEY=VALUE or KEY="VALUE" and ignores comments/blank lines.
    Sets values into os.environ if not already set.
    """
    p = Path(path)
    if not p.exists():
        return
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip()
        # strip optional surrounding quotes
        if (val.startswith('"') and val.endswith('"')) or (
            val.startswith("'") and val.endswith("'")
        ):
            val = val[1:-1]
        # only set if not already in environment
        if key not in os.environ:
            os.environ[key] = val


# Try to load a .env placed next to the package (plot_type_generator/.env)
_load_env_file(
    os.path.join(os.path.dirname(__file__), "..", "plot_type_generator", ".env")
)


def main():
    # Minimal compatible PlotGenState for choosing plots — we provide an
    # execution_plan so the chooser is grounded in the planner output.
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
        "execution_plan": (
            "STEP 1 - Data Characterization:\n"
            "The dataframe contains temporal data (date), categorical variables (product, region), and a numerical measure (sales_amount).\n"
            "STEP 2 - User Intent Analysis:\n"
            "User wants trends over time for product A in 2024.\n"
            "STEP 3 - Variable Relationship Query:\n"
            "Focus on date (x) vs sales_amount (y), filter product == 'A'.\n"
            "Generate a plan to aggregate monthly sales for product A and show time-series."
        ),
        "execution_plan": "",
        "code": "",
        "figure_path": "",
        "numeric_feedback": "",
        "lexical_feedback": "",
        "visual_feedback": "",
        "iteration_count": 0,
        "max_iterations": 3,
        "status": "pending",
        "llm_model": ENV_PLOT_TYPE_CHOOSER_AGENT_LLM_MODEL,
        "plot_recommendations": None,
        "suggestion_k": 3,
        "plot_recommendations_path": "./plot_recommendations.json",
    }

    # If no FEATHERLESS_API_KEY set, warn and exit early to avoid noisy errors
    if not ENV_FEATHERLESS_API_KEY:
        print(
            "FEATHERLESS_API_KEY is not set. Set it in the environment or in plot_type_generator/.env to run this test."
        )
        return

    try:
        new_state = plot_type_chooser_agent(state, k=3)
    except Exception as e:
        print("Plot chooser failed:", e)
        return

    print("--- Plot Recommendations ---")
    print(new_state.get("plot_recommendations"))


if __name__ == "__main__":
    main()
