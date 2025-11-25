import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from plot_type_generator.query_planning_agent import query_planning_agent
from plot_type_generator.plot_type_chooser_agent import plot_type_chooser_agent
from plot_type_generator.plot_gen_state import PlotGenState
import os
from env_vars import *


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

import os


def main():
    # Minimal compatible PlotGenState based on TypedDict definition
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
        "llm_model": ENV_QUERY_PLANNING_AGENT_LLM_MODEL,
        "plot_recommendations": None,
        "suggestion_k": 3,
        "plot_recommendations_path": "./plot_recommendations.json",
    }

    try:
        new_state = query_planning_agent(state)
    except Exception as e:
        print("Agent call failed:", e)
        return

    print("--- Execution Plan ---")
    print(new_state.get("execution_plan"))

    try:
        new_state = plot_type_chooser_agent(new_state, k=3)
    except Exception as e:
        print("Plot chooser failed:", e)
        return

    print("--- Plot Recommendations ---")
    print(new_state.get("plot_recommendations"))


if __name__ == "__main__":
    main()
