import os, sys

from pathlib import Path
from dotenv import load_dotenv

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from plot_type_generator.query_planning_agent import query_planning_agent
from plot_type_generator.plot_type_chooser_agent import plot_type_chooser_agent
from plot_type_generator.numeric_analysis_agent import numeric_analysis_agent
from plot_type_generator.lexical_analysis_agent import lexical_analysis_agent
from plot_type_generator.visual_appropriateness_agent import visual_appropriateness_agent
from plot_type_generator.plot_gen_state import PlotGenState
import plot_type_generator.utils as utils
import json


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

    # Multi-agent refinement loop
    max_iterations = state.get("max_iterations") or 3

    for iteration in range(max_iterations):
        state["iteration_count"] = iteration
        print(f"\n=== Iteration {iteration + 1}/{max_iterations} ===")

        # Generate plot recommendations
        try:
            state = plot_type_chooser_agent(state, k=state.get("suggestion_k") or 3)
        except Exception as e:
            print(f"Plot type chooser failed: {e}")
            return

        print("\n--- Plot Recommendations ---")
        print(state.get("plot_recommendations"))

        # Run feedback agents sequentially
        all_feedback_passed = True

        # Numeric Analysis
        try:
            state = numeric_analysis_agent(state)
            numeric_feedback = state.get("numeric_feedback", "")
            print("\n--- Numeric Feedback ---")
            print(numeric_feedback)

            # Check if issues were found
            try:
                feedback_json = json.loads(numeric_feedback)
                if feedback_json.get("validation_status") == "ISSUES_FOUND":
                    all_feedback_passed = False
                    print("⚠️  Numeric issues found")
            except Exception:
                pass
        except Exception as e:
            print(f"Numeric analysis failed: {e}")

        # Lexical Analysis
        try:
            state = lexical_analysis_agent(state)
            lexical_feedback = state.get("lexical_feedback", "")
            print("\n--- Lexical Feedback ---")
            print(lexical_feedback)

            try:
                feedback_json = json.loads(lexical_feedback)
                if feedback_json.get("validation_status") == "ISSUES_FOUND":
                    all_feedback_passed = False
                    print("⚠️  Lexical issues found")
            except Exception:
                pass
        except Exception as e:
            print(f"Lexical analysis failed: {e}")

        # Visual Appropriateness
        try:
            state = visual_appropriateness_agent(state)
            visual_feedback = state.get("visual_feedback", "")
            print("\n--- Visual Feedback ---")
            print(visual_feedback)

            try:
                feedback_json = json.loads(visual_feedback)
                if feedback_json.get("validation_status") == "ISSUES_FOUND":
                    all_feedback_passed = False
                    print("⚠️  Visual appropriateness issues found")
            except Exception:
                pass
        except Exception as e:
            print(f"Visual appropriateness analysis failed: {e}")

        # Check if we should continue iterating
        if all_feedback_passed:
            print("\n✅ All feedback agents passed! Recommendations are validated.")
            state["status"] = "completed"
            break
        else:
            print(f"\n🔄 Issues found. Regenerating recommendations (iteration {iteration + 1}/{max_iterations})...")
            # The next iteration will regenerate with the feedback in context

    # Save final recommendations
    print("\n--- Final Plot Recommendations ---")
    print(state.get("plot_recommendations"))
    if state.get("plot_recommendations"):
        parsed_text = utils.extract_json_content(state.get("plot_recommendations"))
        utils.save_recommendations(parsed_text)


if __name__ == "__main__":
    main()
