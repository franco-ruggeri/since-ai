import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional, Dict, Any

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from plot_type_generator.query_planning_agent import query_planning_agent
from plot_type_generator.plot_type_chooser_agent import plot_type_chooser_agent
from plot_type_generator.numeric_analysis_agent import numeric_analysis_agent
from plot_type_generator.lexical_analysis_agent import lexical_analysis_agent
from plot_type_generator.visual_appropriateness_agent import (
    visual_appropriateness_agent,
)
from plot_type_generator.plot_gen_state import PlotGenState
import plot_type_generator.utils as utils


def run_plot_generation_pipeline(
    user_query: str,
    data_table: Dict[str, Any],
    max_iterations: int = 3,
    suggestion_k: int = 3,
    verbose: bool = True,
    output_callback=None,
) -> PlotGenState:
    """
    Run the complete plot generation pipeline.

    Args:
        user_query: The user's visualization request
        data_table: Dictionary describing the data with keys:
            - columns: list of column names
            - dtypes: dict mapping column names to data types
            - sample_rows (optional): list of sample data rows
            - total_rows (optional): total number of rows
        max_iterations: Maximum number of refinement iterations (default: 3)
        suggestion_k: Number of plot type suggestions to generate (default: 3)
        verbose: Whether to print progress messages (default: True)

    Returns:
        Dictionary containing processed data ready for plotting with structure:
        {
            "columns": [...],
            "data": [[row1], [row2], ...]
        }
        Returns None if the pipeline fails.
    """
    load_dotenv()

    # Validate API key based on provider
    provider = os.environ.get("LLM_PROVIDER", "featherless").lower()
    if provider == "featherless":
        if not os.environ.get("FEATHERLESS_API_KEY"):
            raise ValueError(
                "FEATHERLESS_API_KEY not set. Set it in the environment or .env file"
            )
    elif provider in ("gemini", "google"):
        if not os.environ.get("GOOGLE_API_KEY"):
            raise ValueError(
                "GOOGLE_API_KEY not set. Set it in the environment or .env file"
            )
    else:
        raise ValueError(
            f"Invalid LLM_PROVIDER: {provider}. Supported: featherless, gemini"
        )

    # Helper function to write output
    def write_output(message: str):
        if verbose:
            print(message)
        if output_callback:
            output_callback(message)
    
    # Initialize state
    state: PlotGenState = {
        "user_query": user_query,
        "data_table": data_table,
        "execution_plan": "",
        "code": "",
        "figure_path": "",
        "numeric_feedback": "",
        "lexical_feedback": "",
        "visual_feedback": "",
        "iteration_count": 0,
        "max_iterations": max_iterations,
        "status": "pending",
        "llm_model": None,
        "suggestion_k": suggestion_k,
        "plot_recommendations": None,
        "plot_recommendations_path": "./plot_recommendations.json",
        "processed_data": None,
    }

    # Step 1: Query Planning
    write_output("=" * 80)
    write_output("STEP 1: Query Planning")
    write_output("=" * 80)

    try:
        state = query_planning_agent(state)
        write_output(
            f"✓ Execution plan generated ({len(state.get('execution_plan', ''))} chars)"
        )
        write_output("\n--- Execution Plan ---")
        write_output(state.get("execution_plan"))
    except Exception as e:
        write_output(f"❌ Query planning agent failed: {e}")
        return None

    # Step 2: Multi-agent refinement loop
    for iteration in range(max_iterations):
        state["iteration_count"] = iteration

        write_output(f"\n{'=' * 80}")
        write_output(f"ITERATION {iteration + 1}/{max_iterations}")
        write_output("=" * 80)

        # Generate plot recommendations
        try:
            state = plot_type_chooser_agent(state, k=suggestion_k)

            write_output("\n--- Plot Recommendations ---")
            recommendations_raw = state.get("plot_recommendations")
            if recommendations_raw:
                try:
                    recommendations = utils.extract_json_content(
                        recommendations_raw
                    )
                    write_output(json.dumps(recommendations, indent=2))
                except Exception as e:
                    write_output(f"Warning: Could not parse recommendations: {e}")
        except Exception as e:
            write_output(f"❌ Plot type chooser failed: {e}")
            return None

        # Show processed data if generated
        if state.get("processed_data"):
            write_output("\n--- Processed Data Generated ---")
            processed = state.get("processed_data")
            write_output(f"Columns: {processed.get('columns')}")
            write_output(f"Data rows: {len(processed.get('data', []))}")

        # Run feedback agents
        all_feedback_passed = True

        # Numeric Analysis
        try:
            state = numeric_analysis_agent(state)
            numeric_feedback = state.get("numeric_feedback", "")

            write_output("\n--- Numeric Feedback ---")
            write_output(numeric_feedback)

            try:
                feedback_json = json.loads(numeric_feedback)
                if feedback_json.get("validation_status") == "ISSUES_FOUND":
                    all_feedback_passed = False
                    write_output("⚠️  Numeric issues found")
            except Exception:
                pass
        except Exception as e:
            write_output(f"Warning: Numeric analysis failed: {e}")

        # Lexical Analysis
        try:
            state = lexical_analysis_agent(state)
            lexical_feedback = state.get("lexical_feedback", "")

            write_output("\n--- Lexical Feedback ---")
            write_output(lexical_feedback)

            try:
                feedback_json = json.loads(lexical_feedback)
                if feedback_json.get("validation_status") == "ISSUES_FOUND":
                    all_feedback_passed = False
                    write_output("⚠️  Lexical issues found")
            except Exception:
                pass
        except Exception as e:
            write_output(f"Warning: Lexical analysis failed: {e}")

        # Visual Appropriateness
        try:
            state = visual_appropriateness_agent(state)
            visual_feedback = state.get("visual_feedback", "")

            write_output("\n--- Visual Feedback ---")
            write_output(visual_feedback)

            try:
                feedback_json = json.loads(visual_feedback)
                if feedback_json.get("validation_status") == "ISSUES_FOUND":
                    all_feedback_passed = False
                    write_output("⚠️  Visual appropriateness issues found")
            except Exception:
                pass
        except Exception as e:
            write_output(f"Warning: Visual appropriateness analysis failed: {e}")

        # Check if we should continue iterating
        if all_feedback_passed:
            write_output("\n✅ All feedback agents passed! Recommendations are validated.")
            state["status"] = "completed"
            break
        else:
            write_output(f"\n🔄 Issues found. Regenerating recommendations...")

    # Save final recommendations
    if state.get("plot_recommendations"):
        try:
            parsed_recommendations = utils.extract_json_content(
                state.get("plot_recommendations")
            )
            utils.save_recommendations(parsed_recommendations)
        except Exception as e:
            write_output(f"Warning: Could not save recommendations: {e}")

    processed_data = state.get("processed_data")

    return state


def main():
    """Example usage of the plot generation pipeline."""

    # Load Stairways test case from data_english.json
    data_path = project_root / "data" / "data_english.json"
    with open(data_path, "r") as f:
        data = json.load(f)

    # Get Specifically for a particular data test case
    test_case = data["Lokaatio"]
    raw_data = test_case["Data"]
    user_query = (
        test_case["prompt"] + " Translation/Summary: " + test_case["translation"]
    )

    # Run the pipeline
    processed_data = run_plot_generation_pipeline(
        user_query=user_query,
        data_table=test_case["Data"],
        max_iterations=2,
        suggestion_k=1,
        verbose=True,
    )
    if processed_data:
        print("\n" + "=" * 80)
        print("PROCESSED DATA READY FOR PLOTTING")
        print("=" * 80)
        print(processed_data["plot_recommendations"])
        print(processed_data["processed_data"])
    else:
        print("\n❌ Pipeline failed to generate processed data")


if __name__ == "__main__":
    main()
