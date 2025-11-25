"""Test the full multi-agent plot type recommendation pipeline."""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from env_vars import *
from plot_type_generator.query_planning_agent import query_planning_agent
from plot_type_generator.plot_type_chooser_agent import plot_type_chooser_agent
from plot_type_generator.numeric_analysis_agent import numeric_analysis_agent
from plot_type_generator.lexical_analysis_agent import lexical_analysis_agent
from plot_type_generator.visual_appropriateness_agent import visual_appropriateness_agent
from plot_type_generator.plot_gen_state import PlotGenState
import plot_type_generator.utils as utils
import json


def test_simple_time_series():
    """Test with a simple time series query."""
    load_dotenv()

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
        "max_iterations": 2,
        "status": "pending",
        "llm_model": None,
        "suggestion_k": 3,
        "plot_recommendations": None,
        "plot_recommendations_path": None,
    }

    print("=" * 80)
    print("TEST: Simple Time Series")
    print("=" * 80)

    # Step 1: Query Planning
    print("\n[1] Running Query Planning Agent...")
    state = query_planning_agent(state)
    print(f"✓ Execution plan generated ({len(state.get('execution_plan', ''))} chars)")

    # Step 2: Plot Type Recommendation
    print("\n[2] Running Plot Type Chooser Agent...")
    state = plot_type_chooser_agent(state, k=3)
    print(f"✓ Recommendations generated")
    print(state.get("plot_recommendations"))

    # Step 3: Numeric Analysis
    print("\n[3] Running Numeric Analysis Agent...")
    state = numeric_analysis_agent(state)
    numeric_feedback = state.get("numeric_feedback", "")
    try:
        numeric_json = json.loads(numeric_feedback)
        print(f"✓ Numeric validation: {numeric_json.get('validation_status')}")
        if numeric_json.get('issues'):
            print(f"  Issues found: {len(numeric_json['issues'])}")
            for issue in numeric_json['issues'][:3]:
                print(f"  - [{issue['severity']}] {issue['description']}")
    except Exception as e:
        print(f"✗ Could not parse numeric feedback: {e}")

    # Step 4: Lexical Analysis
    print("\n[4] Running Lexical Analysis Agent...")
    state = lexical_analysis_agent(state)
    lexical_feedback = state.get("lexical_feedback", "")
    try:
        lexical_json = json.loads(lexical_feedback)
        print(f"✓ Lexical validation: {lexical_json.get('validation_status')}")
        if lexical_json.get('issues'):
            print(f"  Issues found: {len(lexical_json['issues'])}")
            for issue in lexical_json['issues'][:3]:
                print(f"  - [{issue['severity']}] {issue['description']}")
    except Exception as e:
        print(f"✗ Could not parse lexical feedback: {e}")

    # Step 5: Visual Appropriateness
    print("\n[5] Running Visual Appropriateness Agent...")
    state = visual_appropriateness_agent(state)
    visual_feedback = state.get("visual_feedback", "")
    try:
        visual_json = json.loads(visual_feedback)
        print(f"✓ Visual validation: {visual_json.get('validation_status')}")
        if visual_json.get('issues'):
            print(f"  Issues found: {len(visual_json['issues'])}")
            for issue in visual_json['issues'][:3]:
                print(f"  - [{issue['severity']}] {issue['description']}")
    except Exception as e:
        print(f"✗ Could not parse visual feedback: {e}")

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)

    return state


def test_categorical_comparison():
    """Test with a categorical comparison query."""
    load_dotenv()

    state: PlotGenState = {
        "user_query": "Compare sales amount across different regions",
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
                ["2024-01-03", "A", 50.0, "east"],
                ["2024-01-04", "C", 200.0, "west"],
            ],
        },
        "execution_plan": "",
        "code": "",
        "figure_path": "",
        "numeric_feedback": "",
        "lexical_feedback": "",
        "visual_feedback": "",
        "iteration_count": 0,
        "max_iterations": 2,
        "status": "pending",
        "llm_model": None,
        "suggestion_k": 3,
        "plot_recommendations": None,
        "plot_recommendations_path": None,
    }

    print("\n\n" + "=" * 80)
    print("TEST: Categorical Comparison")
    print("=" * 80)

    # Run through the full pipeline
    print("\n[1] Query Planning...")
    state = query_planning_agent(state)
    print("✓ Done")

    print("\n[2] Plot Type Recommendation...")
    state = plot_type_chooser_agent(state, k=3)
    print("✓ Done")
    print(state.get("plot_recommendations"))

    print("\n[3] Running All Feedback Agents...")
    state = numeric_analysis_agent(state)
    state = lexical_analysis_agent(state)
    state = visual_appropriateness_agent(state)
    print("✓ Done")

    # Summary
    print("\n--- FEEDBACK SUMMARY ---")
    for agent_type in ["numeric", "lexical", "visual"]:
        feedback = state.get(f"{agent_type}_feedback", "")
        try:
            feedback_json = json.loads(feedback)
            status = feedback_json.get("validation_status", "UNKNOWN")
            issue_count = len(feedback_json.get("issues", []))
            print(f"{agent_type.capitalize():15} {status:15} ({issue_count} issues)")
        except Exception:
            print(f"{agent_type.capitalize():15} PARSE_ERROR")

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)

    return state


if __name__ == "__main__":
    if not ENV_FEATHERLESS_API_KEY:
        print("ERROR: FEATHERLESS_API_KEY not set. Set it in .env or environment.")
        sys.exit(1)

    # Run tests
    print("Running multi-agent pipeline tests...\n")

    try:
        state1 = test_simple_time_series()
    except Exception as e:
        print(f"Test 1 failed: {e}")
        import traceback
        traceback.print_exc()

    try:
        state2 = test_categorical_comparison()
    except Exception as e:
        print(f"Test 2 failed: {e}")
        import traceback
        traceback.print_exc()

    print("\n\nAll tests completed!")
