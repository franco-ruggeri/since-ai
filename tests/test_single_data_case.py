"""
Quick test script to run a single test case from data.json.

Usage:
    python tests/test_single_data_case.py Rappuset
    python tests/test_single_data_case.py Korjauspyyntö
    python tests/test_single_data_case.py --list  # Show available test cases
"""
import os
import sys
import json
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from plot_type_generator.query_planning_agent import query_planning_agent
from plot_type_generator.plot_type_chooser_agent import plot_type_chooser_agent
from plot_type_generator.numeric_analysis_agent import numeric_analysis_agent
from plot_type_generator.lexical_analysis_agent import lexical_analysis_agent
from plot_type_generator.visual_appropriateness_agent import visual_appropriateness_agent
from plot_type_generator.plot_gen_state import PlotGenState
import plot_type_generator.utils as utils


def list_test_cases():
    """List all available test cases."""
    data_path = project_root / "data" / "data.json"
    with open(data_path, 'r', encoding='utf-8') as f:
        test_cases = json.load(f)

    print("\nAvailable test cases:")
    for i, (name, data) in enumerate(test_cases.items(), 1):
        print(f"  {i}. {name}")
        print(f"     Prompt: {data['prompt'][:100]}...")
        print(f"     Data: {len(data['data'])} rows")
        print()


def prepare_data_table(data):
    """Convert raw data to structured data table description."""
    df = pd.DataFrame(data)

    # Infer better dtypes
    for col in df.columns:
        if 'pvm' in col.lower() or 'date' in col.lower():
            try:
                df[col] = pd.to_datetime(df[col])
            except:
                pass

    data_table = {
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "shape": df.shape,
        "sample_rows": df.head(3).values.tolist(),
    }

    return data_table


def run_test_case(test_name: str):
    """Run a single test case."""
    # Load data
    data_path = project_root / "data" / "data.json"
    with open(data_path, 'r', encoding='utf-8') as f:
        test_cases = json.load(f)

    if test_name not in test_cases:
        print(f"❌ Test case '{test_name}' not found!")
        print("\nAvailable test cases:")
        for name in test_cases.keys():
            print(f"  - {name}")
        sys.exit(1)

    test_data = test_cases[test_name]
    prompt = test_data["prompt"]
    data = test_data["data"]

    print(f"\n{'='*80}")
    print(f"Running Test Case: {test_name}")
    print(f"{'='*80}")
    print(f"\nPrompt: {prompt}")
    print(f"\nData: {len(data)} rows")

    # Prepare data
    data_table = prepare_data_table(data)
    print(f"Columns: {data_table['columns']}")
    print(f"Dtypes: {data_table['dtypes']}")

    # Initialize state
    state: PlotGenState = {
        "user_query": prompt,
        "data_table": data_table,
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

    # Run pipeline
    print("\n" + "─"*80)
    print("STEP 1: Query Planning")
    print("─"*80)
    state = query_planning_agent(state)
    print(f"\n✓ Execution plan generated ({len(state.get('execution_plan', ''))} chars)")

    # Run 2 iterations
    for iteration in range(2):
        print("\n" + "="*80)
        print(f"ITERATION {iteration + 1}/2")
        print("="*80)

        # Plot type recommendation
        print("\nSTEP 2: Plot Type Recommendation")
        state = plot_type_chooser_agent(state, k=3)
        print("\n--- Recommendations ---")
        print(state.get("plot_recommendations"))

        all_passed = True

        # Numeric feedback
        print("\n" + "─"*80)
        print("STEP 3: Numeric Analysis")
        print("─"*80)
        state = numeric_analysis_agent(state)
        numeric_feedback = state.get("numeric_feedback", "")
        try:
            feedback = json.loads(numeric_feedback)
            print(f"\nStatus: {feedback.get('validation_status')}")
            if feedback.get("issues"):
                all_passed = False
                print(f"Issues: {len(feedback['issues'])}")
                for issue in feedback["issues"]:
                    print(f"  [{issue['severity']}] {issue['description']}")
                    print(f"    Fix: {issue['suggested_fix']}")
            else:
                print("No issues found!")
        except Exception as e:
            print(f"Could not parse feedback: {e}")

        # Lexical feedback
        print("\n" + "─"*80)
        print("STEP 4: Lexical Analysis")
        print("─"*80)
        state = lexical_analysis_agent(state)
        lexical_feedback = state.get("lexical_feedback", "")
        try:
            feedback = json.loads(lexical_feedback)
            print(f"\nStatus: {feedback.get('validation_status')}")
            if feedback.get("issues"):
                all_passed = False
                print(f"Issues: {len(feedback['issues'])}")
                for issue in feedback["issues"]:
                    print(f"  [{issue['severity']}] {issue['description']}")
                    print(f"    Current: {issue.get('current_value', 'N/A')}")
                    print(f"    Fix: {issue['suggested_fix']}")
            else:
                print("No issues found!")
        except Exception as e:
            print(f"Could not parse feedback: {e}")

        # Visual feedback
        print("\n" + "─"*80)
        print("STEP 5: Visual Appropriateness")
        print("─"*80)
        state = visual_appropriateness_agent(state)
        visual_feedback = state.get("visual_feedback", "")
        try:
            feedback = json.loads(visual_feedback)
            print(f"\nStatus: {feedback.get('validation_status')}")
            if feedback.get("issues"):
                all_passed = False
                print(f"Issues: {len(feedback['issues'])}")
                for issue in feedback["issues"]:
                    print(f"  [{issue['severity']}] {issue['description']}")
                    print(f"    Current: {issue.get('current_plot_type', 'N/A')}")
                    print(f"    Suggested: {issue.get('suggested_plot_type', 'N/A')}")
                    print(f"    Reasoning: {issue.get('reasoning', 'N/A')}")
            else:
                print("No issues found!")
        except Exception as e:
            print(f"Could not parse feedback: {e}")

        # Check if we should stop
        if all_passed:
            print("\n" + "="*80)
            print("✅ ALL VALIDATION PASSED!")
            print("="*80)
            break
        else:
            print("\n⚠️  Issues found, will regenerate in next iteration...")

    # Final summary
    print("\n" + "="*80)
    print("FINAL RECOMMENDATIONS")
    print("="*80)
    print(state.get("plot_recommendations"))

    return state


if __name__ == "__main__":
    load_dotenv()

    if not os.environ.get("FEATHERLESS_API_KEY"):
        print("ERROR: FEATHERLESS_API_KEY not set. Set it in .env or environment.")
        sys.exit(1)

    # Parse command line arguments
    if len(sys.argv) < 2 or sys.argv[1] == "--list":
        list_test_cases()
        sys.exit(0)

    test_name = sys.argv[1]

    try:
        state = run_test_case(test_name)
        print("\n✅ Test completed successfully!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
