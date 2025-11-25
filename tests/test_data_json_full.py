"""
Comprehensive test script for all test cases in data/data.json.

This script:
1. Loads all test cases from data.json
2. Converts each dataset to proper format with inferred dtypes
3. Runs the full multi-agent pipeline on each test case
4. Collects and reports validation results
5. Saves individual and aggregate results
"""

import os
import sys
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
from dotenv import load_dotenv

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from env_vars import *
from plot_type_generator.query_planning_agent import query_planning_agent
from plot_type_generator.plot_type_chooser_agent import plot_type_chooser_agent
from plot_type_generator.numeric_analysis_agent import numeric_analysis_agent
from plot_type_generator.lexical_analysis_agent import lexical_analysis_agent
from plot_type_generator.visual_appropriateness_agent import (
    visual_appropriateness_agent,
)
from plot_type_generator.plot_gen_state import PlotGenState
import plot_type_generator.utils as utils


def load_test_cases(data_path: str) -> Dict[str, Dict[str, Any]]:
    """Load all test cases from data.json."""
    with open(data_path, "r", encoding="utf-8") as f:
        return json.load(f)


def prepare_data_table(data: List[Dict]) -> Dict[str, Any]:
    """Convert raw data to structured data table description."""
    df = pd.DataFrame(data)

    # Infer better dtypes
    for col in df.columns:
        # Try to convert date columns
        if "pvm" in col.lower() or "date" in col.lower():
            try:
                df[col] = pd.to_datetime(df[col])
            except:
                pass

    # Create data table structure
    data_table = {
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "shape": df.shape,
        "sample_rows": df.head(5).values.tolist(),
        "summary": {"total_rows": len(df), "date_range": None},
    }

    # Add date range if date columns exist
    date_cols = [col for col in df.columns if df[col].dtype == "datetime64[ns]"]
    if date_cols:
        date_col = date_cols[0]
        data_table["summary"]["date_range"] = {
            "start": str(df[date_col].min()),
            "end": str(df[date_col].max()),
        }

    return data_table


def run_pipeline_on_test_case(
    test_name: str,
    prompt: str,
    data: List[Dict],
    max_iterations: int = 2,
    suggestion_k: int = 3,
) -> Dict[str, Any]:
    """Run the full multi-agent pipeline on a single test case."""
    print(f"\n{'='*80}")
    print(f"TEST CASE: {test_name}")
    print(f"{'='*80}")
    print(f"Prompt: {prompt[:150]}...")

    # Prepare data table
    data_table = prepare_data_table(data)
    print(f"Data: {data_table['shape'][0]} rows × {data_table['shape'][1]} columns")
    print(f"Columns: {data_table['columns']}")

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
        "max_iterations": max_iterations,
        "status": "pending",
        "llm_model": None,
        "suggestion_k": suggestion_k,
        "plot_recommendations": None,
        "plot_recommendations_path": None,
    }

    result = {
        "test_name": test_name,
        "prompt": prompt,
        "data_shape": data_table["shape"],
        "timestamp": datetime.now().isoformat(),
        "success": False,
        "iterations": [],
        "final_status": None,
        "errors": [],
    }

    try:
        # Step 1: Query Planning
        print("\n[1/5] Running Query Planning Agent...")
        state = query_planning_agent(state)
        print(
            f"✓ Execution plan generated ({len(state.get('execution_plan', ''))} chars)"
        )

        # Multi-agent refinement loop
        for iteration in range(max_iterations):
            print(f"\n{'─'*80}")
            print(f"ITERATION {iteration + 1}/{max_iterations}")
            print(f"{'─'*80}")

            state["iteration_count"] = iteration
            iteration_result = {
                "iteration": iteration + 1,
                "recommendations": None,
                "feedback": {
                    "numeric": {"status": None, "issues": 0},
                    "lexical": {"status": None, "issues": 0},
                    "visual": {"status": None, "issues": 0},
                },
                "all_passed": False,
            }

            # Step 2: Plot Type Recommendation
            print("\n[2/5] Running Plot Type Chooser Agent...")
            try:
                state = plot_type_chooser_agent(state, k=suggestion_k)
                recommendations = state.get("plot_recommendations")
                iteration_result["recommendations"] = recommendations
                print(f"✓ Recommendations generated")

                # Try to parse and show plot type
                try:
                    rec_json = utils.extract_json_content(recommendations)
                    print(f"  Primary plot type: {rec_json.get('plot_type', 'N/A')}")
                except:
                    pass
            except Exception as e:
                print(f"✗ Plot type chooser failed: {e}")
                result["errors"].append(f"Iteration {iteration+1} - Chooser: {str(e)}")
                break

            all_feedback_passed = True

            # Step 3: Numeric Analysis
            print("\n[3/5] Running Numeric Analysis Agent...")
            try:
                state = numeric_analysis_agent(state)
                numeric_feedback = state.get("numeric_feedback", "")
                feedback_json = json.loads(numeric_feedback)

                status = feedback_json.get("validation_status")
                issues = len(feedback_json.get("issues", []))

                iteration_result["feedback"]["numeric"]["status"] = status
                iteration_result["feedback"]["numeric"]["issues"] = issues

                if status == "ISSUES_FOUND":
                    all_feedback_passed = False
                    print(f"⚠️  Numeric validation: {status} ({issues} issues)")
                    for issue in feedback_json.get("issues", [])[:2]:
                        print(
                            f"    - [{issue['severity']}] {issue['description'][:80]}..."
                        )
                else:
                    print(f"✓ Numeric validation: {status}")
            except Exception as e:
                print(f"✗ Numeric analysis failed: {e}")
                result["errors"].append(f"Iteration {iteration+1} - Numeric: {str(e)}")

            # Step 4: Lexical Analysis
            print("\n[4/5] Running Lexical Analysis Agent...")
            try:
                state = lexical_analysis_agent(state)
                lexical_feedback = state.get("lexical_feedback", "")
                feedback_json = json.loads(lexical_feedback)

                status = feedback_json.get("validation_status")
                issues = len(feedback_json.get("issues", []))

                iteration_result["feedback"]["lexical"]["status"] = status
                iteration_result["feedback"]["lexical"]["issues"] = issues

                if status == "ISSUES_FOUND":
                    all_feedback_passed = False
                    print(f"⚠️  Lexical validation: {status} ({issues} issues)")
                    for issue in feedback_json.get("issues", [])[:2]:
                        print(
                            f"    - [{issue['severity']}] {issue['description'][:80]}..."
                        )
                else:
                    print(f"✓ Lexical validation: {status}")
            except Exception as e:
                print(f"✗ Lexical analysis failed: {e}")
                result["errors"].append(f"Iteration {iteration+1} - Lexical: {str(e)}")

            # Step 5: Visual Appropriateness
            print("\n[5/5] Running Visual Appropriateness Agent...")
            try:
                state = visual_appropriateness_agent(state)
                visual_feedback = state.get("visual_feedback", "")
                feedback_json = json.loads(visual_feedback)

                status = feedback_json.get("validation_status")
                issues = len(feedback_json.get("issues", []))

                iteration_result["feedback"]["visual"]["status"] = status
                iteration_result["feedback"]["visual"]["issues"] = issues

                if status == "ISSUES_FOUND":
                    all_feedback_passed = False
                    print(f"⚠️  Visual validation: {status} ({issues} issues)")
                    for issue in feedback_json.get("issues", [])[:2]:
                        print(
                            f"    - [{issue['severity']}] {issue['description'][:80]}..."
                        )
                else:
                    print(f"✓ Visual validation: {status}")
            except Exception as e:
                print(f"✗ Visual appropriateness analysis failed: {e}")
                result["errors"].append(f"Iteration {iteration+1} - Visual: {str(e)}")

            iteration_result["all_passed"] = all_feedback_passed
            result["iterations"].append(iteration_result)

            # Check if we should continue
            if all_feedback_passed:
                print(f"\n✅ All feedback agents passed!")
                state["status"] = "completed"
                result["final_status"] = "completed"
                result["success"] = True
                break
            else:
                print(f"\n🔄 Issues found. Will regenerate in next iteration...")

        # If we exhausted iterations without passing
        if not result["success"]:
            result["final_status"] = "max_iterations_reached"
            print(f"\n⚠️  Max iterations reached without full validation pass")

        # Save final recommendations
        if state.get("plot_recommendations"):
            result["final_recommendations"] = state.get("plot_recommendations")
            try:
                result["final_recommendations_parsed"] = utils.extract_json_content(
                    state.get("plot_recommendations")
                )
            except:
                pass

    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}")
        import traceback

        traceback.print_exc()
        result["errors"].append(f"Pipeline error: {str(e)}")
        result["final_status"] = "error"

    return result


def generate_summary_report(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate aggregate summary of all test results."""
    total_tests = len(results)
    successful_tests = sum(1 for r in results if r["success"])
    failed_tests = total_tests - successful_tests

    # Count validation issues by type
    total_numeric_issues = 0
    total_lexical_issues = 0
    total_visual_issues = 0

    for result in results:
        for iteration in result.get("iterations", []):
            total_numeric_issues += iteration["feedback"]["numeric"].get("issues", 0)
            total_lexical_issues += iteration["feedback"]["lexical"].get("issues", 0)
            total_visual_issues += iteration["feedback"]["visual"].get("issues", 0)

    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_tests": total_tests,
        "successful_tests": successful_tests,
        "failed_tests": failed_tests,
        "success_rate": f"{(successful_tests/total_tests*100):.1f}%",
        "total_issues_found": {
            "numeric": total_numeric_issues,
            "lexical": total_lexical_issues,
            "visual": total_visual_issues,
            "total": total_numeric_issues + total_lexical_issues + total_visual_issues,
        },
        "test_details": [],
    }

    for result in results:
        summary["test_details"].append(
            {
                "name": result["test_name"],
                "success": result["success"],
                "final_status": result["final_status"],
                "iterations": len(result["iterations"]),
                "errors": len(result["errors"]),
            }
        )

    return summary


def main():
    """Run tests on all data.json entries."""
    load_dotenv()

    if not ENV_FEATHERLESS_API_KEY:
        print("ERROR: FEATHERLESS_API_KEY not set. Set it in .env or environment.")
        sys.exit(1)

    # Load test cases
    data_path = project_root / "data" / "data.json"
    print(f"Loading test cases from: {data_path}")
    test_cases = load_test_cases(str(data_path))
    print(f"Found {len(test_cases)} test cases: {list(test_cases.keys())}\n")

    # Run tests
    results = []

    for test_name, test_data in test_cases.items():
        result = run_pipeline_on_test_case(
            test_name=test_name,
            prompt=test_data["prompt"],
            data=test_data["data"],
            max_iterations=2,
            suggestion_k=3,
        )
        results.append(result)

    # Generate summary
    print("\n" + "=" * 80)
    print("SUMMARY REPORT")
    print("=" * 80)

    summary = generate_summary_report(results)

    print(f"\nTotal tests: {summary['total_tests']}")
    print(f"Successful: {summary['successful_tests']} ({summary['success_rate']})")
    print(f"Failed: {summary['failed_tests']}")

    print(f"\nTotal issues found across all tests:")
    print(f"  Numeric:  {summary['total_issues_found']['numeric']}")
    print(f"  Lexical:  {summary['total_issues_found']['lexical']}")
    print(f"  Visual:   {summary['total_issues_found']['visual']}")
    print(f"  TOTAL:    {summary['total_issues_found']['total']}")

    print(f"\nTest Details:")
    for detail in summary["test_details"]:
        status_icon = "✓" if detail["success"] else "✗"
        print(
            f"  {status_icon} {detail['name']:20} - {detail['final_status']:25} ({detail['iterations']} iterations)"
        )

    # Save results
    results_dir = project_root / "test_results"
    results_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%SZ")

    # Save detailed results
    results_file = results_dir / f"data_json_test_results_{timestamp}.json"
    with open(results_file, "w", encoding="utf-8") as f:
        json.dump(
            {"summary": summary, "detailed_results": results},
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"\n📊 Results saved to: {results_file}")

    return results, summary


if __name__ == "__main__":
    results, summary = main()

    # Exit with appropriate code
    if summary["failed_tests"] > 0:
        print(f"\n  {summary['failed_tests']} test(s) failed")
        sys.exit(1)
    else:
        print(f"\n✅ All {summary['total_tests']} tests passed!")
        sys.exit(0)
