"""Test the text extraction and data processing capabilities."""
import os
import sys
import json
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from plot_type_generator.query_planning_agent import query_planning_agent
from plot_type_generator.plot_type_chooser_agent import plot_type_chooser_agent
from plot_type_generator.plot_gen_state import PlotGenState
from plot_type_generator.utils import extract_json_content

load_dotenv()

# Create output file
output_dir = project_root / "test_results"
output_dir.mkdir(exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = output_dir / f"stairways_test_{timestamp}.txt"

# Redirect output to both console and file
class Tee:
    def __init__(self, *files):
        self.files = files
    def write(self, obj):
        for f in self.files:
            f.write(obj)
            f.flush()
    def flush(self):
        for f in self.files:
            f.flush()

output_f = open(output_file, 'w')
original_stdout = sys.stdout
sys.stdout = Tee(sys.stdout, output_f)

# Load Stairways test case from data_english.json
data_path = project_root / "data" / "data_english.json"
with open(data_path, 'r') as f:
    data = json.load(f)

# Get specifically the 'Stairways' test case
test_case = data['Stairways']

print("="*80)
print("STAIRWAYS TEST CASE")
print("="*80)
print(f"Test Case: Stairways")
print(f"Prompt: {test_case['prompt']}\n")

# Prepare data with sample observations
raw_data = test_case['data']
print(f"Raw data: {len(raw_data)} observations")
print(f"Columns: {list(raw_data[0].keys())}")
print(f"\nSample observations:")
for i, obs in enumerate(raw_data[:3], 1):
    print(f"{i}. Title: {obs['Title']}")
    print(f"   Observation: {obs['Observation'][:80]}...")
print()

# Create data table description
data_table = {
    "columns": list(raw_data[0].keys()),
    "dtypes": {
        "Title": "string",
        "Observation": "string",
        "observation_date": "datetime",
        "observation_handled_date": "datetime"
    },
    "total_rows": len(raw_data),
    "sample_observations": [
        f"Title: {obs['Title']}, Observation: {obs['Observation'][:100]}..."
        for obs in raw_data[:5]
    ]
}

# Initialize state
state: PlotGenState = {
    "user_query": test_case['prompt'],
    "data_table": data_table,
    "execution_plan": "",
    "code": "",
    "figure_path": "",
    "numeric_feedback": "",
    "lexical_feedback": "",
    "visual_feedback": "",
    "iteration_count": 0,
    "max_iterations": 1,
    "status": "pending",
    "llm_model": None,
    "suggestion_k": 3,
    "plot_recommendations": None,
    "plot_recommendations_path": None,
    "processed_data": None,
}

print("="*80)
print("STEP 1: Query Planning")
print("="*80)
state = query_planning_agent(state)
print(f"\n✓ Execution plan generated ({len(state.get('execution_plan', ''))} chars)")
print("\n--- Execution Plan ---")
print(state.get("execution_plan"))

print("\n" + "="*80)
print("STEP 2: Plot Type Recommendation & Data Processing")
print("="*80)
state = plot_type_chooser_agent(state, k=3)

print("\n--- Plot Recommendations ---")
plot_recommendations_raw = state.get("plot_recommendations")
print(f"Raw value type: {type(plot_recommendations_raw)}")
print(f"Raw value length: {len(plot_recommendations_raw) if plot_recommendations_raw else 0} chars")

if plot_recommendations_raw:
    try:
        recommendations = extract_json_content(plot_recommendations_raw)
        print(json.dumps(recommendations, indent=2))
    except Exception as e:
        print(f"❌ Failed to parse JSON: {e}")
        print(f"Content preview: {plot_recommendations_raw[:500]}")
        sys.exit(1)
else:
    print("❌ plot_recommendations is None or empty!")
    sys.exit(1)

if state.get("processed_data"):
    print("\n" + "="*80)
    print("✅ PROCESSED DATA GENERATED")
    print("="*80)
    processed = state.get("processed_data")
    print(f"\nColumns: {processed.get('columns')}")
    print(f"Data rows: {len(processed.get('data', []))}")
    print("\nProcessed DataFrame:")
    print(json.dumps(processed, indent=2))
else:
    print("\n⚠️ No processed data was generated")

if 'recommendations' in locals():
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Visualization Type: {recommendations.get('visualization_type')}")
    print(f"Aggregation: {recommendations.get('aggregation')}")
    print(f"X-Axis: {recommendations.get('x_axis')}")
    print(f"Y-Axis: {recommendations.get('y_axis')}")
    print(f"\nPreprocessing Steps:")
    for i, step in enumerate(recommendations.get('preprocessing_steps', []), 1):
        print(f"{i}. {step}")

    if state.get("processed_data"):
        processed = state.get("processed_data")
        print(f"\n✅ Ready to plot with {len(processed.get('data', []))} data points")

# Cleanup and close output file
print("\n" + "="*80)
print(f"Test completed successfully!")
print(f"Output saved to: {output_file}")
print("="*80)

sys.stdout = original_stdout
output_f.close()

print(f"\n✅ Test output saved to: {output_file}")
