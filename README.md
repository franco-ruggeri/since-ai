# Since AI

## 📋 Overview

This guide provides comprehensive instructions for implementing the intelligent
visualization system for your Bayer HSE data RAG application.

## 🏗️ Architecture Integration

### Current Flow

```
User Prompt → SQL Generation → DataFrame → LLM Summary
```

### Enhanced Flow

```
User Prompt → SQL Generation → DataFrame → [VISUALIZATION AGENT] → LLM Summary
```

## 🚀 Quick Start

### 1. File Structure

```
your_project/
├── visualization_agent.py          # Core agent (provided)
├── semantic_viz_extension.py       # Text analysis (provided)
├── streamlit_app.py                # Your existing app
└── requirements.txt                # Dependencies
```

### 2. Install Dependencies

```bash
pip install pandas plotly streamlit numpy
```

### 3. Minimal Integration (3 Lines!)

In your existing Streamlit app, add after SQL query execution:

```python
from visualization_agent import VisualizationAgent

viz_agent = VisualizationAgent()
fig, explanation = viz_agent.generate_visualization(user_prompt, df)
st.plotly_chart(fig, use_container_width=True)
```

## 📊 How It Works

### 1. Context Analysis

The agent analyzes:

- **User Intent**: Comparison, trend, distribution, correlation, aggregation
- **Data Structure**: Column types, data shape, temporal presence
- **HSE Domain**: Safety-specific patterns and requirements

### 2. Visualization Selection

Decision logic:

- **Temporal data + trend keywords** → Line chart
- **Categories + numeric values** → Bar chart
- **Two numeric columns + relationship** → Scatter plot
- **Single numeric + distribution** → Histogram/Box plot
- **Multi-category comparison** → Grouped bar chart

### 3. Data Preprocessing

Automatic handling of:

- Date parsing and sorting
- Aggregations (GROUP BY equivalents)
- Category limiting (top N)
- Missing value treatment

### 4. Chart Generation

Uses Plotly for:

- Interactive visualizations
- Professional appearance
- Streamlit compatibility
- Export capabilities

## 🎯 Key Features

### ✅ Generalization

- No hardcoding for specific datasets
- Adapts to any dataframe structure
- Context-aware decision making

### ✅ Text Data Support

- Semantic analysis for text columns
- Automatic categorization
- Keyword extraction and frequency analysis
- HSE taxonomy built-in

### ✅ Smart Defaults

- Automatic chart type selection
- Intelligent axis assignment
- Appropriate aggregations
- Readable formatting

## 💡 Usage Examples

### Example 1: Temporal Trends

```python
prompt = "Show incident trends over the last 12 months"
df = pd.DataFrame({
    'date': pd.date_range('2024-01-01', periods=12, freq='M'),
    'incidents': [23, 19, 25, 18, 22, 20, 17, 21, 24, 19, 23, 20]
})

fig, explanation = agent.generate_visualization(prompt, df)
# Result: Line chart with time on x-axis, incidents on y-axis
```

### Example 2: Department Comparison

```python
prompt = "Compare incident rates by department"
df = pd.DataFrame({
    'department': ['Safety', 'Operations', 'Maintenance', 'Quality'],
    'incidents': [45, 32, 28, 15]
})

fig, explanation = agent.generate_visualization(prompt, df)
# Result: Bar chart comparing departments
```

### Example 3: Text Analysis

```python
prompt = "Categorize incident descriptions by type"
df = pd.DataFrame({
    'description': [
        "Slip on wet floor in production area",
        "Chemical spill in laboratory",
        "Equipment malfunction in assembly line"
    ]
})

# Use enhanced agent for text
from semantic_viz_extension import EnhancedVisualizationAgent
enhanced_agent = EnhancedVisualizationAgent()
fig, explanation = enhanced_agent.generate_visualization(prompt, df)
# Result: Categorized bar chart with semantic analysis
```

## 🔧 Customization Options

### 1. Modify Chart Preferences

```python
class CustomVisualizationAgent(VisualizationAgent):
    def _select_visualization(self, analysis):
        config = super()._select_visualization(analysis)

        # Force specific chart types for certain scenarios
        if 'safety' in analysis['prompt_lower']:
            config['chart_type'] = 'bar'  # Always use bar for safety data

        return config
```

### 2. Add Custom HSE Categories

```python
# In semantic_viz_extension.py, extend hse_taxonomy:
self.hse_taxonomy['custom_category'] = {
    'my_category': ['keyword1', 'keyword2', 'keyword3']
}
```

### 3. Adjust Preprocessing

```python
def _preprocess_data(self, df, viz_config):
    df = super()._preprocess_data(df, viz_config)

    # Add custom preprocessing
    if 'severity' in df.columns:
        df = df[df['severity'] >= 2]  # Filter low severity

    return df
```

## 🎨 Visualization Types Supported

| Type         | Best For            | Example Use Case            |
| ------------ | ------------------- | --------------------------- |
| Line Chart   | Temporal trends     | Incidents over time         |
| Bar Chart    | Category comparison | Incidents by department     |
| Scatter Plot | Correlations        | Training hours vs incidents |
| Histogram    | Distributions       | Severity score distribution |
| Box Plot     | Statistical spread  | Response time variations    |

## ⚠️ Edge Cases & Error Handling

### Empty DataFrames

```python
if df.empty:
    st.info("No data available to visualize")
    return None, "No data"
```

### Too Many Categories

The agent automatically limits to top 20 categories for readability.

### Missing Columns

Error handling returns informative messages instead of crashing.

### Text-Only Data

Semantic analyzer extracts categorical information from text.

## 📈 Performance Considerations

### Caching

```python
# Cache agent initialization
if 'viz_agent' not in st.session_state:
    st.session_state.viz_agent = VisualizationAgent()
```

### Large Datasets

```python
# Sample large datasets before visualization
if len(df) > 10000:
    df = df.sample(n=10000, random_state=42)
```

## 🧪 Testing Strategy

### Test Cases Included

1. **Temporal data**: Date columns with numeric values
2. **Categorical data**: Text categories with counts
3. **Distribution data**: Single numeric column
4. **Correlation data**: Two numeric columns
5. **Text-heavy data**: Incident descriptions

### Running Tests

```python
from visualization_agent import test_visualization_agent
figures = test_visualization_agent()
```

## 🔄 Integration Points

### Point 1: After SQL Execution

```python
# Your existing code
df = execute_sql_query(generated_sql)

# Add visualization
fig, explanation = viz_agent.generate_visualization(user_prompt, df)
st.plotly_chart(fig)
```

### Point 2: Before LLM Summary

```python
# Generate visualization
viz_section = generate_visualization(user_prompt, df)

# Then generate LLM summary (your existing code)
summary = llm_summarize(user_prompt, df)
```

## 🎯 Hackathon Submission Checklist

- [ ] Core `VisualizationAgent` class implemented
- [ ] Text analysis extension created
- [ ] Streamlit integration working
- [ ] Handles all sample data scenarios
- [ ] No hardcoding for specific datasets
- [ ] Error handling implemented
- [ ] Documentation provided
- [ ] Test cases pass
- [ ] Demo ready

## 📝 Tips for Success

### ✅ DO:

- Test with the provided sample data
- Handle edge cases gracefully
- Keep visualizations simple and clear
- Use built-in Streamlit/Plotly features
- Document your approach

### ❌ DON'T:

- Hardcode chart types for specific data
- Ignore text data (use semantic analysis!)
- Create overly complex visualizations
- Skip error handling
- Forget to test with various prompts

## 🚀 Advanced Features (Optional)

### Multi-Chart Layouts

```python
col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(fig1)
with col2:
    st.plotly_chart(fig2)
```

### Interactive Filters

```python
selected_dept = st.selectbox("Department", df['department'].unique())
filtered_df = df[df['department'] == selected_dept]
fig, _ = agent.generate_visualization(prompt, filtered_df)
```

### Export Capabilities

```python
# Plotly charts can be downloaded as PNG/HTML
config = {'displayModeBar': True, 'displaylogo': False}
st.plotly_chart(fig, config=config)
```

## 📞 Support & Resources

- **Plotly Documentation**: <https://plotly.com/python/>
- **Streamlit Docs**: <https://docs.streamlit.io/>
- **Pandas Guide**: <https://pandas.pydata.org/docs/>

## 🎓 Learning Resources

1. Study the provided reference implementation
2. Review the semantic analysis techniques
3. Experiment with different prompt patterns
4. Test edge cases thoroughly

---

**Good luck with your hackathon! 🚀**

Remember: The goal is to create a generalizable solution that works for ANY HSE
data scenario, not just the examples provided.
