"""
Streamlit Integration for HSE Visualization Agent

This module shows how to integrate the VisualizationAgent into your existing
Streamlit application architecture.
"""

import streamlit as st
import pandas as pd
from visualization_agent import VisualizationAgent


def integrate_visualization_to_existing_flow(user_prompt: str, df: pd.DataFrame):
    """
    This function should be called after your existing RAG pipeline returns
    the dataframe and before the final LLM summarization.

    Integration point in your existing architecture:
    1. User enters prompt
    2. SQL is generated
    3. DataFrame is retrieved
    4. >>> CALL THIS FUNCTION HERE <<<
    5. Final LLM summarization

    Args:
        user_prompt: The original user prompt
        df: The dataframe returned from your SQL query
    """

    # Initialize the agent (you might want to cache this)
    if "viz_agent" not in st.session_state:
        st.session_state.viz_agent = VisualizationAgent()

    agent = st.session_state.viz_agent

    # Check if dataframe is suitable for visualization
    if df is None or df.empty:
        st.info("No data available to visualize.")
        return

    # Check if dataframe has enough information for visualization
    if len(df.columns) < 1:
        return

    # Generate visualization
    try:
        with st.spinner("Generating visualization..."):
            fig, explanation = agent.generate_visualization(user_prompt, df)

        # Display the visualization
        st.subheader("📊 Visualization")
        st.plotly_chart(fig, use_container_width=True)

        # Display explanation
        with st.expander("ℹ️ About this visualization"):
            st.write(explanation)

    except Exception as e:
        st.warning(f"Could not generate visualization: {str(e)}")


def main_streamlit_app():
    """
    Example of a complete Streamlit app integrating the visualization agent.
    This demonstrates the full flow from your architecture diagram.
    """

    st.set_page_config(page_title="HSE Data Analysis", layout="wide")
    st.title("🏭 HSE Data Analysis with Smart Visualization")

    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")
        enable_viz = st.checkbox("Enable Smart Visualization", value=True)
        st.markdown("---")
        st.markdown("### About")
        st.info(
            "This app combines RAG-based SQL generation with intelligent data visualization."
        )

    # Main input
    st.subheader("Ask a question about HSE data")
    user_prompt = st.text_input(
        "Enter your query:",
        placeholder="e.g., Show me incident trends over the last year",
        key="user_prompt",
    )

    # Simulate the existing RAG flow
    if st.button("Analyze", type="primary") and user_prompt:
        # STEP 1: Display user prompt (your existing step)
        st.markdown("### 📝 Your Query")
        st.write(user_prompt)

        # STEP 2: Simulate SQL generation (your existing step)
        with st.expander("🔧 Generated SQL", expanded=False):
            # In your real app, this comes from your LLM agent
            simulated_sql = "SELECT * FROM hse_incidents WHERE date >= '2024-01-01'"
            st.code(simulated_sql, language="sql")

        # STEP 3: Simulate dataframe retrieval (your existing step)
        # In your real app, this comes from your database
        simulated_df = simulate_database_query(user_prompt)

        with st.expander("📊 Retrieved Data", expanded=False):
            st.dataframe(simulated_df, use_container_width=True)
            st.caption(
                f"Retrieved {len(simulated_df)} rows, {len(simulated_df.columns)} columns"
            )

        # STEP 4: NEW - Generate and display visualization
        if enable_viz and not simulated_df.empty:
            st.markdown("---")
            integrate_visualization_to_existing_flow(user_prompt, simulated_df)
            st.markdown("---")

        # STEP 5: Final LLM analysis (your existing step)
        st.markdown("### 🤖 Analysis Summary")
        # In your real app, this comes from your second LLM agent
        simulated_summary = generate_simulated_summary(user_prompt, simulated_df)
        st.write(simulated_summary)


def simulate_database_query(prompt: str) -> pd.DataFrame:
    """
    Simulates database query results based on the prompt.
    In your real app, this is where your SQL query returns actual data.
    """
    import numpy as np

    prompt_lower = prompt.lower()

    # Temporal trend scenario
    if "trend" in prompt_lower or "over time" in prompt_lower:
        return pd.DataFrame(
            {
                "date": pd.date_range("2024-01-01", periods=12, freq="M"),
                "incident_count": np.random.randint(10, 30, 12),
                "severity_avg": np.random.uniform(1.5, 3.5, 12),
            }
        )

    # Comparison scenario
    elif "compare" in prompt_lower or "by department" in prompt_lower:
        return pd.DataFrame(
            {
                "department": [
                    "Safety",
                    "Operations",
                    "Maintenance",
                    "Quality",
                    "Production",
                ],
                "incident_count": np.random.randint(5, 50, 5),
                "resolved_count": np.random.randint(3, 45, 5),
            }
        )

    # Distribution scenario
    elif "distribution" in prompt_lower or "severity" in prompt_lower:
        return pd.DataFrame(
            {
                "incident_id": range(1, 101),
                "severity_score": np.random.gamma(2, 2, 100),
                "resolution_days": np.random.exponential(5, 100),
            }
        )

    # Default scenario
    else:
        return pd.DataFrame(
            {
                "category": [
                    "Near Miss",
                    "Minor Injury",
                    "Property Damage",
                    "First Aid",
                    "Lost Time",
                ],
                "count": np.random.randint(10, 100, 5),
                "avg_cost": np.random.uniform(100, 5000, 5),
            }
        )


def generate_simulated_summary(prompt: str, df: pd.DataFrame) -> str:
    """
    Simulates the final LLM summary.
    In your real app, this is generated by your second LLM agent.
    """
    return f"""
Based on the data analysis, here are the key findings:

- The dataset contains {len(df)} records across {len(df.columns)} dimensions
- Analysis period covers recent HSE activities
- The visualization above provides a clear view of the trends and patterns
- Notable observations include variations in incident patterns and severity levels

The data suggests opportunities for targeted safety interventions in specific areas.
"""


# Configuration for different deployment scenarios
class StreamlitConfig:
    """
    Configuration helper for integrating into your existing app.
    """

    @staticmethod
    def minimal_integration():
        """
        Minimal code needed to add visualization to your existing app.
        """
        code = """
# Add this import at the top of your existing Streamlit app
from visualization_agent import VisualizationAgent

# Initialize agent (once, at app startup)
viz_agent = VisualizationAgent()

# After your SQL query returns a dataframe, add these lines:
if not df.empty:
    fig, explanation = viz_agent.generate_visualization(user_prompt, df)
    st.plotly_chart(fig, use_container_width=True)
    st.caption(explanation)
"""
        return code

    @staticmethod
    def full_integration():
        """
        Full integration with error handling and configuration.
        """
        code = """
# In your main Streamlit app file:

import streamlit as st
from visualization_agent import VisualizationAgent

# Session state for agent (prevents re-initialization)
if 'viz_agent' not in st.session_state:
    st.session_state.viz_agent = VisualizationAgent()

# Add toggle in sidebar
with st.sidebar:
    enable_smart_viz = st.checkbox("Smart Visualization", value=True)

# After getting dataframe from SQL query:
if enable_smart_viz and not df.empty:
    try:
        fig, explanation = st.session_state.viz_agent.generate_visualization(
            user_prompt=user_query,
            df=df
        )
        
        # Display visualization between data and LLM summary
        st.subheader("📊 Visualization")
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("About this visualization"):
            st.info(explanation)
            
    except Exception as e:
        st.warning(f"Visualization unavailable: {str(e)}")
"""
        return code


def show_integration_guide():
    """
    Display integration instructions in Streamlit.
    """
    st.title("Integration Guide")

    st.markdown("""
    ## How to Integrate into Your Existing RAG App
    
    The visualization agent fits into your current architecture at **Step 3.5** 
    (between dataframe retrieval and final LLM summary).
    """)

    tab1, tab2, tab3 = st.tabs(["Minimal", "Recommended", "Advanced"])

    with tab1:
        st.markdown("### Minimal Integration")
        st.markdown("Add just 3 lines of code to your existing app:")
        st.code(StreamlitConfig.minimal_integration(), language="python")

    with tab2:
        st.markdown("### Recommended Integration")
        st.markdown("Full integration with error handling and user controls:")
        st.code(StreamlitConfig.full_integration(), language="python")

    with tab3:
        st.markdown("### Advanced Configuration")
        st.markdown("""
        For advanced use cases, you can:
        - Customize visualization types based on HSE-specific requirements
        - Add custom preprocessing for domain-specific calculations
        - Implement caching for better performance
        - Add user controls for visualization parameters
        """)

        st.code(
            """
# Custom configuration example
viz_config = {
    'preferred_charts': ['line', 'bar'],  # Limit chart types
    'max_categories': 15,  # Limit bar chart categories
    'enable_caching': True,  # Cache visualizations
    'color_scheme': 'Bayer'  # Custom color scheme
}

agent = VisualizationAgent(config=viz_config)
        """,
            language="python",
        )


if __name__ == "__main__":
    # Uncomment one of these to run:
    main_streamlit_app()  # Full demo app
    # show_integration_guide()  # Integration documentation
