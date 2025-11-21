"""
HSE Data Visualization Agent
A smart visualization system that automatically generates appropriate charts
based on user prompts and dataframes.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, Optional, Tuple
import json
import re
from datetime import datetime
import numpy as np


class VisualizationAgent:
    """
    Main agent that orchestrates the visualization generation process.
    """

    def __init__(self, llm_api_key: Optional[str] = None):
        """
        Initialize the visualization agent.

        Args:
            llm_api_key: Optional API key for LLM service (for production use)
        """
        self.llm_api_key = llm_api_key

    def generate_visualization(
        self, user_prompt: str, df: pd.DataFrame
    ) -> Tuple[Any, str]:
        """
        Main entry point: generates visualization from prompt and dataframe.

        Args:
            user_prompt: The user's natural language query
            df: The dataframe returned from the database

        Returns:
            Tuple of (plotly figure object, explanation text)
        """
        # Step 1: Analyze context
        context_analysis = self._analyze_context(user_prompt, df)

        # Step 2: Select visualization strategy
        viz_config = self._select_visualization(context_analysis)

        # Step 3: Preprocess data
        processed_df = self._preprocess_data(df, viz_config)

        # Step 4: Generate chart
        fig = self._generate_chart(processed_df, viz_config)

        # Step 5: Create explanation
        explanation = self._create_explanation(viz_config, processed_df)

        return fig, explanation

    def _analyze_context(self, prompt: str, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze the user prompt and dataframe to understand intent and data structure.
        """
        analysis = {
            "prompt": prompt,
            "prompt_lower": prompt.lower(),
            "df_shape": df.shape,
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.to_dict(),
            "numeric_columns": df.select_dtypes(include=[np.number]).columns.tolist(),
            "datetime_columns": self._detect_datetime_columns(df),
            "text_columns": df.select_dtypes(include=["object"]).columns.tolist(),
            "has_temporal_data": False,
            "intent": None,
            "sample_data": df.head(5).to_dict("records") if len(df) > 0 else [],
        }

        # Detect temporal intent
        temporal_keywords = [
            "over time",
            "trend",
            "timeline",
            "historical",
            "monthly",
            "yearly",
            "daily",
            "weekly",
            "period",
            "evolution",
        ]
        analysis["has_temporal_data"] = any(
            kw in analysis["prompt_lower"] for kw in temporal_keywords
        )

        # Detect intent type
        if any(
            kw in analysis["prompt_lower"]
            for kw in ["compare", "comparison", "versus", "vs", "between"]
        ):
            analysis["intent"] = "comparison"
        elif any(
            kw in analysis["prompt_lower"]
            for kw in ["distribution", "spread", "range", "histogram"]
        ):
            analysis["intent"] = "distribution"
        elif any(
            kw in analysis["prompt_lower"] for kw in ["trend", "over time", "timeline"]
        ):
            analysis["intent"] = "trend"
        elif any(
            kw in analysis["prompt_lower"]
            for kw in ["correlation", "relationship", "related", "scatter"]
        ):
            analysis["intent"] = "correlation"
        elif any(
            kw in analysis["prompt_lower"]
            for kw in ["total", "sum", "count", "how many"]
        ):
            analysis["intent"] = "aggregation"
        else:
            analysis["intent"] = "exploration"

        return analysis

    def _detect_datetime_columns(self, df: pd.DataFrame) -> list:
        """Detect columns that contain datetime data."""
        datetime_cols = []
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                datetime_cols.append(col)
            elif df[col].dtype == "object":
                # Try to parse as datetime
                try:
                    sample = df[col].dropna().head(10)
                    if len(sample) > 0:
                        pd.to_datetime(sample)
                        datetime_cols.append(col)
                except:
                    pass
        return datetime_cols

    def _select_visualization(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Select the most appropriate visualization type and configuration.
        """
        viz_config = {
            "chart_type": None,
            "x_axis": None,
            "y_axis": None,
            "color_by": None,
            "title": "",
            "aggregation": None,
            "preprocessing_steps": [],
        }

        df_shape = analysis["df_shape"]
        intent = analysis["intent"]
        numeric_cols = analysis["numeric_columns"]
        datetime_cols = analysis["datetime_columns"]
        text_cols = analysis["text_columns"]

        # Decision tree for visualization selection

        # 1. Temporal data with trend intent
        if (intent == "trend" or analysis["has_temporal_data"]) and datetime_cols:
            viz_config["chart_type"] = "line"
            viz_config["x_axis"] = datetime_cols[0]
            viz_config["y_axis"] = numeric_cols[0] if numeric_cols else None
            viz_config["title"] = f"Trend Over Time"
            viz_config["preprocessing_steps"].append("sort_by_date")

        # 2. Distribution analysis
        elif intent == "distribution" and numeric_cols:
            if df_shape[0] > 30:
                viz_config["chart_type"] = "histogram"
                viz_config["x_axis"] = numeric_cols[0]
                viz_config["title"] = f"Distribution of {numeric_cols[0]}"
            else:
                viz_config["chart_type"] = "box"
                viz_config["y_axis"] = numeric_cols[0]
                viz_config["title"] = f"Distribution of {numeric_cols[0]}"

        # 3. Correlation/Relationship
        elif intent == "correlation" and len(numeric_cols) >= 2:
            viz_config["chart_type"] = "scatter"
            viz_config["x_axis"] = numeric_cols[0]
            viz_config["y_axis"] = numeric_cols[1]
            viz_config["title"] = f"{numeric_cols[1]} vs {numeric_cols[0]}"

        # 4. Comparison with categories
        elif intent in ["comparison", "aggregation"]:
            if text_cols and numeric_cols:
                viz_config["chart_type"] = "bar"
                viz_config["x_axis"] = text_cols[0]
                viz_config["y_axis"] = numeric_cols[0]
                viz_config["aggregation"] = "sum"
                viz_config["title"] = f"{numeric_cols[0]} by {text_cols[0]}"
                viz_config["preprocessing_steps"].append("aggregate")
            elif len(numeric_cols) >= 2:
                viz_config["chart_type"] = "bar"
                viz_config["x_axis"] = analysis["columns"][0]
                viz_config["y_axis"] = numeric_cols[0]
                viz_config["title"] = f"Comparison of {numeric_cols[0]}"

        # 5. Default: smart fallback based on data structure
        else:
            if datetime_cols and numeric_cols:
                viz_config["chart_type"] = "line"
                viz_config["x_axis"] = datetime_cols[0]
                viz_config["y_axis"] = numeric_cols[0]
                viz_config["title"] = f"{numeric_cols[0]} Over Time"
            elif text_cols and numeric_cols:
                viz_config["chart_type"] = "bar"
                viz_config["x_axis"] = text_cols[0]
                viz_config["y_axis"] = numeric_cols[0]
                viz_config["aggregation"] = "sum"
                viz_config["title"] = f"{numeric_cols[0]} by {text_cols[0]}"
                viz_config["preprocessing_steps"].append("aggregate")
            elif len(numeric_cols) >= 2:
                viz_config["chart_type"] = "scatter"
                viz_config["x_axis"] = numeric_cols[0]
                viz_config["y_axis"] = numeric_cols[1]
                viz_config["title"] = f"{numeric_cols[1]} vs {numeric_cols[0]}"
            elif len(numeric_cols) == 1:
                viz_config["chart_type"] = "histogram"
                viz_config["x_axis"] = numeric_cols[0]
                viz_config["title"] = f"Distribution of {numeric_cols[0]}"
            else:
                # Last resort: value counts of first column
                viz_config["chart_type"] = "bar"
                viz_config["x_axis"] = analysis["columns"][0]
                viz_config["y_axis"] = "count"
                viz_config["title"] = f"Count by {analysis['columns'][0]}"
                viz_config["preprocessing_steps"].append("value_counts")

        # Add color dimension if there are multiple categorical columns
        if len(text_cols) > 1 and viz_config["chart_type"] in [
            "scatter",
            "line",
            "bar",
        ]:
            # Use second categorical column for coloring
            potential_color = [
                col for col in text_cols if col != viz_config.get("x_axis")
            ]
            if potential_color:
                viz_config["color_by"] = potential_color[0]

        return viz_config

    def _preprocess_data(
        self, df: pd.DataFrame, viz_config: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Preprocess the dataframe according to the visualization requirements.
        """
        processed_df = df.copy()

        for step in viz_config.get("preprocessing_steps", []):
            if step == "sort_by_date" and viz_config["x_axis"]:
                # Convert to datetime and sort
                try:
                    processed_df[viz_config["x_axis"]] = pd.to_datetime(
                        processed_df[viz_config["x_axis"]]
                    )
                    processed_df = processed_df.sort_values(viz_config["x_axis"])
                except:
                    pass

            elif step == "aggregate" and viz_config["x_axis"] and viz_config["y_axis"]:
                # Group by x_axis and aggregate y_axis
                agg_func = viz_config.get("aggregation", "sum")
                processed_df = processed_df.groupby(
                    viz_config["x_axis"], as_index=False
                )[viz_config["y_axis"]].agg(agg_func)

            elif step == "value_counts" and viz_config["x_axis"]:
                # Create value counts
                counts = processed_df[viz_config["x_axis"]].value_counts().reset_index()
                counts.columns = [viz_config["x_axis"], "count"]
                processed_df = counts

        # Limit categories if too many (for readability)
        if viz_config["chart_type"] == "bar" and viz_config["x_axis"]:
            if processed_df[viz_config["x_axis"]].nunique() > 20:
                # Keep top 20 by y-axis value
                if viz_config["y_axis"] in processed_df.columns:
                    processed_df = processed_df.nlargest(20, viz_config["y_axis"])

        return processed_df

    def _generate_chart(
        self, df: pd.DataFrame, viz_config: Dict[str, Any]
    ) -> go.Figure:
        """
        Generate the actual Plotly chart.
        """
        chart_type = viz_config["chart_type"]

        # Common parameters
        common_params = {"title": viz_config["title"], "template": "plotly_white"}

        try:
            if chart_type == "line":
                fig = px.line(
                    df,
                    x=viz_config["x_axis"],
                    y=viz_config["y_axis"],
                    color=viz_config.get("color_by"),
                    **common_params,
                )
                fig.update_traces(mode="lines+markers")

            elif chart_type == "bar":
                fig = px.bar(
                    df,
                    x=viz_config["x_axis"],
                    y=viz_config["y_axis"],
                    color=viz_config.get("color_by"),
                    **common_params,
                )

            elif chart_type == "scatter":
                fig = px.scatter(
                    df,
                    x=viz_config["x_axis"],
                    y=viz_config["y_axis"],
                    color=viz_config.get("color_by"),
                    **common_params,
                )

            elif chart_type == "histogram":
                fig = px.histogram(df, x=viz_config["x_axis"], **common_params)

            elif chart_type == "box":
                fig = px.box(df, y=viz_config["y_axis"], **common_params)

            else:
                # Fallback to simple bar chart
                fig = px.bar(df, **common_params)

            # Update layout for better appearance
            fig.update_layout(
                height=500,
                hovermode="closest",
                showlegend=True if viz_config.get("color_by") else False,
            )

            return fig

        except Exception as e:
            # Create error visualization
            fig = go.Figure()
            fig.add_annotation(
                text=f"Unable to create visualization: {str(e)}",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                showarrow=False,
                font=dict(size=14, color="red"),
            )
            return fig

    def _create_explanation(self, viz_config: Dict[str, Any], df: pd.DataFrame) -> str:
        """
        Create a human-readable explanation of the visualization.
        """
        chart_type = viz_config["chart_type"]

        explanations = {
            "line": f"Line chart showing the trend of {viz_config['y_axis']} over {viz_config['x_axis']}.",
            "bar": f"Bar chart comparing {viz_config['y_axis']} across different {viz_config['x_axis']} categories.",
            "scatter": f"Scatter plot exploring the relationship between {viz_config['x_axis']} and {viz_config['y_axis']}.",
            "histogram": f"Histogram showing the distribution of {viz_config['x_axis']} values.",
            "box": f"Box plot displaying the distribution and outliers of {viz_config['y_axis']}.",
        }

        base_explanation = explanations.get(chart_type, "Visualization of the data.")

        if viz_config.get("color_by"):
            base_explanation += f" Color-coded by {viz_config['color_by']}."

        base_explanation += f" Based on {len(df)} data points."

        return base_explanation


# Example usage and testing
def test_visualization_agent():
    """
    Test the visualization agent with various scenarios.
    """
    agent = VisualizationAgent()

    # Test Case 1: Temporal trend data
    print("Test 1: Temporal Trend Data")
    df1 = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=12, freq="M"),
            "incidents": np.random.randint(5, 25, 12),
            "department": ["Safety"] * 6 + ["Operations"] * 6,
        }
    )
    prompt1 = "Show me the trend of incidents over time"
    fig1, explanation1 = agent.generate_visualization(prompt1, df1)
    print(f"Chart Type: {explanation1}\n")

    # Test Case 2: Comparison data
    print("Test 2: Comparison Data")
    df2 = pd.DataFrame(
        {
            "department": ["Safety", "Operations", "Maintenance", "Quality", "HR"],
            "incident_count": [45, 32, 28, 15, 8],
            "severity_avg": [2.3, 3.1, 2.8, 1.5, 1.2],
        }
    )
    prompt2 = "Compare incident counts by department"
    fig2, explanation2 = agent.generate_visualization(prompt2, df2)
    print(f"Chart Type: {explanation2}\n")

    # Test Case 3: Distribution data
    print("Test 3: Distribution Data")
    df3 = pd.DataFrame(
        {
            "severity_score": np.random.gamma(2, 2, 100),
            "resolution_days": np.random.exponential(5, 100),
        }
    )
    prompt3 = "What is the distribution of severity scores?"
    fig3, explanation3 = agent.generate_visualization(prompt3, df3)
    print(f"Chart Type: {explanation3}\n")

    # Test Case 4: Correlation data
    print("Test 4: Correlation Data")
    df4 = pd.DataFrame(
        {
            "training_hours": np.random.uniform(0, 100, 50),
            "incident_rate": 50
            - np.random.uniform(0, 100, 50) * 0.3
            + np.random.normal(0, 5, 50),
        }
    )
    prompt4 = "Is there a relationship between training hours and incident rates?"
    fig4, explanation4 = agent.generate_visualization(prompt4, df4)
    print(f"Chart Type: {explanation4}\n")

    print("All tests completed!")
    return [fig1, fig2, fig3, fig4]


if __name__ == "__main__":
    # Run tests
    figures = test_visualization_agent()
