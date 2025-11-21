"""
Semantic Analysis Extension for HSE Text Data

This module extends the visualization agent with semantic analysis capabilities
for handling text-heavy data (incident descriptions, observations, etc.)
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import re
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go


class SemanticTextAnalyzer:
    """
    Analyzes text data to extract meaningful categories for visualization.
    Uses rule-based NLP and keyword extraction (no external ML dependencies).
    """

    def __init__(self):
        # Common HSE-related categories and keywords
        self.hse_taxonomy = {
            "incident_type": {
                "slip_fall": ["slip", "fall", "tripped", "stumble"],
                "equipment": ["equipment", "machinery", "tool", "device"],
                "chemical": ["chemical", "substance", "spill", "exposure"],
                "electrical": ["electrical", "shock", "electrocution", "power"],
                "fire": ["fire", "burn", "flame", "ignition"],
                "ergonomic": ["ergonomic", "strain", "repetitive", "posture"],
                "struck_by": ["struck", "hit", "collision", "impact"],
                "caught_in": ["caught", "trapped", "pinched", "crushed"],
            },
            "severity": {
                "minor": ["minor", "small", "slight", "negligible"],
                "moderate": ["moderate", "medium", "noticeable"],
                "serious": ["serious", "major", "significant", "severe"],
                "critical": ["critical", "life-threatening", "emergency", "severe"],
            },
            "location": {
                "production_floor": [
                    "production",
                    "manufacturing",
                    "assembly",
                    "shop floor",
                ],
                "warehouse": ["warehouse", "storage", "inventory"],
                "office": ["office", "desk", "administrative"],
                "laboratory": ["lab", "laboratory", "testing"],
                "outdoor": ["outdoor", "outside", "yard", "parking"],
            },
            "body_part": {
                "hand": ["hand", "finger", "wrist", "palm"],
                "foot": ["foot", "toe", "ankle"],
                "back": ["back", "spine", "lumbar"],
                "head": ["head", "skull", "face"],
                "eye": ["eye", "vision", "sight"],
            },
        }

    def analyze_text_column(
        self, text_series: pd.Series, category: str = "auto"
    ) -> pd.DataFrame:
        """
        Analyze a text column and extract categorical information.

        Args:
            text_series: Pandas series containing text data
            category: Type of categorization ('auto', 'incident_type', 'severity', etc.)

        Returns:
            DataFrame with original text and extracted categories
        """
        if category == "auto":
            # Auto-detect best category
            category = self._detect_best_category(text_series)

        results = []
        for text in text_series:
            if pd.isna(text):
                results.append("Unknown")
            else:
                extracted_category = self._extract_category(str(text).lower(), category)
                results.append(extracted_category)

        return pd.DataFrame({"original_text": text_series, "category": results})

    def _detect_best_category(self, text_series: pd.Series) -> str:
        """
        Auto-detect which taxonomy category best fits the text data.
        """
        # Count keyword matches for each taxonomy
        category_scores = {}

        sample_text = " ".join(
            text_series.dropna().astype(str).head(50).tolist()
        ).lower()

        for taxonomy_name, categories in self.hse_taxonomy.items():
            score = 0
            for category, keywords in categories.items():
                for keyword in keywords:
                    score += sample_text.count(keyword)
            category_scores[taxonomy_name] = score

        # Return taxonomy with highest score
        best_category = max(category_scores, key=category_scores.get)
        return best_category if category_scores[best_category] > 0 else "incident_type"

    def _extract_category(self, text: str, taxonomy_type: str) -> str:
        """
        Extract category from text based on keyword matching.
        """
        if taxonomy_type not in self.hse_taxonomy:
            return "Uncategorized"

        categories = self.hse_taxonomy[taxonomy_type]

        # Score each category based on keyword matches
        scores = {}
        for category, keywords in categories.items():
            score = sum(1 for keyword in keywords if keyword in text)
            if score > 0:
                scores[category] = score

        if not scores:
            return "Other"

        # Return category with highest score
        best_category = max(scores, key=scores.get)
        return best_category.replace("_", " ").title()

    def extract_key_terms(self, text_series: pd.Series, top_n: int = 10) -> List[tuple]:
        """
        Extract most common key terms from text data.
        """
        # Combine all text
        all_text = " ".join(text_series.dropna().astype(str).tolist()).lower()

        # Remove common stop words
        stop_words = {
            "the",
            "a",
            "an",
            "and",
            "or",
            "but",
            "in",
            "on",
            "at",
            "to",
            "for",
            "of",
            "with",
            "by",
            "from",
            "was",
            "were",
            "been",
            "be",
            "have",
            "has",
            "had",
            "is",
            "are",
            "this",
            "that",
        }

        # Extract words
        words = re.findall(r"\b\w+\b", all_text)
        filtered_words = [w for w in words if w not in stop_words and len(w) > 3]

        # Count and return top N
        word_counts = Counter(filtered_words)
        return word_counts.most_common(top_n)

    def create_text_based_visualization(
        self, df: pd.DataFrame, text_column: str, viz_type: str = "category_bar"
    ) -> go.Figure:
        """
        Create visualization specifically designed for text data.

        Args:
            df: DataFrame containing text data
            text_column: Name of column containing text
            viz_type: Type of visualization ('category_bar', 'word_cloud', 'trend')
        """
        if text_column not in df.columns:
            raise ValueError(f"Column {text_column} not found in dataframe")

        text_series = df[text_column]

        if viz_type == "category_bar":
            # Categorize and create bar chart
            analyzed = self.analyze_text_column(text_series)
            category_counts = analyzed["category"].value_counts()

            fig = px.bar(
                x=category_counts.index,
                y=category_counts.values,
                title=f"Categorized {text_column}",
                labels={"x": "Category", "y": "Count"},
                template="plotly_white",
            )

        elif viz_type == "top_terms":
            # Extract and visualize top terms
            top_terms = self.extract_key_terms(text_series, top_n=15)

            fig = px.bar(
                x=[term for term, _ in top_terms],
                y=[count for _, count in top_terms],
                title=f"Top Terms in {text_column}",
                labels={"x": "Term", "y": "Frequency"},
                template="plotly_white",
            )
            fig.update_layout(xaxis_tickangle=-45)

        else:
            raise ValueError(f"Unknown viz_type: {viz_type}")

        return fig


class EnhancedVisualizationAgent:
    """
    Enhanced version of VisualizationAgent with semantic text analysis.
    Extends the base agent with capabilities to handle text-heavy HSE data.
    """

    def __init__(self):
        self.text_analyzer = SemanticTextAnalyzer()
        # Import base agent if available
        try:
            from visualization_agent import VisualizationAgent

            self.base_agent = VisualizationAgent()
        except:
            self.base_agent = None

    def generate_visualization(self, user_prompt: str, df: pd.DataFrame) -> tuple:
        """
        Enhanced visualization generation with text analysis support.
        """
        # Detect if we have text-heavy data
        text_columns = df.select_dtypes(include=["object"]).columns.tolist()
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()

        # If mostly text data with few numeric columns, use semantic analysis
        if len(text_columns) > 0 and len(numeric_columns) <= 1:
            return self._generate_text_visualization(user_prompt, df, text_columns)

        # Otherwise, use base agent if available
        elif self.base_agent:
            return self.base_agent.generate_visualization(user_prompt, df)

        # Fallback to basic visualization
        else:
            return self._generate_basic_visualization(df)

    def _generate_text_visualization(
        self, prompt: str, df: pd.DataFrame, text_cols: List[str]
    ) -> tuple:
        """
        Generate visualization focused on text analysis.
        """
        # Select the most relevant text column
        main_text_col = self._select_best_text_column(text_cols, prompt)

        # Determine visualization type based on prompt
        prompt_lower = prompt.lower()

        if any(kw in prompt_lower for kw in ["category", "type", "classify", "group"]):
            fig = self.text_analyzer.create_text_based_visualization(
                df, main_text_col, viz_type="category_bar"
            )
            explanation = (
                f"Categorized {main_text_col} using semantic analysis. "
                f"Identified {df[main_text_col].nunique()} unique categories."
            )

        elif any(kw in prompt_lower for kw in ["word", "term", "common", "frequent"]):
            fig = self.text_analyzer.create_text_based_visualization(
                df, main_text_col, viz_type="top_terms"
            )
            explanation = (
                f"Most frequent terms found in {main_text_col}. "
                f"Analyzed {len(df)} records."
            )

        else:
            # Default: categorize
            fig = self.text_analyzer.create_text_based_visualization(
                df, main_text_col, viz_type="category_bar"
            )
            explanation = f"Automatic categorization of {main_text_col} based on content analysis."

        return fig, explanation

    def _select_best_text_column(self, text_cols: List[str], prompt: str) -> str:
        """
        Select the most relevant text column based on the prompt.
        """
        prompt_lower = prompt.lower()

        # Look for column names in the prompt
        for col in text_cols:
            if col.lower() in prompt_lower:
                return col

        # Prefer columns with certain keywords
        priority_keywords = [
            "description",
            "note",
            "comment",
            "observation",
            "incident",
        ]
        for keyword in priority_keywords:
            for col in text_cols:
                if keyword in col.lower():
                    return col

        # Default to first text column
        return text_cols[0]

    def _generate_basic_visualization(self, df: pd.DataFrame) -> tuple:
        """
        Fallback basic visualization.
        """
        # Simple bar chart of first column value counts
        first_col = df.columns[0]
        value_counts = df[first_col].value_counts().head(10)

        fig = px.bar(
            x=value_counts.index,
            y=value_counts.values,
            title=f"Distribution of {first_col}",
            template="plotly_white",
        )

        explanation = f"Basic distribution of {first_col} (top 10 values shown)."
        return fig, explanation


def demo_semantic_analysis():
    """
    Demonstration of semantic analysis capabilities on HSE text data.
    """
    import streamlit as st

    st.title("🔍 Semantic Text Analysis Demo")
    st.markdown("Demonstration of text analysis for HSE incident data")

    # Create sample HSE text data
    sample_incidents = [
        "Employee slipped on wet floor in production area, minor injury to ankle",
        "Chemical spill in laboratory, contained quickly, no injuries",
        "Equipment malfunction caused production stoppage, no injuries",
        "Worker experienced back strain from lifting heavy materials",
        "Near miss: forklift nearly struck pedestrian in warehouse",
        "Electrical shock incident at workstation, employee received first aid",
        "Fire alarm malfunction caused evacuation, false alarm",
        "Ergonomic issue reported: repetitive strain from computer work",
        "Fall from ladder in maintenance area, serious injury",
        "Chemical exposure in lab, employee sent for medical evaluation",
    ] * 5  # Repeat for more data

    df = pd.DataFrame(
        {
            "incident_id": range(1, len(sample_incidents) + 1),
            "description": sample_incidents,
            "date": pd.date_range(
                "2024-01-01", periods=len(sample_incidents), freq="3D"
            ),
        }
    )

    # Show sample data
    with st.expander("📋 Sample Incident Data", expanded=True):
        st.dataframe(df.head(10), use_container_width=True)

    # Initialize analyzer
    agent = EnhancedVisualizationAgent()

    # Different analysis scenarios
    st.subheader("Analysis Options")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Categorize Incident Types", use_container_width=True):
            prompt = "Categorize these incidents by type"
            fig, explanation = agent.generate_visualization(prompt, df)

            st.plotly_chart(fig, use_container_width=True)
            st.info(explanation)

    with col2:
        if st.button("Find Common Terms", use_container_width=True):
            prompt = "What are the most common terms in incident descriptions?"
            fig, explanation = agent.generate_visualization(prompt, df)

            st.plotly_chart(fig, use_container_width=True)
            st.info(explanation)

    # Show detailed analysis
    st.subheader("Detailed Semantic Analysis")

    analyzer = SemanticTextAnalyzer()
    analyzed = analyzer.analyze_text_column(df["description"])

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("**Categorized Incidents:**")
        result_df = pd.DataFrame(
            {"Category": analyzed["category"], "Description": df["description"]}
        )
        st.dataframe(result_df, use_container_width=True, height=300)

    with col2:
        st.markdown("**Category Distribution:**")
        category_counts = analyzed["category"].value_counts()
        st.bar_chart(category_counts)

    # Key terms analysis
    st.subheader("Key Terms Frequency")
    top_terms = analyzer.extract_key_terms(df["description"], top_n=20)

    terms_df = pd.DataFrame(top_terms, columns=["Term", "Frequency"])

    col1, col2 = st.columns([1, 2])
    with col1:
        st.dataframe(terms_df, use_container_width=True)
    with col2:
        fig = px.bar(
            terms_df,
            x="Term",
            y="Frequency",
            title="Most Common Terms in Incident Descriptions",
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    demo_semantic_analysis()
