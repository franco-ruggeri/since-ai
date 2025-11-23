import streamlit as st
import logging
import json


from plot_type_generator.plot_gen_state import PlotGenState
from plot_type_generator.utils import _load_prompt, extract_json_content
from plot_type_generator.llm_provider import get_llm_provider

logger = logging.getLogger(__name__)


def lexical_analysis_agent(state: PlotGenState) -> PlotGenState:
    """Validates the lexical correctness and clarity of plot recommendations.

    This agent analyzes:
    - Feature name accuracy and existence in data
    - Clarity and specificity of rationales
    - Consistency of terminology across recommendations
    - Language quality (English-only, no mixed languages)
    - Specificity of filtering/aggregation instructions
    - Completeness of explanations

    Populates state['lexical_feedback'] with validation results and suggestions.
    """
    lexical_template = _load_prompt("lexical_analysis.txt")

    plot_recommendations = state.get("plot_recommendations") or ""
    data_table = state.get("data_table") or {}
    user_query = state.get("user_query") or ""
    execution_plan = state.get("execution_plan") or ""

    # Parse the plot recommendations to analyze
    try:
        recommendations_json = extract_json_content(plot_recommendations)
    except Exception:
        logger.warning("Could not parse plot recommendations as JSON")
        recommendations_json = plot_recommendations

    user_content = (
        f"User Query: {user_query}\n\n"
        f"Execution Plan: {execution_plan}\n\n"
        f"Data Table Structure:\n{data_table}\n\n"
        f"Plot Recommendations to Validate:\n{json.dumps(recommendations_json, indent=2)}\n\n"
        "Analyze the lexical quality and correctness of these recommendations."
    )

    messages = [
        ("system", lexical_template),
        ("human", user_content),
    ]

    # Get LLM provider
    try:
        provider = get_llm_provider()
    except Exception:
        logger.exception("Failed to initialize LLM provider")
        raise

    # Get model from state or environment
    model = (
        state.get("llm_model")
        or st.secrets["LEXICAL_ANALYSIS_AGENT_LLM_MODEL"]
        or st.secrets["PLOT_TYPE_CHOOSER_AGENT_LLM_MODEL"]
    )

    # Invoke the provider
    try:
        text = provider.invoke(messages, model=model, temperature=0, seed=42)
    except Exception as e:
        logger.exception("Lexical analysis agent failed: %s", e)
        raise

    lexical_feedback_str = text if isinstance(text, str) else str(text)
    state["lexical_feedback"] = lexical_feedback_str

    return state
