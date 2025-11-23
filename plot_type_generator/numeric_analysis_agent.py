import os
import logging
import json
from dotenv import load_dotenv


from plot_type_generator.plot_gen_state import PlotGenState
from plot_type_generator.utils import _load_prompt, extract_json_content
from plot_type_generator.llm_provider import get_llm_provider

from model_orchestrator.integration import get_model_for_specific_agent

logger = logging.getLogger(__name__)
load_dotenv()


def numeric_analysis_agent(state: PlotGenState) -> PlotGenState:
    """Validates the numeric appropriateness of plot type recommendations.

    This agent analyzes:
    - Data types compatibility with suggested plot types
    - Whether features are numeric, categorical, temporal, etc.
    - Statistical appropriateness of plot type for the data distribution
    - Dimensionality considerations (e.g., when PCA is needed)
    - Whether aggregations/transformations are needed

    Populates state['numeric_feedback'] with validation results and suggestions.
    If issues are found, provides specific corrections.
    """
    numeric_template = _load_prompt("numeric_analysis.txt")

    plot_recommendations = state.get("plot_recommendations") or ""
    data_table = state.get("data_table") or {}
    user_query = state.get("user_query") or ""

    # Parse the plot recommendations to analyze
    try:
        recommendations_json = extract_json_content(plot_recommendations)
    except Exception:
        logger.warning("Could not parse plot recommendations as JSON")
        recommendations_json = plot_recommendations

    user_content = (
        f"User Query: {user_query}\n\n"
        f"Data Table Structure:\n{data_table}\n\n"
        f"Plot Recommendations to Validate:\n{json.dumps(recommendations_json, indent=2)}\n\n"
        "Analyze the numeric appropriateness of these recommendations."
    )

    messages = [
        ("system", numeric_template),
        ("human", user_content),
    ]

    # Get LLM provider
    try:
        provider = get_llm_provider()
    except Exception:
        logger.exception("Failed to initialize LLM provider")
        raise

    model = (
        state.get("llm_model")
        or os.environ.get("NUMERIC_ANALYSIS_AGENT_LLM_MODEL")
        or os.environ.get("PLOT_TYPE_CHOOSER_AGENT_LLM_MODEL")
        or get_model_for_specific_agent("numeric_analysis")
    )
    # Invoke the provider
    try:
        text = provider.invoke(messages, model=model, temperature=0, seed=42)
    except Exception as e:
        logger.exception("Numeric analysis agent failed: %s", e)
        raise

    numeric_feedback_str = text if isinstance(text, str) else str(text)
    state["numeric_feedback"] = numeric_feedback_str

    return state
