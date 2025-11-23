import os
import logging
import json
from dotenv import load_dotenv


from plot_type_generator.plot_gen_state import PlotGenState
from plot_type_generator.utils import _load_prompt, extract_json_content
from plot_type_generator.llm_provider import get_llm_provider

logger = logging.getLogger(__name__)
load_dotenv()


def visual_appropriateness_agent(state: PlotGenState) -> PlotGenState:
    """Validates the visual appropriateness and best practices of plot recommendations.

    This agent analyzes:
    - Whether the plot type is the best choice for the user's intent
    - Adherence to data visualization best practices
    - Appropriateness of alternatives offered
    - User experience and interpretability considerations
    - Plot type diversity and value of alternatives
    - Alignment with user query and execution plan

    Populates state['visual_feedback'] with validation results and suggestions.
    """
    visual_template = _load_prompt("visual_appropriateness.txt")

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
        "Analyze the visual appropriateness and best practices of these recommendations."
    )

    messages = [
        ("system", visual_template),
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
        or os.environ.get("VISUAL_APPROPRIATENESS_AGENT_LLM_MODEL")
        or os.environ.get("PLOT_TYPE_CHOOSER_AGENT_LLM_MODEL")
    )

    # Invoke the provider
    try:
        text = provider.invoke(messages, model=model, temperature=0, seed=42)
    except Exception as e:
        logger.exception("Visual appropriateness agent failed: %s", e)
        raise

    visual_feedback_str = text if isinstance(text, str) else str(text)
    state["visual_feedback"] = visual_feedback_str

    return state
