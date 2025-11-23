import os
import streamlit as st
import logging
from dotenv import load_dotenv


from plot_type_generator.plot_gen_state import PlotGenState
from plot_type_generator.utils import _load_prompt, extract_json_content
from plot_type_generator.llm_provider import get_llm_provider

logger = logging.getLogger(__name__)
load_dotenv()


def plot_type_chooser_agent(state: PlotGenState, k: int = 3) -> PlotGenState:
    """Given a state with `execution_plan` and `data_table`, ask the LLM to
    recommend the top-k plot types and relevant features. Puts the JSON
    string under `plot_recommendations` in the returned state.
    """
    chooser_template = _load_prompt("plot_type_chooser.txt")

    execution_plan = state.get("execution_plan") or state.get("user_query") or ""
    data_table = state.get("data_table") or {}

    # Use the execution plan (produced by the query planning agent) as the
    # system prompt so the chooser is grounded in the planner's output.
    # The chooser template provides the format and additional instructions
    # and will be sent as the human message.
    user_content = (
        f"Data Description:\n{data_table}\n\n"
        f"Return the top {k} plot type suggestions as requested."
    )

    # system prompt: execution_plan (preferred). human message: chooser template + data/user content
    human_message = chooser_template + "\n\n" + user_content

    messages = [
        ("system", execution_plan),
        ("human", human_message),
    ]

    # Get LLM provider
    try:
        provider = get_llm_provider()
    except Exception:
        logger.exception("Failed to initialize LLM provider")
        raise

    # Get model from state or environment
    model = state.get("llm_model") or st.secrets["PLOT_TYPE_CHOOSER_AGENT_LLM_MODEL"]

    # Invoke the provider
    try:
        text = provider.invoke(messages, model=model, temperature=0.7, seed=42)
    except Exception as e:
        logger.exception("LLM provider invoke failed: %s", e)
        raise

    plot_recommendations_str = text if isinstance(text, str) else str(text)
    state["plot_recommendations"] = plot_recommendations_str

    # Extract processed_data if present in the response
    try:
        recommendations_json = extract_json_content(plot_recommendations_str)
        if "processed_data" in recommendations_json:
            state["processed_data"] = recommendations_json["processed_data"]
            logger.info("Extracted processed_data from plot recommendations")
    except Exception as e:
        logger.warning(f"Could not extract processed_data: {e}")

    return state
