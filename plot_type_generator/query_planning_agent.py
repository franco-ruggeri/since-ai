import os
import logging
from dotenv import load_dotenv
import streamlit as st


from plot_type_generator.plot_gen_state import PlotGenState
from plot_type_generator.utils import _load_prompt
from plot_type_generator.llm_provider import get_llm_provider

logger = logging.getLogger(__name__)
load_dotenv()


def query_planning_agent(state: PlotGenState) -> PlotGenState:
    """Break down a user request into executable steps using configured LLM provider.

    Behavior:
    - Loads prompt template from `prompts/query_refiner_text_extraction.txt`.
    - Constructs `system` message with the prompt file and `user` message with
      the user's query and a lightweight data description.
    - Calls LLM provider (Featherless or Gemini) based on configuration.
    - Places the returned text into `state['execution_plan']`.
    """
    try:
        refiner_prompt = _load_prompt("query_refiner_text_extraction.txt")
    except FileNotFoundError:
        logger.exception("Prompt file not found")
        raise

    user_query = state.get("user_query") or ""
    data_table = state.get("data_table") or {}

    data_summary = data_table
    try:
        if isinstance(data_table, dict):
            data_summary = data_table
        else:
            data_summary = repr(data_table)
    except Exception:
        data_summary = str(data_table)

    # Build messages
    system_prompt = refiner_prompt
    user_content = (
        f"User Query: {user_query}\n\n"
        f"Data Table Structure/Summary: {data_summary}\n\n"
        "Generate a detailed execution plan as specified."
    )

    messages = [
        ("system", system_prompt),
        ("human", user_content),
    ]

    # Get model from state or environment
    model = state.get("llm_model") or st.secrets("QUERY_PLANNING_AGENT_LLM_MODEL")

    # Get LLM provider (defaults to environment LLM_PROVIDER or "featherless")
    try:
        provider = get_llm_provider()
    except Exception:
        logger.exception("Failed to initialize LLM provider")
        raise

    # Invoke the provider
    try:
        text = provider.invoke(messages, model=model, temperature=0.7, seed=42)
    except Exception:
        logger.exception("Failed to call LLM provider")
        raise

    state["execution_plan"] = text

    # Note: do not run downstream agents here — orchestration should be
    # handled by the caller (e.g., `main` or a pipeline runner). The planner
    # only populates `state['execution_plan']`.

    return state
