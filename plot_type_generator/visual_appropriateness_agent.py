import os
import logging
import json
from typing import Any, Dict, Optional, cast
from dotenv import load_dotenv


from plot_type_generator.plot_gen_state import PlotGenState
from plot_type_generator.utils import (
    _load_prompt,
    _get_api_key,
    extract_json_content,
)

logger = logging.getLogger(__name__)
load_dotenv()

try:
    from langchain_featherless_ai import ChatFeatherlessAi
except Exception:
    ChatFeatherlessAi = None

try:
    from pydantic import SecretStr
except Exception:
    SecretStr = None


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

    if ChatFeatherlessAi is None:
        raise RuntimeError("langchain-featherless-ai is not installed")

    api_key = _get_api_key()
    api_url = os.environ.get("FEATHERLESS_API_URL", "https://api.featherless.ai/v1")

    if SecretStr is not None:
        client = ChatFeatherlessAi(api_key=SecretStr(api_key), base_url=api_url)
    else:
        client = ChatFeatherlessAi(api_key=cast(Any, api_key), base_url=api_url)

    try:
        model = state.get("llm_model") or os.environ.get(
            "VISUAL_APPROPRIATENESS_AGENT_LLM_MODEL"
        ) or os.environ.get("PLOT_TYPE_CHOOSER_AGENT_LLM_MODEL")

        if model:
            response = client.invoke(messages, model=model, temperature=0, seed=42)
        else:
            response = client.invoke(messages, temperature=0, seed=42)
    except Exception as e:
        logger.exception("Visual appropriateness agent failed: %s", e)
        raise

    # Normalize response
    if isinstance(response, str):
        text = response
    elif isinstance(response, dict):
        try:
            text = response.get("choices", [])[0].get("message", {}).get("content")
        except Exception:
            text = str(response)
    elif hasattr(response, "content"):
        text = response.content
    else:
        text = str(response)

    visual_feedback_str = text if isinstance(text, str) else str(text)
    state["visual_feedback"] = visual_feedback_str

    return state
