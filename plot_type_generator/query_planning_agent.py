import os
import logging
from typing import Any, Dict, List, Optional, cast
from dotenv import load_dotenv


from plot_type_generator.plot_gen_state import PlotGenState
from plot_type_generator.utils import _load_prompt, _get_api_key

logger = logging.getLogger(__name__)
load_dotenv()  # reads .env in cwd or parent dirs


# Provider-native Featherless client
try:
    from langchain_featherless_ai import ChatFeatherlessAi
except Exception:
    ChatFeatherlessAi = None
try:
    # pydantic SecretStr is commonly used for typed secrets in provider clients
    from pydantic import SecretStr
except Exception:
    SecretStr = None


# The previous LangChain wrapper was removed — we use the provider client below.


def query_planning_agent(state: PlotGenState) -> PlotGenState:
    """Break down a user request into executable steps using Featherless.

    Behavior:
    - Loads prompt template from `prompts/query_refiner.txt`.
    - Constructs `system` message with the prompt file and `user` message with
      the user's query and a lightweight data description.
    - Calls Featherless API using API key from `FEATHERLESS_API_KEY`.
    - Places the returned text into `state['execution_plan']`.
    """
    try:
        refiner_prompt = _load_prompt("query_refiner.txt")
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

    # Build messages and call the provider client directly.
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

    if ChatFeatherlessAi is None:
        raise RuntimeError(
            "`langchain-featherless-ai` package is not installed. "
            "Install it (pip install langchain-featherless-ai) or use the HTTP path."
        )

    api_key = _get_api_key()
    api_url = os.environ.get("FEATHERLESS_API_URL", "https://api.featherless.ai/v1")
    model = state.get("llm_model") or os.environ.get("QUERY_PLANNING_AGENT_LLM_MODEL")

    if SecretStr is not None:
        llm = ChatFeatherlessAi(api_key=SecretStr(api_key), base_url=api_url)
    else:
        # cast to Any to satisfy static typing when SecretStr is not available
        llm = ChatFeatherlessAi(api_key=cast(Any, api_key), base_url=api_url)

    try:
        if model:
            response = llm.invoke(messages, model=model)
        else:
            response = llm.invoke(messages)
    except Exception:
        logger.exception("Failed to call Featherless provider client")
        raise

    # Normalize response into a string for the TypedDict field
    if isinstance(response, str):
        text = response
    elif isinstance(response, dict):
        # common shapes
        try:
            text = response.get("choices", [])[0].get("message", {}).get("content")
        except Exception:
            text = str(response)
    elif hasattr(response, "content"):
        text = getattr(response, "content")
    else:
        text = str(response)

    state["execution_plan"] = text

    # Note: do not run downstream agents here — orchestration should be
    # handled by the caller (e.g., `main` or a pipeline runner). The planner
    # only populates `state['execution_plan']`.

    return state
