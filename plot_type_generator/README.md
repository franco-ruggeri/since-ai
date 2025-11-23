**Plot Type Generator**

A compact toolkit that analyzes user prompt and dataframe and chooses/constructs an appropriate plot type using small agent modules and configurable LLM backends.

**Purpose:**
- **Summary:** Drive automated plot-type recommendations and short planning steps for chart generation from textual queries or data summaries.
- **Scope:** Supports multiple LLM backends, modular agents (lexical/numeric analysis, query planning, plot-type chooser) and a light state object for generation.

**Quick Usage:**
- **Run demo:** `python plot_type_generator/main.py`
- **Environment:** set `LLM_PROVIDER` to `featherless` or `gemini` and provide matching API key env var (`FEATHERLESS_API_KEY` or `GOOGLE_API_KEY`).
- **Python API (example):**
  ```py
  from plot_type_generator.llm_provider import get_llm_provider

  provider = get_llm_provider('featherless', api_key='YOUR_KEY')
  messages = [("system","You are a helpful assistant"),("human","Suggest a plot for sales by month")]
  text = provider.invoke(messages, model=None, temperature=0.7)
  print(text)
  ```

**Key Modules & References:**
- **`llm_provider.py`**: Factory and provider wrappers. Implements `FeatherlessProvider` and `GeminiProvider` and `get_llm_provider(...)`.
- **`plot_type_chooser_agent.py`**: Central agent that uses analysis agents and the LLM provider to propose chart types and configurations.
- **`query_planning_agent.py`**: Translates high-level requests into stepwise queries and planning text for downstream agents.
- **`numeric_analysis_agent.py` / `lexical_analysis_agent.py`**: Helpers that analyze dataset columns or textual queries to detect numeric vs categorical cues.
- **`plot_gen_state.py`**: Small state container used by agents to accumulate decisions and intermediate artifacts.
- **`main.py`**: Example runner that wires agents together and prints recommendations.

**Notes & Tips:**
- The provider interface exposes `invoke(messages, model=None, temperature=0.7, seed=42, **kwargs)`. Providers may ignore some kwargs (e.g. `seed`) depending on backend support.
- If you add providers, update `get_llm_provider` in `llm_provider.py` to register them.
- Tests and examples live under `plot_type_generator/` and the top-level `tests/` folder — run `python -m pytest tests` if you have test dependencies installed.

**Contact / Contributing:**
- Read the source files referenced above for implementation details. Open issues or PRs for new provider integrations or agent improvements.
