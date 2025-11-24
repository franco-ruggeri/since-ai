# Since AI

Concise toolkit for automated plot recommendations and lightweight model orchestration, given user prompt and a dataframe (single excel, csv, json file)

This repository contains several cooperating components:

- Streamlit web app: `streamlit_app.py` — interactive front-end for uploading data and getting visual recommendations.
- Plot type generator: `plot_type_generator/` — modular agents and LLM provider adapters that analyze queries/data and propose chart types and generation plans. Key files: `main.py`, `llm_provider.py`, `plot_type_chooser_agent.py`, `query_planning_agent.py`, `plot_gen_state.py`.
- Recommendations: `recommendations/` — JSON outputs of generated plot recommendations and examples used for testing and evaluation.
- Model orchestrator & demos: `model_orchestrator/` — utilities and demo scripts for selecting and orchestrating LLMs and agents across providers. Key files: `orchestrator.py`, `model_registry.py`, `agent_types.py`, `example_usage.py`.

Quick start
-----------

1. Install dependencies:

```bash
pip install uv
uv sync
```

2. Run Streamlit app (local):

```bash
streamlit run streamlit_app.py
```

3. Run plot-type generator demo:

```bash
python plot_type_generator/main.py
```

Configuration & environment
---------------------------
- Set `LLM_PROVIDER` to `featherless` or `gemini` (default: `featherless`).
- Provide matching API key via env var: `FEATHERLESS_API_KEY` or `GOOGLE_API_KEY`.
- The provider factory is implemented in `plot_type_generator/llm_provider.py` — add new providers there.

Code pointers
-------------
- `streamlit_app.py`: Streamlit UI and wiring to recommendation functions.
- `plot_type_generator/main.py`: Example runner that wires agents and prints recommendations.
- `plot_type_generator/llm_provider.py`: Abstraction over LLM backends (Featherless, Gemini).
- `plot_type_generator/plot_type_chooser_agent.py`: Main agent that decides which plot(s) to recommend.
- `plot_type_generator/query_planning_agent.py`: Transforms requests into stepwise LLM prompts/queries.
- `recommendations/`: Store and inspect generated recommendation JSON files.


Repository structure
--------------------

```
since-ai/
├─ streamlit_app.py
├─ README.md
├─ requirements.txt
├─ pyproject.toml
├─ styles.css
├─ charts/
│  ├─ bar_chart.py
│  ├─ line_chart.py
│  └─ ...
├─ plot_type_generator/
│  ├─ main.py
│  ├─ llm_provider.py
│  ├─ plot_type_chooser_agent.py
│  ├─ query_planning_agent.py
│  ├─ plot_gen_state.py
│  └─ ...
├─ recommendations/
│  └─ *.json
├─ model_orchestrator/
│  ├─ ARCHITECTURE.md
│  ├─ README.md
│  ├─ orchestrator.py
│  ├─ model_registry.py
│  ├─ agent_types.py
│  ├─ prompt_analyzer.py
│  ├─ example_usage.py
│  ├─ integration.py
│  └─ config.py
├─ tests/
│  └─ *.py
├─ data/
│  └─ data.json
└─ clustering/
	└─ semantic_clustering.py
```

(Use `tree -L 2` in the repo root for a live view.)

