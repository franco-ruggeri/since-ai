# HSE Visualization Agent (Bayer Challenge - Since AI)

AI-powered visualization agent that automatically generates data charts and plots based on user prompts. This toolkit combines LLM-driven agentic analysis with intelligent and extensible plot type selection to create appropriate visualizations for the given user prompt + dataset for the Bayer challenge of the Since AI Hackathon.

## Components

- **Streamlit Web App** (`streamlit_app.py`) — Interactive front-end labeled "HSE Bot - Visualization Agent" for uploading data files (CSV/XLSX) and receiving chart recommendations with explanations.
- **Plot Type Generator** (`plot_type_generator/`) — Multi-agent system that analyzes queries and data to recommend optimal chart types. Includes:
  - `query_planning_agent.py` — Breaks down user requests into analysis steps
  - `numeric_analysis_agent.py` — Analyzes numerical data properties
  - `lexical_analysis_agent.py` — Processes textual queries and metadata
  - `plot_type_chooser_agent.py` — Recommends specific chart types
  - `visual_appropriateness_agent.py` — Validates visualization fitness
  - `llm_provider.py` — Abstract layer supporting Featherless and Google Gemini
- **Chart Factory** (`chart_factory.py`, `charts/`) — Renders recommended visualizations using Plotly:
  - Bar charts, line charts, histograms, pie charts, box plots, heatmaps
  - Pluggable chart registry system
- **Model Orchestrator** (`model_orchestrator/`) — Utilities for LLM selection and agent coordination across providers (orchestrator.py, model_registry.py, agent_types.py).
- **Clustering Module** (`clustering/`) — Semantic clustering via sentence transformers for data grouping and analysis.
- **Recommendations** (`recommendations/`) — Generated JSON outputs from plot recommendations (used for evaluation).

## Quick Start

1. Install dependencies:

```bash
uv sync
```

2. Configure environment variables in `.streamlit/streamlit.toml`:

```toml
LLM_PROVIDER = "featherless"  # or "gemini"
FEATHERLESS_API_KEY = "your_key"  # if using featherless (default)
GOOGLE_API_KEY = "your_key"  # if using gemini
```

3. Run the Streamlit app:

```bash
streamlit run streamlit_app.py
```

The app accepts user prompts in English or Finnish and generates visualizations with preprocessing steps and rationale.

### Alternative: Run the plot-type generator directly

For testing the multi-agent pipeline without the UI:

```bash
python plot_type_generator/main.py
```

## Configuration

- **LLM Provider**: Configure via `streamlit.toml` (default: `featherless`, or use `gemini`)
- **API Keys**: `GOOGLE_API_KEY` for Gemini, `FEATHERLESS_API_KEY` for Featherless
- **Provider Implementation**: `plot_type_generator/llm_provider.py` handles provider abstraction

## Architecture

**Data Flow**: User query + Dataset → Query Planning → Multi-Agent Analysis (Numeric, Lexical, Appropriateness) → Plot Type Selection → Chart Rendering

**Available Chart Types**:
- Bar charts (categorical data)
- Line charts (time series)
- Histograms (distributions)
- Pie charts (proportions)
- Box plots (statistical summaries)
- Heatmaps (2D patterns)

## Key Files

- `streamlit_app.py` — Main UI and pipeline orchestration
- `agent_caller.py` — Entry point connecting agents to UI
- `components.py` — UI components (visualization display, logging)
- `chart_factory.py` — Chart instance creation and routing
- `plot_type_generator/llm_provider.py` — LLM backend abstraction
- `plot_type_generator/plot_gen_state.py` — Pipeline state management
- `model_orchestrator/orchestrator.py` — Agent and model coordination

## Dependencies

Key packages (see `pyproject.toml` for complete list):
- `langchain` / `langchain-core` — LLM integration framework
- `langchain-featherless-ai`, `langchain-google-genai` — Provider adapters
- `streamlit` — Web UI framework
- `plotly` — Interactive chart rendering
- `pandas` — Data processing
- `sentence-transformers` — Semantic clustering
- `python-dotenv` — Environment configuration

## Project Structure

```
since-ai/
├── streamlit_app.py              # Main Streamlit UI entry point
├── agent_caller.py               # Agent orchestration and API
├── components.py                 # Streamlit UI components
├── chart_factory.py              # Chart creation factory
├── styles.css                    # UI styling
├── requirements.txt              # Pip dependencies (legacy)
├── pyproject.toml                # Project metadata and uv dependencies
├── .streamlit/
│   └── streamlit.toml            # Streamlit configuration
│
├── plot_type_generator/          # Multi-agent plot recommendation system
│   ├── main.py                   # Pipeline orchestration and demo
│   ├── llm_provider.py           # LLM provider abstraction (Featherless/Gemini)
│   ├── plot_type_chooser_agent.py # Main recommendation agent
│   ├── query_planning_agent.py   # Query analysis and planning
│   ├── numeric_analysis_agent.py # Numerical data analysis
│   ├── lexical_analysis_agent.py # Text query processing
│   ├── visual_appropriateness_agent.py # Visualization validation
│   ├── plot_gen_state.py         # Pipeline state management
│   ├── utils.py                  # Utility functions
│   └── prompts/                  # Agent prompt templates
│
├── charts/                       # Chart rendering implementations
│   ├── base_chart.py             # Base chart class
│   ├── bar_chart.py
│   ├── line_chart.py
│   ├── histogram_chart.py
│   ├── pie_chart.py
│   ├── box_plot_chart.py
│   ├── heatmap_chart.py
│   ├── chart_registry.py         # Chart type registry
│   └── __init__.py
│
├── model_orchestrator/           # LLM and agent orchestration utilities
│   ├── orchestrator.py           # Main orchestrator
│   ├── model_registry.py         # LLM model registry
│   ├── agent_types.py            # Agent type definitions
│   ├── config.py                 # Configuration
│   ├── prompt_analyzer.py        # Prompt analysis
│   ├── integration.py            # Provider integration
│   ├── example_usage.py          # Usage examples
│   ├── ARCHITECTURE.md           # Detailed architecture doc
│   └── README.md                 # Module documentation
│
├── clustering/                   # Semantic clustering module
│   ├── main.py                   # Clustering pipeline
│   └── semantic_clustering.py    # Semantic clustering implementation
│
├── data/                         # Sample datasets
│   ├── data.json
│   ├── data_english.json
│   └── *.csv                     # Various CSV datasets
│
├── recommendations/              # Generated plot recommendations
│   └── *.json                    # Recommendation outputs
│
├── tests/                        # Test suite
│   ├── test_multi_agent_pipeline.py
│   ├── test_single_data_case.py
│   ├── test_data_json_full.py
│   ├── test_text_extraction.py
│   ├── run_plot_chooser.py
│   ├── run_query_agent.py
│   └── run_parse_unparsed_reco.py
│
└── README.md
```

