# 📊 HSE Visualization Agent
#### 🏆 Bayer Challenge - Since AI Hackathon 2025

# Team Members

| Name | Email Address | LinkedIn | Available for Hire |
|------|---------------|----------|----------|
| Somoy Barua | somoytunu@gmail.com | [LinkedIn - Somoy](https://www.linkedin.com/in/somoy) | Yes
| Abhishek Roy | theabhishekroy77@gmail.com | [LinkedIn - Abhishek](https://www.linkedin.com/in/abhishekroy1709) | Yes
| Franco Ruggeri | franco.ruggeri.pro@gmail.com | [LinkedIn - Franco](https://www.linkedin.com/in/ruggeri-franco) | No
| Kevin Dall Torres | kevin.dalla.torre@hotmail.com | [LinkedIn - Kevin ](https://www.linkedin.com/in/kevin-dalla-torre-153764252) | No


> **Intelligent visualization recommendations powered by multi-agent LLM analysis**

AI-powered visualization agent that automatically generates data charts and plots based on user prompts and dataframes. A collaborative Multi-Agent System that uses AI consensus and Model Orchestration to select the best, most cost-effective model and verify the plot idea before generating any visualization for safety reports.

---

## 🎯 Components

- **🎨 Streamlit Web App** (`streamlit_app.py`)  
  Interactive front-end labeled "HSE Bot - Visualization Agent" for uploading data files (CSV/XLSX) and receiving chart recommendations with explanations.

- **🤖 Plot Type Generator** (`plot_type_generator/`)  
  Multi-agent system that analyzes queries and data to recommend optimal chart types:
  - `query_planning_agent.py` — Breaks down user requests into analysis steps
  - `numeric_analysis_agent.py` — Analyzes numerical data properties
  - `lexical_analysis_agent.py` — Processes textual queries and metadata
  - `plot_type_chooser_agent.py` — Recommends specific chart types
  - `visual_appropriateness_agent.py` — Validates visualization fitness
  - `llm_provider.py` — Abstract layer supporting Featherless and Google Gemini

- **📈 Chart Factory** (`chart_factory.py`, `charts/`)  
  Renders recommended visualizations using Plotly with support for:
  - 📊 Bar charts • 📉 Line charts • 📶 Histograms • 🥧 Pie charts • 📦 Box plots • 🔥 Heatmaps
  - Pluggable chart registry system

- **⚙️ Model Orchestrator** (`model_orchestrator/`)  
  Utilities for LLM selection and agent coordination across providers

- **🔗 Clustering Module** (`clustering/`)  
  Semantic clustering via sentence transformers for data grouping and analysis

- **💾 Recommendations** (`recommendations/`)  
  Generated JSON outputs from plot recommendations for evaluation

---

## 🚀 Quick Start

### 0️⃣ Setup Dev Container (Recommended)

This project is containerized with a **dev container** for consistent development environments. 

**Option A: VS Code** (Recommended)
- Install the [Remote - Containers extension](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-containers)
- Open the project folder in VS Code
- Click "Reopen in Container" when prompted

**Option B: Manual Setup**
- Install Docker and Docker Compose
- Continue with steps below

### 1️⃣ Install dependencies

```bash
uv sync
```

### 2️⃣ Configure environment

Create `.streamlit/streamlit.toml` and add:

```toml
LLM_PROVIDER = "featherless"  # or "gemini"
FEATHERLESS_API_KEY = "your_key"  # if using featherless (default)
GOOGLE_API_KEY = "your_key"  # if using gemini
```

### 3️⃣ Run the app

```bash
streamlit run streamlit_app.py
```

The app accepts user prompts in **English or Finnish** 🇬🇧 🇫🇮 and generates visualizations with preprocessing steps and rationale.

### 🔬 Alternative: Test the pipeline directly

```bash
python plot_type_generator/main.py
```

---

## ⚙️ Configuration

| Setting | Details |
|---------|---------|
| **LLM Provider** | Configure via `streamlit.toml` (default: `featherless`, or use `gemini`) |
| **API Keys** | `GOOGLE_API_KEY` for Gemini, `FEATHERLESS_API_KEY` for Featherless |
| **Implementation** | `plot_type_generator/llm_provider.py` handles provider abstraction |

---

## 🏗️ Architecture

```
User Query + Dataset
        ↓
   Query Planning
        ↓
Multi-Agent Analysis (Numeric, Lexical, Appropriateness)
        ↓
  Plot Type Selection
        ↓
  Chart Rendering
```

### 📊 Supported Chart Types

| Chart Type | Use Case |
|-----------|----------|
| 📊 **Bar Charts** | Categorical data comparison |
| 📉 **Line Charts** | Time series & trends |
| 📶 **Histograms** | Distribution analysis |
| 🥧 **Pie Charts** | Proportion visualization |
| 📦 **Box Plots** | Statistical summaries |
| 🔥 **Heatmaps** | 2D pattern detection |

---

## 📁 Key Files

| File | Purpose |
|------|---------|
| `streamlit_app.py` | Main UI and pipeline orchestration |
| `agent_caller.py` | Entry point connecting agents to UI |
| `components.py` | UI components (visualization, logging) |
| `chart_factory.py` | Chart creation and routing |
| `plot_type_generator/llm_provider.py` | LLM backend abstraction |
| `plot_type_generator/plot_gen_state.py` | Pipeline state management |
| `model_orchestrator/orchestrator.py` | Agent coordination |

---

## 📦 Dependencies

Key packages (see `pyproject.toml` for complete list):

| Package | Purpose |
|---------|---------|
| `langchain` / `langchain-core` | LLM integration framework |
| `langchain-featherless-ai` | Featherless AI provider integration |
| `langchain-google-genai` | Google Gemini provider integration |
| `streamlit` | Web UI framework |
| `plotly` | Interactive chart rendering |
| `pandas` | Data processing |
| `sentence-transformers` | Semantic clustering |

---

## 📂 Project Structure

```
since-ai/
├── streamlit_app.py              # Main Streamlit UI entry point
├── agent_caller.py               # Agent orchestration and API
├── components.py                 # Streamlit UI components
├── chart_factory.py              # Chart creation factory
├── styles.css                    # UI styling
├── requirements.txt              # Pip dependencies (legacy)
├── pyproject.toml                # Project metadata and uv dependencies
│
├── plot_type_generator/          # Multi-agent plot recommendation system
│   ├── main.py                   # Pipeline orchestration and demo
│   ├── llm_provider.py           # LLM provider abstraction
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
│   ├── base_chart.py
│   ├── bar_chart.py
│   ├── line_chart.py
│   ├── histogram_chart.py
│   ├── pie_chart.py
│   ├── box_plot_chart.py
│   ├── heatmap_chart.py
│   ├── chart_registry.py
│   └── __init__.py
│
├── model_orchestrator/           # LLM and agent orchestration utilities
│   ├── orchestrator.py
│   ├── model_registry.py
│   ├── agent_types.py
│   ├── config.py
│   ├── prompt_analyzer.py
│   ├── integration.py
│   ├── example_usage.py
│   ├── ARCHITECTURE.md
│   └── README.md
│
├── clustering/                   # Semantic clustering module
│   ├── main.py
│   └── semantic_clustering.py
│
├── data/                         # Sample datasets
│   ├── data.json
│   ├── data_english.json
│   └── *.csv
│
├── recommendations/              # Generated plot recommendations
│   └── *.json
│
├── tests/                        # Test suite
│   └── *.py
│
└── README.md
```

---

**Developed for the Bayer Challenge - Since AI Hackathon 2025**

