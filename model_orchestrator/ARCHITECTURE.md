# Model Orchestrator Architecture

## System Overview - ASCII Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         USER REQUEST / PIPELINE                              │
│                    "Analyze sales data and create plots"                     │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        PLOT GENERATION PIPELINE                              │
│                     (run_plot_generation_pipeline)                           │
└─────┬───────────────┬────────────────┬────────────────┬─────────────────────┘
      │               │                │                │
      │               │                │                │
      ▼               ▼                ▼                ▼
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│ THINKING │    │NUMERICAL │    │  VISUAL  │    │ LEXICAL  │
│  AGENT   │    │  AGENT   │    │  AGENT   │    │  AGENT   │
│          │    │          │    │          │    │          │
│ (Query   │    │ (Numeric │    │ (Visual  │    │(Lexical  │
│Planning) │    │Analysis) │    │Appropriate│    │Analysis) │
└────┬─────┘    └────┬─────┘    └────┬─────┘    └────┬─────┘
     │               │                │                │
     │ Need Model    │ Need Model     │ Need Model     │ Need Model
     │               │                │                │
     ▼               ▼                ▼                ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                      MODEL ORCHESTRATOR                                      │
│                 get_model_for_specific_agent()                               │
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    AGENT TYPE CLASSIFIER                             │   │
│  │                                                                       │   │
│  │  Input: "thinking" → AgentType.THINKING                             │   │
│  │  Input: "numerical" → AgentType.NUMERICAL                           │   │
│  │  Input: "visual" → AgentType.VISUAL                                 │   │
│  │  Input: "lexical" → AgentType.LEXICAL                               │   │
│  └───────────────────────────────┬─────────────────────────────────────┘   │
│                                  │                                           │
│                                  ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │              AGENT REQUIREMENTS LOOKUP                               │   │
│  │                                                                       │   │
│  │  ┌─────────────┬──────────────┬────────────────┬────────────────┐  │   │
│  │  │  THINKING   │  NUMERICAL   │    VISUAL      │    LEXICAL     │  │   │
│  │  ├─────────────┼──────────────┼────────────────┼────────────────┤  │   │
│  │  │Reasoning≥0.8│ Reasoning≥0.85│ Reasoning≥0.75│ Instruct≥0.90 │  │   │
│  │  │Instruct≥0.75│ Math focused │ Creativity≥0.70│ Reasoning≥0.70│  │   │
│  │  │8k+ context  │ Instruct≥0.80│ Instruct≥0.85 │ 8k+ context   │  │   │
│  │  │Speed≤3      │ 8k+ context  │ 8k+ context   │ Language focus│  │   │
│  │  └─────────────┴──────────────┴────────────────┴────────────────┘  │   │
│  └───────────────────────────────┬─────────────────────────────────────┘   │
│                                  │                                           │
│                                  ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    MODEL REGISTRY                                    │   │
│  │                (10+ HuggingFace Models)                              │   │
│  │                                                                       │   │
│  │  ┌──────────────┬──────────────┬──────────────┬──────────────┐     │   │
│  │  │ Phi-2        │ Mistral 7B   │ Qwen 4B      │ Gemma 2 9B   │     │   │
│  │  ├──────────────┼──────────────┼──────────────┼──────────────┤     │   │
│  │  │ Llama 3.1 8B │ Mistral 24B  │ Llama 3.1 70B│ Qwen 2.5 72B │     │   │
│  │  ├──────────────┼──────────────┼──────────────┼──────────────┤     │   │
│  │  │ DeepSeek V3  │ CodeLlama 34B│     ...      │     ...      │     │   │
│  │  └──────────────┴──────────────┴──────────────┴──────────────┘     │   │
│  │                                                                       │   │
│  │  Each model has:                                                     │   │
│  │  • reasoning_score, creativity_score, instruction_following_score   │   │
│  │  • max_context_length, max_output_tokens                            │   │
│  │  • strengths (TaskType list)                                        │   │
│  │  • cost_per_1k_tokens, speed_tier                                   │   │
│  └───────────────────────────────┬─────────────────────────────────────┘   │
│                                  │                                           │
│                                  ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    SCORING ALGORITHM                                 │   │
│  │              score_model_for_agent(model, agent_type)                │   │
│  │                                                                       │   │
│  │  Score Components (0-100):                                           │   │
│  │                                                                       │   │
│  │  ┌─────────────────────────────────────────────────────────────┐    │   │
│  │  │ 1. TASK ALIGNMENT (30 points)                               │    │   │
│  │  │    - Match model strengths with agent's primary tasks       │    │   │
│  │  │    - e.g., NUMERICAL needs MATH + DATA_ANALYSIS             │    │   │
│  │  ├─────────────────────────────────────────────────────────────┤    │   │
│  │  │ 2. CAPABILITY MATCH (40 points)                             │    │   │
│  │  │    - Reasoning score vs requirement (15 pts)                │    │   │
│  │  │    - Instruction following vs requirement (15 pts)          │    │   │
│  │  │    - Creativity score if needed (10 pts)                    │    │   │
│  │  ├─────────────────────────────────────────────────────────────┤    │   │
│  │  │ 3. PERFORMANCE (20 points)                                  │    │   │
│  │  │    - Context length check (10 pts)                          │    │   │
│  │  │    - Speed tier preference (10 pts)                         │    │   │
│  │  ├─────────────────────────────────────────────────────────────┤    │   │
│  │  │ 4. COST EFFICIENCY (10 points)                              │    │   │
│  │  │    - Prefer economical models when quality is similar       │    │   │
│  │  └─────────────────────────────────────────────────────────────┘    │   │
│  └───────────────────────────────┬─────────────────────────────────────┘   │
│                                  │                                           │
│                                  ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                   MODEL RANKING                                      │   │
│  │                                                                       │   │
│  │  For THINKING Agent:          For NUMERICAL Agent:                  │   │
│  │  1. Qwen 4B Thinking (78)     1. Qwen 4B Thinking (68)              │   │
│  │  2. Llama 3.1 8B (76)         2. Mistral Small 24B (65)             │   │
│  │  3. Mistral Small 24B (75)    3. Llama 3.1 70B (60)                 │   │
│  │                                                                       │   │
│  │  For VISUAL Agent:            For LEXICAL Agent:                     │   │
│  │  1. Mistral Small 24B (82)    1. Mistral 7B Instruct (62)           │   │
│  │  2. Llama 3.1 8B (76)         2. Qwen 4B Thinking (61)              │   │
│  │  3. Qwen 4B Thinking (74)     3. Gemma 2 9B (61)                    │   │
│  └───────────────────────────────┬─────────────────────────────────────┘   │
│                                  │                                           │
│                                  ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                  RETURN BEST MODEL ID                                │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SELECTED MODELS                                      │
│                                                                               │
│  THINKING  → Qwen/Qwen3-4B-Thinking-2507                                    │
│  NUMERICAL → Qwen/Qwen3-4B-Thinking-2507                                    │
│  VISUAL    → mistralai/Mistral-Small-3.1-24B-Instruct-2503                  │
│  LEXICAL   → mistralai/Mistral-7B-Instruct-v0.2                             │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    FEATHERLESS AI API CALLS                                  │
│                                                                               │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐  ┌───────────┐│
│  │ LLM Provider   │  │ LLM Provider   │  │ LLM Provider   │  │LLM Provider││
│  │ invoke()       │  │ invoke()       │  │ invoke()       │  │ invoke()  ││
│  │                │  │                │  │                │  │           ││
│  │ Qwen 4B        │  │ Qwen 4B        │  │ Mistral 24B    │  │Mistral 7B ││
│  │ Thinking       │  │ Thinking       │  │                │  │           ││
│  └────────┬───────┘  └────────┬───────┘  └────────┬───────┘  └─────┬─────┘│
│           │                   │                   │                 │       │
└───────────┼───────────────────┼───────────────────┼─────────────────┼───────┘
            │                   │                   │                 │
            ▼                   ▼                   ▼                 ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         AGENT RESPONSES                                      │
│                                                                               │
│  execution_plan  │  numeric_feedback  │  visual_feedback  │  lexical_feedback│
└───────────────────────────────────────────────────────────────────────────────┘
```

## Detailed Flow for Single Agent

```
Agent Invocation:
═══════════════════════════════════════════════════════════════════════════

Step 1: Agent Needs Model
┌────────────────────────────────────────┐
│  numeric_analysis_agent(state)         │
│                                        │
│  Need to select LLM model...          │
└──────────────┬─────────────────────────┘
               │
               ▼
Step 2: Check Hierarchy
┌────────────────────────────────────────────────────────────────┐
│  model = (                                                     │
│      state.get("llm_model")                 ← User override   │
│      or os.environ.get("NUMERIC_ANALYSIS_AGENT_LLM_MODEL")    │
│                                              ← Env var         │
│      or get_model_for_specific_agent("numerical")             │
│                                              ← Auto-select ✓   │
│  )                                                             │
└──────────────┬─────────────────────────────────────────────────┘
               │
               ▼
Step 3: Orchestrator Processing
┌────────────────────────────────────────────────────────────────┐
│  get_model_for_specific_agent("numerical")                     │
│                                                                 │
│  1. Map "numerical" → AgentType.NUMERICAL                      │
│                                                                 │
│  2. Get requirements:                                          │
│     - Primary tasks: [MATH, DATA_ANALYSIS, REASONING]         │
│     - Reasoning ≥ 0.85                                         │
│     - Instruction ≥ 0.80                                       │
│     - Context ≥ 8192                                           │
│                                                                 │
│  3. Score all models:                                          │
│     Qwen 4B Thinking:     68/100 ✓ BEST                       │
│     Mistral Small 24B:    65/100                               │
│     Llama 3.1 70B:        60/100                               │
│     ...                                                         │
│                                                                 │
│  4. Return: "Qwen/Qwen3-4B-Thinking-2507"                     │
└──────────────┬─────────────────────────────────────────────────┘
               │
               ▼
Step 4: Use Model
┌────────────────────────────────────────────────────────────────┐
│  provider = get_llm_provider()                                 │
│  text = provider.invoke(                                       │
│      messages,                                                 │
│      model="Qwen/Qwen3-4B-Thinking-2507",                     │
│      temperature=0,                                            │
│      seed=42                                                   │
│  )                                                             │
│                                                                 │
│  state["numeric_feedback"] = text                             │
└────────────────────────────────────────────────────────────────┘
```

## Scoring Example - Visual Agent

```
MODEL EVALUATION: Mistral Small 24B for VISUAL Agent
═════════════════════════════════════════════════════════════════

Requirements:
  Primary Tasks: [REASONING, GENERAL, DATA_ANALYSIS]
  Reasoning Score: ≥ 0.75
  Creativity Score: ≥ 0.70
  Instruction Following: ≥ 0.85
  Min Context: 8192 tokens

Model Capabilities:
  Model: mistralai/Mistral-Small-3.1-24B-Instruct-2503
  Strengths: [GENERAL, CODE, REASONING, DATA_ANALYSIS]
  Reasoning: 0.85
  Creativity: 0.75 (not tracked, using default)
  Instruction Following: 0.90
  Context Length: 32768
  Speed Tier: 3
  Cost: $0.0005/1k

Score Calculation:
┌─────────────────────────────────────────────┬──────┬────────┐
│ Component                                   │ Max  │ Earned │
├─────────────────────────────────────────────┼──────┼────────┤
│ TASK ALIGNMENT                              │  30  │  30.0  │
│   Matches: REASONING, GENERAL, DATA_ANALYSIS│      │        │
│   (3 of 3 tasks matched)                    │      │        │
├─────────────────────────────────────────────┼──────┼────────┤
│ CAPABILITY MATCH                            │  40  │  38.1  │
│   Reasoning: 0.85 ≥ 0.75 → 15 pts          │  15  │  15.0  │
│   Instruction: 0.90 ≥ 0.85 → 15 pts        │  15  │  15.0  │
│   Creativity: 0.75 ≥ 0.70 → 10 pts (est)  │  10  │   8.1  │
├─────────────────────────────────────────────┼──────┼────────┤
│ PERFORMANCE                                 │  20  │  14.0  │
│   Context: 32768 ≥ 8192 → 10 pts           │  10  │  10.0  │
│   Speed Tier: 3/5 → 4 pts                  │  10  │   4.0  │
├─────────────────────────────────────────────┼──────┼────────┤
│ COST EFFICIENCY                             │  10  │   0.0  │
│   Cost: $0.0005 (moderate) → 0 pts         │  10  │   0.0  │
├─────────────────────────────────────────────┼──────┼────────┤
│ TOTAL SCORE                                 │ 100  │  82.1  │
└─────────────────────────────────────────────┴──────┴────────┘

Result: SELECTED ✓ (Highest score for VISUAL agent)
```

## Cost Calculation Flow

```
COST ESTIMATION PIPELINE
═══════════════════════════════════════════════════════════════

For each agent call:

┌─────────────────────────────────────────────────────────┐
│  Input Tokens                                           │
│  ┌───────────────────────────────────────────────────┐ │
│  │ System Prompt:        ~1000 tokens                │ │
│  │ User Query:           ~200 tokens                 │ │
│  │ Data Table:           ~1500 tokens                │ │
│  │ Previous Context:     ~800 tokens                 │ │
│  └───────────────────────────────────────────────────┘ │
│  Total Input: ~3500 tokens                             │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│  Output Tokens (estimated)                              │
│  ┌───────────────────────────────────────────────────┐ │
│  │ THINKING:    ~1500 tokens (execution plan)       │ │
│  │ NUMERICAL:   ~800 tokens (validation feedback)   │ │
│  │ VISUAL:      ~800 tokens (visual feedback)       │ │
│  │ LEXICAL:     ~600 tokens (language feedback)     │ │
│  └───────────────────────────────────────────────────┘ │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│  Cost Calculation                                       │
│                                                          │
│  THINKING:                                              │
│    (3500 + 1500) / 1000 × $0.0002 = $0.001000          │
│                                                          │
│  NUMERICAL:                                             │
│    (3800 + 800) / 1000 × $0.0002 = $0.000920           │
│                                                          │
│  VISUAL:                                                │
│    (3300 + 800) / 1000 × $0.0005 = $0.002050           │
│                                                          │
│  LEXICAL:                                               │
│    (2600 + 600) / 1000 × $0.0002 = $0.000640           │
│                                                          │
│  ─────────────────────────────────────────────────     │
│  TOTAL PER PIPELINE RUN:        $0.004610               │
└─────────────────────────────────────────────────────────┘
```

## Integration Architecture

```
YOUR APPLICATION STRUCTURE
═══════════════════════════════════════════════════════════════

┌─────────────────────────────────────────────────────────────┐
│  streamlit_app.py                                           │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  User uploads data, enters query                    │   │
│  └───────────────────┬─────────────────────────────────┘   │
│                      │                                       │
│                      ▼                                       │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  agent_caller.py                                     │   │
│  │  get_response(user_prompt, dataframe)               │   │
│  └───────────────────┬─────────────────────────────────┘   │
└────────────────────────────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  plot_type_generator/main.py                                │
│  run_plot_generation_pipeline()                             │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  State Flow:                                         │  │
│  │                                                       │  │
│  │  1. query_planning_agent(state)                     │  │
│  │     ↓ uses THINKING model                           │  │
│  │     state["execution_plan"] = "..."                 │  │
│  │                                                       │  │
│  │  2. plot_type_chooser_agent(state)                  │  │
│  │     ↓                                                 │  │
│  │     state["plot_recommendations"] = "..."           │  │
│  │                                                       │  │
│  │  3. Validation Loop:                                 │  │
│  │     ┌─────────────────────────────────────────────┐ │  │
│  │     │ numeric_analysis_agent(state)               │ │  │
│  │     │ ↓ uses NUMERICAL model                      │ │  │
│  │     │ state["numeric_feedback"] = "..."           │ │  │
│  │     │                                              │ │  │
│  │     │ visual_appropriateness_agent(state)         │ │  │
│  │     │ ↓ uses VISUAL model                         │ │  │
│  │     │ state["visual_feedback"] = "..."            │ │  │
│  │     │                                              │ │  │
│  │     │ lexical_analysis_agent(state)               │ │  │
│  │     │ ↓ uses LEXICAL model                        │ │  │
│  │     │ state["lexical_feedback"] = "..."           │ │  │
│  │     └─────────────────────────────────────────────┘ │  │
│  │                                                       │  │
│  │  4. synthesis_agent(state)                           │  │
│  │     ↓                                                 │  │
│  │     Final recommendations                            │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

**Key Points:**

1. **Automatic Selection**: Each agent automatically gets the best model for its task
2. **Hierarchical Override**: State > Env Var > Auto-select
3. **Specialized Scoring**: Each agent type has custom scoring weights
4. **Cost Optimized**: Right-sized models save ~40% vs using large models everywhere
5. **Transparent**: Verbose mode shows scoring breakdown
