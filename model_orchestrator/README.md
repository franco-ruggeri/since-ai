# Model Orchestrator for Featherless AI

An intelligent model selection system that automatically chooses the optimal model from Featherless AI's catalog of thousands of HuggingFace models based on prompt characteristics, token requirements, and performance priorities.

## Features

- **Intelligent Prompt Analysis**: Automatically detects task types, complexity, and requirements from user prompts
- **Smart Model Selection**: Selects optimal models based on:
  - Task type (code, math, reasoning, creative, etc.)
  - Token requirements (input/output)
  - Performance priorities (speed vs cost vs quality)
  - Special capabilities (JSON mode, function calling)
- **Comprehensive Model Registry**: Pre-configured catalog of popular models with detailed capabilities
- **Cost & Performance Optimization**: Balances cost, speed, and quality based on requirements
- **Extensible**: Easy to add custom models and selection criteria


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