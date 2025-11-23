"""
Model Orchestrator for Featherless AI
Intelligently selects the best model based on prompt characteristics and token requirements.

Includes specialized support for agent-specific model selection:
- Thinking Agent (Query Planning)
- Numerical Agent (Numeric Analysis)
- Visual Agent (Visual Appropriateness)
- Lexical Agent (Lexical Analysis)
"""

from .orchestrator import ModelOrchestrator
from .model_registry import ModelRegistry
from .prompt_analyzer import PromptAnalyzer

try:
    from .agent_types import (
        AgentType,
        get_agent_requirements,
        get_model_recommendations_for_agent,
        score_model_for_agent
    )
    __all__ = [
        "ModelOrchestrator",
        "ModelRegistry",
        "PromptAnalyzer",
        "AgentType",
        "get_agent_requirements",
        "get_model_recommendations_for_agent",
        "score_model_for_agent"
    ]
except ImportError:
    __all__ = ["ModelOrchestrator", "ModelRegistry", "PromptAnalyzer"]
