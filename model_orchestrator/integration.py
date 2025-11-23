"""
Integration helpers for using Model Orchestrator with existing codebase.
Provides easy-to-use wrappers for langchain-featherless-ai integration.
"""

import os
from typing import Optional, Any, Dict

try:
    from .orchestrator import ModelOrchestrator
    from .config import ConfigManager, OrchestratorConfig
except ImportError:
    from orchestrator import ModelOrchestrator
    from config import ConfigManager, OrchestratorConfig


class FeatherlessAIIntegration:
    """
    Integration helper for Featherless AI with automatic model selection.

    This class wraps the orchestrator and provides easy methods to get
    LLM instances with automatically selected models.
    """

    def __init__(
        self,
        config: Optional[OrchestratorConfig] = None,
        api_key: Optional[str] = None
    ):
        """
        Initialize the integration.

        Args:
            config: Optional orchestrator configuration
            api_key: Optional Featherless API key (defaults to env var)
        """
        self.config_manager = ConfigManager(config)
        self.orchestrator = ModelOrchestrator(
            default_priority=self.config_manager.get("default_priority", "balanced")
        )
        self.api_key = api_key or self.config_manager.get("featherless_api_key") or os.getenv("FEATHERLESS_API_KEY")

        if not self.api_key:
            raise ValueError("FEATHERLESS_API_KEY not found in environment or config")

    def get_llm(
        self,
        prompt: str,
        priority: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ):
        """
        Get a ChatFeatherlessAI instance with automatically selected model.

        Args:
            prompt: The prompt that will be used (for model selection)
            priority: Optional priority override ("speed", "cost", "quality", "balanced")
            temperature: Temperature for generation
            max_tokens: Maximum tokens to generate
            **kwargs: Additional arguments for ChatFeatherlessAI

        Returns:
            Tuple of (ChatFeatherlessAI instance, ModelSelectionResult)
        """
        try:
            from langchain_featherless_ai import ChatFeatherlessAi
        except ImportError:
            raise ImportError(
                "langchain-featherless-ai is required. Install with: pip install langchain-featherless-ai"
            )

        # Select the best model
        result = self.orchestrator.select_model(
            prompt=prompt,
            priority=priority,
            max_cost_per_1k=self.config_manager.get("max_cost_per_1k_default"),
            min_quality_score=self.config_manager.get("min_quality_score_default")
        )

        # Create LLM instance
        llm = ChatFeatherlessAi(
            model=result.model.model_id,
            api_key=self.api_key,
            temperature=temperature,
            max_tokens=max_tokens or result.model.max_output_tokens,
            **kwargs
        )

        return llm, result

    def get_llm_for_task(
        self,
        task_description: str,
        temperature: float = 0.7,
        **kwargs
    ):
        """
        Get LLM optimized for a specific task description.

        Args:
            task_description: Description of the task
            temperature: Temperature for generation
            **kwargs: Additional arguments

        Returns:
            Tuple of (ChatFeatherlessAI instance, ModelSelectionResult)
        """
        return self.get_llm(
            prompt=task_description,
            priority="quality",
            temperature=temperature,
            **kwargs
        )

    def get_fast_llm(self, prompt: str, **kwargs):
        """Get the fastest suitable LLM for a prompt"""
        return self.get_llm(prompt, priority="speed", **kwargs)

    def get_cheap_llm(self, prompt: str, **kwargs):
        """Get the most cost-effective LLM for a prompt"""
        return self.get_llm(prompt, priority="cost", **kwargs)

    def get_quality_llm(self, prompt: str, **kwargs):
        """Get the highest quality LLM for a prompt"""
        return self.get_llm(prompt, priority="quality", **kwargs)

    def invoke(
        self,
        prompt: str,
        priority: Optional[str] = None,
        temperature: float = 0.7,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Invoke model selection and generation in one call.

        Args:
            prompt: The input prompt
            priority: Optional priority
            temperature: Temperature for generation
            **kwargs: Additional arguments

        Returns:
            Dictionary with response and metadata
        """
        llm, result = self.get_llm(prompt, priority, temperature, **kwargs)

        response = llm.invoke(prompt)

        return {
            "response": response,
            "model_used": result.model.model_id,
            "model_name": result.model.name,
            "confidence": result.confidence,
            "estimated_cost": result.estimated_cost,
            "reasoning": result.reasoning
        }

    def get_statistics(self) -> Dict[str, Any]:
        """Get orchestrator statistics"""
        return self.orchestrator.get_statistics()


def create_orchestrated_llm(
    prompt: str,
    priority: str = "balanced",
    api_key: Optional[str] = None,
    **kwargs
):
    """
    Convenience function to quickly create an LLM with automatic model selection.

    Args:
        prompt: The prompt for model selection
        priority: Selection priority
        api_key: Optional API key
        **kwargs: Additional arguments for ChatFeatherlessAI

    Returns:
        Tuple of (ChatFeatherlessAI instance, ModelSelectionResult)

    Example:
        llm, result = create_orchestrated_llm(
            "Write Python code to sort a list",
            priority="quality"
        )
        response = llm.invoke("Write a bubble sort function")
    """
    integration = FeatherlessAIIntegration(api_key=api_key)
    return integration.get_llm(prompt, priority=priority, **kwargs)


def get_model_for_agent(
    agent_type: str,
    context: Optional[str] = None,
    api_key: Optional[str] = None
) -> str:
    """
    Get optimal model ID for a specific agent type.

    Args:
        agent_type: Type of agent (e.g., "query_planning", "plot_type_chooser")
        context: Optional context about the task
        api_key: Optional API key

    Returns:
        Model ID string for use with Featherless AI

    Example:
        model_id = get_model_for_agent("query_planning")
        llm = ChatFeatherlessAI(model=model_id, api_key=api_key)
    """
    integration = FeatherlessAIIntegration(api_key=api_key)

    # Map agent types to task descriptions
    agent_descriptions = {
        "query_planning": "Analyze and plan data queries with logical reasoning",
        "plot_type_chooser": "Select appropriate data visualization types based on data analysis",
        "data_analysis": "Analyze datasets and identify patterns and trends",
        "code_generation": "Generate high-quality Python code for data processing",
        "summarization": "Summarize complex information concisely",
        "reasoning": "Perform complex logical reasoning and inference",
        "general": "General-purpose language understanding and generation"
    }

    description = agent_descriptions.get(agent_type, agent_descriptions["general"])
    if context:
        description = f"{description}. Context: {context}"

    _, result = integration.get_llm(description, priority="balanced")

    return result.model.model_id


def get_model_for_specific_agent(
    agent_type: str,
    api_key: Optional[str] = None,
    verbose: bool = False
) -> str:
    """
    Get optimal model for Numerical, Visual, Lexical, or Thinking agents.

    This function uses specialized agent-type scoring to select the best model
    for your specific agent needs.

    Args:
        agent_type: One of "thinking", "numerical", "visual", "lexical"
        api_key: Optional API key
        verbose: If True, print detailed reasoning

    Returns:
        Model ID string for use with Featherless AI

    Example:
        # For Query Planning Agent
        model = get_model_for_specific_agent("thinking")

        # For Numeric Analysis Agent
        model = get_model_for_specific_agent("numerical")

        # For Visual Appropriateness Agent
        model = get_model_for_specific_agent("visual")

        # For Lexical Analysis Agent
        model = get_model_for_specific_agent("lexical")
    """
    try:
        from agent_types import AgentType, get_model_recommendations_for_agent, explain_agent_model_selection
        from model_registry import ModelRegistry
    except ImportError:
        # Fallback to standard method if agent_types not available
        return get_model_for_agent(agent_type, api_key=api_key)

    # Map string to AgentType enum
    agent_map = {
        "thinking": AgentType.THINKING,
        "numerical": AgentType.NUMERICAL,
        "visual": AgentType.VISUAL,
        "lexical": AgentType.LEXICAL,
        # Aliases
        "query_planning": AgentType.THINKING,
        "numeric_analysis": AgentType.NUMERICAL,
        "visual_appropriateness": AgentType.VISUAL,
        "lexical_analysis": AgentType.LEXICAL,
    }

    agent_enum = agent_map.get(agent_type.lower())
    if not agent_enum:
        raise ValueError(
            f"Unknown agent type: {agent_type}. "
            f"Use one of: {', '.join(agent_map.keys())}"
        )

    # Get all available models
    registry = ModelRegistry()
    available_models = registry.get_all_models()

    # Get recommendations
    recommendations = get_model_recommendations_for_agent(
        agent_enum,
        available_models,
        top_n=3
    )

    if not recommendations:
        raise ValueError(f"No suitable models found for {agent_type}")

    # Select best model
    best_model, best_score, reasoning = recommendations[0]

    if verbose:
        print(f"\n{'='*70}")
        print(f"Model Selection for {agent_type.upper()} Agent")
        print(f"{'='*70}")
        print(explain_agent_model_selection(agent_enum, best_model))
        print(f"\nTop 3 Recommendations:")
        for i, (model, score, reason) in enumerate(recommendations, 1):
            print(f"{i}. {model.name} - Score: {score:.1f}/100")
            print(f"   {reason}")
        print(f"{'='*70}\n")

    return best_model.model_id


def get_all_agent_models(
    api_key: Optional[str] = None,
    verbose: bool = False
) -> Dict[str, str]:
    """
    Get optimal models for all four agent types at once.

    Returns:
        Dictionary mapping agent types to model IDs

    Example:
        models = get_all_agent_models(verbose=True)
        # Returns:
        # {
        #   "thinking": "Qwen/Qwen3-4B-Thinking-2507",
        #   "numerical": "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
        #   "visual": "meta-llama/Llama-3.1-8B-Instruct",
        #   "lexical": "google/gemma-2-9b-it"
        # }
    """
    agent_types = ["thinking", "numerical", "visual", "lexical"]
    models = {}

    for agent_type in agent_types:
        models[agent_type] = get_model_for_specific_agent(
            agent_type,
            api_key=api_key,
            verbose=verbose
        )

    return models
