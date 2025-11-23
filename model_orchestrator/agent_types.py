"""
Agent-specific model selection for Numerical, Visual, Lexical, and Thinking agents.
"""

from enum import Enum
from dataclasses import dataclass
from typing import List, Optional
from model_registry import ModelCapabilities, TaskType


class AgentType(Enum):
    """Types of agents in the plot generation pipeline"""
    THINKING = "thinking"  # Query Planning Agent
    NUMERICAL = "numerical"  # Numeric Analysis Agent
    VISUAL = "visual"  # Visual Appropriateness Agent
    LEXICAL = "lexical"  # Lexical Analysis Agent


@dataclass
class AgentRequirements:
    """Requirements for each agent type"""
    agent_type: AgentType
    primary_tasks: List[TaskType]
    required_capabilities: List[str]
    min_context_length: int
    preferred_model_traits: dict
    description: str


# Agent-specific requirements
AGENT_REQUIREMENTS = {
    AgentType.THINKING: AgentRequirements(
        agent_type=AgentType.THINKING,
        primary_tasks=[TaskType.REASONING, TaskType.GENERAL, TaskType.QUESTION_ANSWERING],
        required_capabilities=[],
        min_context_length=8192,
        preferred_model_traits={
            "reasoning_score": 0.8,  # High reasoning required
            "instruction_following_score": 0.75,
            "speed_tier_max": 3,  # Reasonably fast
        },
        description="Query Planning Agent - Breaks down user requests into executable steps, "
                   "requires strong reasoning and planning capabilities"
    ),

    AgentType.NUMERICAL: AgentRequirements(
        agent_type=AgentType.NUMERICAL,
        primary_tasks=[TaskType.MATH, TaskType.DATA_ANALYSIS, TaskType.REASONING],
        required_capabilities=[],
        min_context_length=8192,
        preferred_model_traits={
            "reasoning_score": 0.85,  # Very high for numeric validation
            "instruction_following_score": 0.8,
            "math_capable": True,
        },
        description="Numeric Analysis Agent - Validates numeric appropriateness, data types, "
                   "statistical correctness. Requires strong math and analytical reasoning"
    ),

    AgentType.VISUAL: AgentRequirements(
        agent_type=AgentType.VISUAL,
        primary_tasks=[TaskType.REASONING, TaskType.GENERAL, TaskType.DATA_ANALYSIS],
        required_capabilities=[],
        min_context_length=8192,
        preferred_model_traits={
            "reasoning_score": 0.75,
            "creativity_score": 0.7,  # Visual judgment needs some creativity
            "instruction_following_score": 0.85,
        },
        description="Visual Appropriateness Agent - Validates visualization best practices, "
                   "user experience, and design principles. Requires reasoning and judgment"
    ),

    AgentType.LEXICAL: AgentRequirements(
        agent_type=AgentType.LEXICAL,
        primary_tasks=[TaskType.GENERAL, TaskType.QUESTION_ANSWERING],
        required_capabilities=[],
        min_context_length=8192,
        preferred_model_traits={
            "instruction_following_score": 0.9,  # Very precise language checking
            "reasoning_score": 0.7,
        },
        description="Lexical Analysis Agent - Validates language quality, terminology consistency, "
                   "and clarity. Requires excellent language understanding and precision"
    ),
}


def get_agent_requirements(agent_type: AgentType) -> AgentRequirements:
    """Get requirements for a specific agent type"""
    return AGENT_REQUIREMENTS[agent_type]


def score_model_for_agent(model: ModelCapabilities, agent_type: AgentType) -> float:
    """
    Score a model's suitability for a specific agent type.

    Returns a score from 0-100 based on how well the model matches
    the agent's requirements.
    """
    requirements = AGENT_REQUIREMENTS[agent_type]
    score = 0.0

    # Task alignment (30 points)
    if requirements.primary_tasks:
        matches = sum(1 for task in requirements.primary_tasks if task in model.strengths)
        task_score = (matches / len(requirements.primary_tasks)) * 30
        score += task_score
    else:
        score += 15  # Neutral

    # Context length requirement (10 points)
    if model.max_context_length >= requirements.min_context_length:
        score += 10
    else:
        # Partial credit if close
        ratio = model.max_context_length / requirements.min_context_length
        score += ratio * 10

    # Model trait matching (40 points)
    traits = requirements.preferred_model_traits

    if "reasoning_score" in traits:
        min_reasoning = traits["reasoning_score"]
        if model.reasoning_score >= min_reasoning:
            score += 15
        else:
            # Partial credit
            score += (model.reasoning_score / min_reasoning) * 15

    if "instruction_following_score" in traits:
        min_instruction = traits["instruction_following_score"]
        if model.instruction_following_score >= min_instruction:
            score += 15
        else:
            score += (model.instruction_following_score / min_instruction) * 15

    if "creativity_score" in traits:
        min_creativity = traits["creativity_score"]
        if model.creativity_score >= min_creativity:
            score += 10
        else:
            score += (model.creativity_score / min_creativity) * 10

    # Speed consideration (10 points)
    if "speed_tier_max" in traits:
        max_tier = traits["speed_tier_max"]
        if model.speed_tier <= max_tier:
            score += 10
        else:
            # Penalize slower models
            score += max(0, 10 - (model.speed_tier - max_tier) * 3)

    # Cost efficiency (10 points) - prefer cheaper models when quality is similar
    max_reasonable_cost = 0.001
    if model.cost_per_1k_tokens <= max_reasonable_cost:
        cost_ratio = 1 - (model.cost_per_1k_tokens / max_reasonable_cost)
        score += cost_ratio * 10

    return min(100, max(0, score))


def get_model_recommendations_for_agent(
    agent_type: AgentType,
    available_models: List[ModelCapabilities],
    top_n: int = 3
) -> List[tuple[ModelCapabilities, float, str]]:
    """
    Get top N model recommendations for a specific agent type.

    Returns:
        List of (model, score, reasoning) tuples sorted by score
    """
    requirements = AGENT_REQUIREMENTS[agent_type]

    # Filter models that meet minimum requirements
    candidates = []
    for model in available_models:
        # Must meet context length
        if model.max_context_length < requirements.min_context_length:
            continue

        # Score the model
        score = score_model_for_agent(model, agent_type)

        # Build reasoning
        reasoning = (
            f"{model.name} scored {score:.1f}/100 for {agent_type.value} agent. "
            f"Reasoning: {model.reasoning_score:.2f}, "
            f"Instruction: {model.instruction_following_score:.2f}, "
            f"Cost: ${model.cost_per_1k_tokens:.4f}/1k"
        )

        candidates.append((model, score, reasoning))

    # Sort by score descending
    candidates.sort(key=lambda x: x[1], reverse=True)

    return candidates[:top_n]


def explain_agent_model_selection(agent_type: AgentType, model: ModelCapabilities) -> str:
    """Generate a detailed explanation of why a model was selected for an agent"""
    requirements = AGENT_REQUIREMENTS[agent_type]
    score = score_model_for_agent(model, agent_type)

    explanation = f"""
Agent Type: {agent_type.value.upper()}
Description: {requirements.description}

Selected Model: {model.name} ({model.model_id})
Overall Score: {score:.1f}/100

Why this model?
- Primary Tasks: {', '.join(t.value for t in requirements.primary_tasks)}
  Model Strengths: {', '.join(t.value for t in model.strengths)}

- Reasoning Capability: {model.reasoning_score:.2f}
  Required: ≥{requirements.preferred_model_traits.get('reasoning_score', 'N/A')}

- Instruction Following: {model.instruction_following_score:.2f}
  Required: ≥{requirements.preferred_model_traits.get('instruction_following_score', 'N/A')}

- Context Length: {model.max_context_length:,} tokens
  Required: ≥{requirements.min_context_length:,} tokens

- Speed Tier: {model.speed_tier}/5
- Cost: ${model.cost_per_1k_tokens:.4f} per 1k tokens

This model provides the right balance of {', '.join(requirements.primary_tasks[0].value if requirements.primary_tasks else 'general')}
capabilities needed for {agent_type.value} tasks.
"""
    return explanation.strip()
