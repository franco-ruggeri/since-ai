"""
Model Orchestrator - Main logic for intelligent model selection.
Selects the optimal model based on prompt analysis and requirements.
"""

from typing import Optional, Dict, List, Any
from dataclasses import dataclass

try:
    from .model_registry import ModelRegistry, ModelCapabilities, TaskType, ModelSize
    from .prompt_analyzer import PromptAnalyzer, PromptAnalysis
except ImportError:
    from model_registry import ModelRegistry, ModelCapabilities, TaskType, ModelSize
    from prompt_analyzer import PromptAnalyzer, PromptAnalysis


@dataclass
class ModelSelectionResult:
    """Result of model selection"""
    model: ModelCapabilities
    confidence: float  # 0-1 scale
    reasoning: str
    alternative_models: List[ModelCapabilities]
    estimated_cost: float
    estimated_latency: str


class ModelOrchestrator:
    """
    Orchestrates model selection based on prompt analysis and requirements.

    This class analyzes incoming prompts and selects the most appropriate model
    from the Featherless AI catalog based on:
    - Task type and complexity
    - Token requirements (input/output)
    - Performance requirements (speed vs quality)
    - Cost constraints
    - Special capabilities (JSON mode, function calling, etc.)
    """

    def __init__(self, default_priority: str = "balanced"):
        """
        Initialize the orchestrator.

        Args:
            default_priority: Default optimization priority ("speed", "cost", "quality", "balanced")
        """
        self.registry = ModelRegistry()
        self.analyzer = PromptAnalyzer()
        self.default_priority = default_priority
        self.selection_history: List[Dict[str, Any]] = []

    def select_model(
        self,
        prompt: str,
        context: Optional[str] = None,
        priority: Optional[str] = None,
        max_cost_per_1k: Optional[float] = None,
        min_quality_score: Optional[float] = None,
        required_capabilities: Optional[List[str]] = None
    ) -> ModelSelectionResult:
        """
        Select the optimal model for a given prompt.

        Args:
            prompt: The user's input prompt
            context: Optional context from previous conversation
            priority: Override priority ("speed", "cost", "quality", "balanced")
            max_cost_per_1k: Maximum cost per 1k tokens
            min_quality_score: Minimum quality score required
            required_capabilities: List of required capabilities (e.g., ["json_mode", "function_calling"])

        Returns:
            ModelSelectionResult with selected model and metadata
        """
        # Analyze the prompt
        analysis = self.analyzer.analyze(prompt, context)

        # Use provided priority or fall back to detected/default
        effective_priority = priority or analysis.priority or self.default_priority

        # Get candidate models
        candidates = self._filter_candidates(
            analysis,
            max_cost_per_1k,
            min_quality_score,
            required_capabilities
        )

        if not candidates:
            # Fallback to a safe general-purpose model
            candidates = [self.registry.get_model("mistralai/Mistral-7B-Instruct-v0.2")]

        # Score and rank candidates
        scored_candidates = self._score_candidates(candidates, analysis, effective_priority)

        # Select the best model
        best_model, confidence, reasoning = self._select_best(scored_candidates, analysis, effective_priority)

        # Calculate estimated cost
        total_tokens = analysis.estimated_token_count + analysis.estimated_output_tokens
        estimated_cost = (total_tokens / 1000) * best_model.cost_per_1k_tokens

        # Estimate latency
        estimated_latency = self._estimate_latency(best_model, analysis.estimated_output_tokens)

        # Get alternatives
        alternatives = [model for model, _, _ in scored_candidates[1:4]]

        result = ModelSelectionResult(
            model=best_model,
            confidence=confidence,
            reasoning=reasoning,
            alternative_models=alternatives,
            estimated_cost=estimated_cost,
            estimated_latency=estimated_latency
        )

        # Record selection
        self._record_selection(prompt, analysis, result)

        return result

    def _filter_candidates(
        self,
        analysis: PromptAnalysis,
        max_cost_per_1k: Optional[float],
        min_quality_score: Optional[float],
        required_capabilities: Optional[List[str]]
    ) -> List[ModelCapabilities]:
        """Filter models based on hard requirements"""
        candidates = self.registry.get_all_models()

        # Filter by context length
        required_context = analysis.estimated_token_count + 512  # Add buffer
        candidates = [m for m in candidates if m.max_context_length >= required_context]

        # Filter by output tokens
        candidates = [m for m in candidates if m.max_output_tokens >= analysis.estimated_output_tokens]

        # Filter by cost
        if max_cost_per_1k:
            candidates = [m for m in candidates if m.cost_per_1k_tokens <= max_cost_per_1k]

        # Filter by capabilities
        if required_capabilities:
            if "json_mode" in required_capabilities:
                candidates = [m for m in candidates if m.supports_json_mode]
            if "function_calling" in required_capabilities:
                candidates = [m for m in candidates if m.supports_function_calling]

        # Filter by quality score
        if min_quality_score:
            candidates = [
                m for m in candidates
                if (m.reasoning_score + m.instruction_following_score) / 2 >= min_quality_score
            ]

        # Filter by task compatibility
        if analysis.detected_tasks:
            task_compatible = []
            for model in candidates:
                if any(task in model.strengths for task in analysis.detected_tasks):
                    task_compatible.append(model)
            if task_compatible:
                candidates = task_compatible

        return candidates

    def _score_candidates(
        self,
        candidates: List[ModelCapabilities],
        analysis: PromptAnalysis,
        priority: str
    ) -> List[tuple[ModelCapabilities, float, str]]:
        """Score and rank candidates based on requirements and priority"""
        scored = []

        for model in candidates:
            score = 0.0
            score_breakdown = []

            # Task alignment score (0-30 points)
            task_score = self._calculate_task_score(model, analysis)
            score += task_score
            score_breakdown.append(f"task:{task_score:.1f}")

            # Complexity alignment (0-20 points)
            complexity_score = self._calculate_complexity_score(model, analysis)
            score += complexity_score
            score_breakdown.append(f"complexity:{complexity_score:.1f}")

            # Capability match (0-20 points)
            capability_score = self._calculate_capability_score(model, analysis)
            score += capability_score
            score_breakdown.append(f"capability:{capability_score:.1f}")

            # Priority-based scoring (0-30 points)
            priority_score = self._calculate_priority_score(model, priority)
            score += priority_score
            score_breakdown.append(f"priority:{priority_score:.1f}")

            reasoning = " | ".join(score_breakdown)
            scored.append((model, score, reasoning))

        # Sort by score (descending)
        scored.sort(key=lambda x: x[1], reverse=True)

        return scored

    def _calculate_task_score(self, model: ModelCapabilities, analysis: PromptAnalysis) -> float:
        """Calculate how well the model matches the task requirements"""
        if not analysis.detected_tasks:
            return 15.0  # Neutral score

        matches = sum(1 for task in analysis.detected_tasks if task in model.strengths)
        return (matches / len(analysis.detected_tasks)) * 30.0

    def _calculate_complexity_score(self, model: ModelCapabilities, analysis: PromptAnalysis) -> float:
        """Calculate how well the model's capabilities match the complexity"""
        size_complexity_map = {
            ModelSize.TINY: 0.3,
            ModelSize.SMALL: 0.5,
            ModelSize.MEDIUM: 0.7,
            ModelSize.LARGE: 0.9,
            ModelSize.XLARGE: 1.0
        }

        model_capability = size_complexity_map.get(model.size, 0.5)
        difference = abs(model_capability - analysis.complexity_score)

        # Perfect match = 20 points, decreasing with difference
        return max(0, 20.0 - (difference * 30))

    def _calculate_capability_score(self, model: ModelCapabilities, analysis: PromptAnalysis) -> float:
        """Calculate score based on special capability requirements"""
        score = 0.0

        if analysis.requires_reasoning:
            score += model.reasoning_score * 8

        if analysis.requires_creativity:
            score += model.creativity_score * 8

        if analysis.requires_json and model.supports_json_mode:
            score += 4

        if analysis.requires_function_calling and model.supports_function_calling:
            score += 4

        # Instruction following is always valuable
        score += model.instruction_following_score * 4

        return min(score, 20.0)

    def _calculate_priority_score(self, model: ModelCapabilities, priority: str) -> float:
        """Calculate score based on optimization priority"""
        if priority == "speed":
            # Prefer faster models
            return (6 - model.speed_tier) * 6  # Tier 1 = 30pts, Tier 5 = 6pts

        elif priority == "cost":
            # Prefer cheaper models (inverse of cost)
            max_cost = 0.002
            cost_ratio = 1 - min(model.cost_per_1k_tokens / max_cost, 1.0)
            return cost_ratio * 30

        elif priority == "quality":
            # Prefer higher quality models
            quality = (
                model.reasoning_score * 0.4 +
                model.instruction_following_score * 0.4 +
                model.creativity_score * 0.2
            )
            return quality * 30

        else:  # balanced
            # Balanced scoring
            speed_score = (6 - model.speed_tier) * 3
            cost_score = (1 - min(model.cost_per_1k_tokens / 0.001, 1.0)) * 10
            quality_score = (
                model.reasoning_score * 0.4 +
                model.instruction_following_score * 0.4
            ) * 17
            return speed_score + cost_score + quality_score

    def _select_best(
        self,
        scored_candidates: List[tuple[ModelCapabilities, float, str]],
        analysis: PromptAnalysis,
        priority: str
    ) -> tuple[ModelCapabilities, float, str]:
        """Select the best model from scored candidates"""
        if not scored_candidates:
            # Emergency fallback
            fallback = self.registry.get_model("mistralai/Mistral-7B-Instruct-v0.2")
            return fallback, 0.5, "Fallback model selected"

        best_model, best_score, score_breakdown = scored_candidates[0]

        # Calculate confidence based on score gap
        if len(scored_candidates) > 1:
            second_score = scored_candidates[1][1]
            score_gap = best_score - second_score
            confidence = min(0.95, 0.6 + (score_gap / 100))
        else:
            confidence = 0.95

        # Build reasoning
        reasoning = (
            f"Selected {best_model.name} for {', '.join(t.value for t in analysis.detected_tasks[:2])} "
            f"with {priority} priority. Score: {best_score:.1f} ({score_breakdown})"
        )

        return best_model, confidence, reasoning

    def _estimate_latency(self, model: ModelCapabilities, output_tokens: int) -> str:
        """Estimate response latency"""
        base_latency = {
            1: 0.5,
            2: 1.0,
            3: 2.0,
            4: 4.0,
            5: 8.0
        }

        base = base_latency.get(model.speed_tier, 2.0)
        token_time = (output_tokens / 1000) * base

        total = base + token_time

        if total < 1:
            return "< 1 second"
        elif total < 5:
            return f"{total:.1f} seconds"
        elif total < 30:
            return f"~{int(total)} seconds"
        else:
            return f"~{int(total/60)} minutes"

    def _record_selection(
        self,
        prompt: str,
        analysis: PromptAnalysis,
        result: ModelSelectionResult
    ):
        """Record selection for analysis and improvement"""
        self.selection_history.append({
            "prompt_preview": prompt[:100],
            "model_selected": result.model.model_id,
            "confidence": result.confidence,
            "detected_tasks": [t.value for t in analysis.detected_tasks],
            "priority": analysis.priority,
            "estimated_cost": result.estimated_cost
        })

        # Keep only last 100 selections
        if len(self.selection_history) > 100:
            self.selection_history.pop(0)

    def get_model_recommendations(
        self,
        task_type: TaskType,
        max_results: int = 5
    ) -> List[ModelCapabilities]:
        """Get recommended models for a specific task type"""
        candidates = self.registry.filter_by_task(task_type)

        # Sort by a combination of capability and cost
        candidates.sort(
            key=lambda m: (
                -(m.reasoning_score + m.instruction_following_score) / 2,
                m.cost_per_1k_tokens
            )
        )

        return candidates[:max_results]

    def get_statistics(self) -> Dict[str, Any]:
        """Get orchestrator statistics"""
        if not self.selection_history:
            return {"total_selections": 0}

        total = len(self.selection_history)
        avg_confidence = sum(s["confidence"] for s in self.selection_history) / total
        total_cost = sum(s["estimated_cost"] for s in self.selection_history)

        model_counts = {}
        for selection in self.selection_history:
            model = selection["model_selected"]
            model_counts[model] = model_counts.get(model, 0) + 1

        return {
            "total_selections": total,
            "average_confidence": avg_confidence,
            "total_estimated_cost": total_cost,
            "model_usage": model_counts
        }
