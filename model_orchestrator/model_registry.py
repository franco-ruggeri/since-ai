"""
Model Registry for Featherless AI models.
Maintains catalog of available models with their capabilities and constraints.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict
from enum import Enum


class TaskType(Enum):
    """Types of tasks that models can perform"""
    GENERAL = "general"
    CODE = "code"
    MATH = "math"
    REASONING = "reasoning"
    CREATIVE = "creative"
    SUMMARIZATION = "summarization"
    TRANSLATION = "translation"
    QUESTION_ANSWERING = "qa"
    DATA_ANALYSIS = "data_analysis"
    INSTRUCTION_FOLLOWING = "instruction"


class ModelSize(Enum):
    """Model size categories"""
    TINY = "tiny"  # < 1B params
    SMALL = "small"  # 1-7B params
    MEDIUM = "medium"  # 7-20B params
    LARGE = "large"  # 20-70B params
    XLARGE = "xlarge"  # > 70B params


@dataclass
class ModelCapabilities:
    """Defines model capabilities and constraints"""
    model_id: str
    name: str
    size: ModelSize
    max_context_length: int
    max_output_tokens: int
    strengths: List[TaskType]
    cost_per_1k_tokens: float
    speed_tier: int  # 1 (fastest) to 5 (slowest)
    supports_function_calling: bool = False
    supports_json_mode: bool = False
    reasoning_score: float = 0.5  # 0-1 scale
    creativity_score: float = 0.5  # 0-1 scale
    instruction_following_score: float = 0.5  # 0-1 scale


class ModelRegistry:
    """Registry of available Featherless AI models"""

    def __init__(self):
        self.models = self._initialize_models()

    def _initialize_models(self) -> List[ModelCapabilities]:
        """Initialize the model catalog with popular HuggingFace models available on Featherless AI"""
        return [
            # Tiny models - fast and cheap
            ModelCapabilities(
                model_id="microsoft/phi-2",
                name="Phi-2",
                size=ModelSize.TINY,
                max_context_length=2048,
                max_output_tokens=1024,
                strengths=[TaskType.CODE, TaskType.MATH, TaskType.REASONING],
                cost_per_1k_tokens=0.0001,
                speed_tier=1,
                supports_json_mode=True,
                reasoning_score=0.7,
                instruction_following_score=0.6
            ),

            # Small models - good balance
            ModelCapabilities(
                model_id="mistralai/Mistral-7B-Instruct-v0.2",
                name="Mistral 7B Instruct",
                size=ModelSize.SMALL,
                max_context_length=8192,
                max_output_tokens=4096,
                strengths=[TaskType.GENERAL, TaskType.INSTRUCTION_FOLLOWING, TaskType.CODE],
                cost_per_1k_tokens=0.0002,
                speed_tier=2,
                supports_json_mode=True,
                reasoning_score=0.75,
                instruction_following_score=0.85
            ),

            ModelCapabilities(
                model_id="Qwen/Qwen3-4B-Thinking-2507",
                name="Qwen 4B Thinking",
                size=ModelSize.SMALL,
                max_context_length=32768,
                max_output_tokens=8192,
                strengths=[TaskType.REASONING, TaskType.MATH, TaskType.GENERAL],
                cost_per_1k_tokens=0.0002,
                speed_tier=2,
                reasoning_score=0.85,
                instruction_following_score=0.8
            ),

            ModelCapabilities(
                model_id="google/gemma-2-9b-it",
                name="Gemma 2 9B",
                size=ModelSize.SMALL,
                max_context_length=8192,
                max_output_tokens=4096,
                strengths=[TaskType.GENERAL, TaskType.INSTRUCTION_FOLLOWING, TaskType.SUMMARIZATION],
                cost_per_1k_tokens=0.0003,
                speed_tier=2,
                supports_json_mode=True,
                reasoning_score=0.75,
                instruction_following_score=0.85
            ),

            # Medium models - high capability
            ModelCapabilities(
                model_id="meta-llama/Llama-3.1-8B-Instruct",
                name="Llama 3.1 8B",
                size=ModelSize.MEDIUM,
                max_context_length=131072,
                max_output_tokens=16384,
                strengths=[TaskType.GENERAL, TaskType.CODE, TaskType.REASONING, TaskType.CREATIVE],
                cost_per_1k_tokens=0.0004,
                speed_tier=2,
                supports_function_calling=True,
                supports_json_mode=True,
                reasoning_score=0.8,
                creativity_score=0.75,
                instruction_following_score=0.85
            ),

            ModelCapabilities(
                model_id="mistralai/Mistral-Small-3.1-24B-Instruct-2503",
                name="Mistral Small 24B",
                size=ModelSize.MEDIUM,
                max_context_length=32768,
                max_output_tokens=16384,
                strengths=[TaskType.GENERAL, TaskType.CODE, TaskType.REASONING, TaskType.DATA_ANALYSIS],
                cost_per_1k_tokens=0.0005,
                speed_tier=3,
                supports_json_mode=True,
                supports_function_calling=True,
                reasoning_score=0.85,
                instruction_following_score=0.9
            ),

            # Large models - highest capability
            ModelCapabilities(
                model_id="meta-llama/Llama-3.1-70B-Instruct",
                name="Llama 3.1 70B",
                size=ModelSize.LARGE,
                max_context_length=131072,
                max_output_tokens=16384,
                strengths=[TaskType.GENERAL, TaskType.CODE, TaskType.REASONING, TaskType.CREATIVE, TaskType.MATH],
                cost_per_1k_tokens=0.001,
                speed_tier=4,
                supports_function_calling=True,
                supports_json_mode=True,
                reasoning_score=0.95,
                creativity_score=0.9,
                instruction_following_score=0.95
            ),

            ModelCapabilities(
                model_id="Qwen/Qwen2.5-72B-Instruct",
                name="Qwen 2.5 72B",
                size=ModelSize.LARGE,
                max_context_length=131072,
                max_output_tokens=16384,
                strengths=[TaskType.GENERAL, TaskType.CODE, TaskType.MATH, TaskType.REASONING],
                cost_per_1k_tokens=0.001,
                speed_tier=4,
                supports_json_mode=True,
                reasoning_score=0.95,
                instruction_following_score=0.92
            ),

            ModelCapabilities(
                model_id="deepseek-ai/DeepSeek-V3",
                name="DeepSeek V3",
                size=ModelSize.XLARGE,
                max_context_length=65536,
                max_output_tokens=8192,
                strengths=[TaskType.CODE, TaskType.REASONING, TaskType.MATH, TaskType.GENERAL],
                cost_per_1k_tokens=0.0015,
                speed_tier=5,
                supports_json_mode=True,
                reasoning_score=0.98,
                instruction_following_score=0.95
            ),

            # Specialized models
            ModelCapabilities(
                model_id="codellama/CodeLlama-34b-Instruct-hf",
                name="Code Llama 34B",
                size=ModelSize.MEDIUM,
                max_context_length=16384,
                max_output_tokens=8192,
                strengths=[TaskType.CODE],
                cost_per_1k_tokens=0.0006,
                speed_tier=3,
                reasoning_score=0.8,
                instruction_following_score=0.85
            ),
        ]

    def get_model(self, model_id: str) -> Optional[ModelCapabilities]:
        """Get model by ID"""
        for model in self.models:
            if model.model_id == model_id:
                return model
        return None

    def filter_by_task(self, task_type: TaskType) -> List[ModelCapabilities]:
        """Filter models by task type"""
        return [model for model in self.models if task_type in model.strengths]

    def filter_by_context_length(self, min_context_length: int) -> List[ModelCapabilities]:
        """Filter models by minimum context length"""
        return [model for model in self.models if model.max_context_length >= min_context_length]

    def filter_by_size(self, max_size: ModelSize) -> List[ModelCapabilities]:
        """Filter models by maximum size"""
        size_order = [ModelSize.TINY, ModelSize.SMALL, ModelSize.MEDIUM, ModelSize.LARGE, ModelSize.XLARGE]
        max_index = size_order.index(max_size)
        return [model for model in self.models if size_order.index(model.size) <= max_index]

    def get_all_models(self) -> List[ModelCapabilities]:
        """Get all registered models"""
        return self.models

    def add_custom_model(self, model: ModelCapabilities):
        """Add a custom model to the registry"""
        self.models.append(model)
