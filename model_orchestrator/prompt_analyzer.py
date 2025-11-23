"""
Prompt Analyzer for determining task requirements and characteristics.
Analyzes prompts to determine optimal model selection criteria.
"""

import re
from typing import Dict, List, Optional
from dataclasses import dataclass

try:
    from .model_registry import TaskType
except ImportError:
    from model_registry import TaskType


@dataclass
class PromptAnalysis:
    """Results of prompt analysis"""
    estimated_token_count: int
    estimated_output_tokens: int
    detected_tasks: List[TaskType]
    complexity_score: float  # 0-1 scale
    requires_reasoning: bool
    requires_creativity: bool
    requires_code: bool
    requires_json: bool
    requires_function_calling: bool
    language: str = "en"
    priority: str = "balanced"  # "speed", "cost", "quality", "balanced"


class PromptAnalyzer:
    """Analyzes prompts to determine requirements for model selection"""

    def __init__(self):
        # Keywords for task detection
        self.task_keywords = {
            TaskType.CODE: [
                "code", "function", "program", "algorithm", "debug", "implement",
                "script", "api", "class", "method", "syntax", "compile", "python",
                "javascript", "java", "c++", "programming", "refactor", "bug"
            ],
            TaskType.MATH: [
                "calculate", "compute", "equation", "solve", "mathematics",
                "algebra", "geometry", "calculus", "arithmetic", "formula",
                "statistics", "probability", "derivative", "integral"
            ],
            TaskType.REASONING: [
                "analyze", "reason", "logic", "deduce", "infer", "conclude",
                "think", "evaluate", "assess", "determine", "why", "how",
                "explain", "prove", "justify", "argue"
            ],
            TaskType.CREATIVE: [
                "write", "story", "poem", "creative", "imagine", "compose",
                "design", "brainstorm", "generate ideas", "invent", "craft",
                "narrative", "fiction", "novel"
            ],
            TaskType.SUMMARIZATION: [
                "summarize", "summary", "brief", "overview", "condense",
                "abstract", "tldr", "key points", "main ideas", "recap"
            ],
            TaskType.TRANSLATION: [
                "translate", "translation", "convert to", "from english to",
                "language", "español", "français", "中文", "日本語"
            ],
            TaskType.QUESTION_ANSWERING: [
                "what is", "who is", "when", "where", "which", "answer",
                "question", "tell me", "explain", "define"
            ],
            TaskType.DATA_ANALYSIS: [
                "analyze data", "dataset", "dataframe", "statistics", "trends",
                "patterns", "visualization", "chart", "plot", "graph",
                "correlation", "distribution", "insights", "metrics"
            ],
            TaskType.INSTRUCTION_FOLLOWING: [
                "follow", "step by step", "instructions", "guide", "tutorial",
                "how to", "procedure", "process", "method"
            ]
        }

        # Complexity indicators
        self.complexity_indicators = {
            "high": [
                "complex", "sophisticated", "advanced", "comprehensive",
                "detailed", "thorough", "in-depth", "multi-step", "intricate"
            ],
            "medium": [
                "moderate", "standard", "regular", "typical", "normal"
            ],
            "low": [
                "simple", "basic", "easy", "straightforward", "quick", "brief"
            ]
        }

    def analyze(self, prompt: str, context: Optional[str] = None) -> PromptAnalysis:
        """
        Analyze a prompt to determine requirements for model selection.

        Args:
            prompt: The user's input prompt
            context: Optional additional context (e.g., previous conversation)

        Returns:
            PromptAnalysis with detected characteristics
        """
        full_text = f"{context or ''} {prompt}".lower()

        # Estimate token counts
        estimated_tokens = self._estimate_tokens(prompt)
        estimated_output = self._estimate_output_tokens(prompt)

        # Detect tasks
        detected_tasks = self._detect_tasks(full_text)

        # Analyze complexity
        complexity = self._analyze_complexity(full_text)

        # Detect special requirements
        requires_reasoning = self._requires_reasoning(full_text)
        requires_creativity = self._requires_creativity(full_text)
        requires_code = TaskType.CODE in detected_tasks
        requires_json = self._requires_json(full_text)
        requires_function_calling = self._requires_function_calling(full_text)

        # Detect language
        language = self._detect_language(prompt)

        # Determine priority
        priority = self._determine_priority(prompt)

        return PromptAnalysis(
            estimated_token_count=estimated_tokens,
            estimated_output_tokens=estimated_output,
            detected_tasks=detected_tasks if detected_tasks else [TaskType.GENERAL],
            complexity_score=complexity,
            requires_reasoning=requires_reasoning,
            requires_creativity=requires_creativity,
            requires_code=requires_code,
            requires_json=requires_json,
            requires_function_calling=requires_function_calling,
            language=language,
            priority=priority
        )

    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation (1 token ≈ 4 characters for English)"""
        return len(text) // 4 + 50  # Add buffer

    def _estimate_output_tokens(self, prompt: str) -> int:
        """Estimate required output tokens based on prompt"""
        prompt_lower = prompt.lower()

        # Look for length indicators
        if any(word in prompt_lower for word in ["brief", "short", "quick", "one sentence"]):
            return 256
        elif any(word in prompt_lower for word in ["detailed", "comprehensive", "thorough", "extensive"]):
            return 4096
        elif any(word in prompt_lower for word in ["long", "full", "complete"]):
            return 2048
        else:
            return 1024  # Default

    def _detect_tasks(self, text: str) -> List[TaskType]:
        """Detect task types based on keywords"""
        detected = []

        for task_type, keywords in self.task_keywords.items():
            if any(keyword in text for keyword in keywords):
                detected.append(task_type)

        # Remove duplicates and prioritize
        return list(set(detected))

    def _analyze_complexity(self, text: str) -> float:
        """Analyze prompt complexity (0-1 scale)"""
        score = 0.5  # Default medium

        # Check for complexity indicators
        for indicator in self.complexity_indicators["high"]:
            if indicator in text:
                score = max(score, 0.8)

        for indicator in self.complexity_indicators["low"]:
            if indicator in text:
                score = min(score, 0.3)

        # Adjust based on prompt length
        if len(text) > 500:
            score += 0.1
        elif len(text) < 100:
            score -= 0.1

        # Check for multiple questions or steps
        question_count = text.count("?")
        if question_count > 2:
            score += 0.1

        return max(0.0, min(1.0, score))

    def _requires_reasoning(self, text: str) -> bool:
        """Check if prompt requires reasoning capabilities"""
        reasoning_patterns = [
            r"\bwhy\b", r"\bhow\b", r"\bexplain\b", r"\breason\b",
            r"\bcause\b", r"\beffect\b", r"\banalyze\b", r"\bcompare\b"
        ]
        return any(re.search(pattern, text) for pattern in reasoning_patterns)

    def _requires_creativity(self, text: str) -> bool:
        """Check if prompt requires creative capabilities"""
        creative_patterns = [
            r"\bwrite\b", r"\bcreate\b", r"\bgenerate\b", r"\bimagine\b",
            r"\bstory\b", r"\bpoem\b", r"\bnovel\b", r"\bcreative\b"
        ]
        return any(re.search(pattern, text) for pattern in creative_patterns)

    def _requires_json(self, text: str) -> bool:
        """Check if prompt requires JSON output"""
        json_indicators = ["json", "structured data", "format as json", "return json"]
        return any(indicator in text for indicator in json_indicators)

    def _requires_function_calling(self, text: str) -> bool:
        """Check if prompt requires function calling"""
        function_indicators = [
            "call function", "use tool", "api call", "execute function",
            "invoke", "tool use"
        ]
        return any(indicator in text for indicator in function_indicators)

    def _detect_language(self, text: str) -> str:
        """Detect primary language (simple heuristic)"""
        # Check for non-English characters
        if re.search(r'[\u4e00-\u9fff]', text):
            return "zh"
        elif re.search(r'[\u3040-\u309f\u30a0-\u30ff]', text):
            return "ja"
        elif re.search(r'[\u0400-\u04ff]', text):
            return "ru"
        elif re.search(r'[\u0600-\u06ff]', text):
            return "ar"
        else:
            return "en"

    def _determine_priority(self, prompt: str) -> str:
        """Determine optimization priority based on prompt"""
        prompt_lower = prompt.lower()

        if any(word in prompt_lower for word in ["fast", "quick", "immediately", "asap"]):
            return "speed"
        elif any(word in prompt_lower for word in ["cheap", "budget", "cost-effective"]):
            return "cost"
        elif any(word in prompt_lower for word in ["best", "highest quality", "accurate", "precise"]):
            return "quality"
        else:
            return "balanced"
