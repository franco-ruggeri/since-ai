"""
Configuration management for Model Orchestrator.
"""

import os
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict
import json


@dataclass
class OrchestratorConfig:
    """Configuration for the orchestrator"""

    default_priority: str = "balanced"
    featherless_api_key: Optional[str] = None
    max_cost_per_1k_default: Optional[float] = None
    min_quality_score_default: Optional[float] = None
    enable_caching: bool = True
    enable_statistics: bool = True
    log_selections: bool = False
    fallback_model: str = "Qwen/Qwen3-4B-Thinking-2507"


class ConfigManager:
    """Manages configuration for the orchestrator"""

    def __init__(self, config: Optional[OrchestratorConfig] = None):
        self.config = config or self._load_default_config()

    def _load_default_config(self) -> OrchestratorConfig:
        """Load configuration from environment variables"""
        return OrchestratorConfig(
            default_priority=os.getenv("ORCHESTRATOR_PRIORITY", "balanced"),
            featherless_api_key=os.getenv("FEATHERLESS_API_KEY"),
            max_cost_per_1k_default=self._parse_float_env("ORCHESTRATOR_MAX_COST"),
            min_quality_score_default=self._parse_float_env("ORCHESTRATOR_MIN_QUALITY"),
            enable_caching=os.getenv("ORCHESTRATOR_CACHE", "true").lower() == "true",
            enable_statistics=os.getenv("ORCHESTRATOR_STATS", "true").lower() == "true",
            log_selections=os.getenv("ORCHESTRATOR_LOG", "false").lower() == "true",
            fallback_model=os.getenv(
                "ORCHESTRATOR_FALLBACK", "mistralai/Mistral-7B-Instruct-v0.2"
            ),
        )

    def _parse_float_env(self, key: str) -> Optional[float]:
        """Parse float from environment variable"""
        value = os.getenv(key)
        if value:
            try:
                return float(value)
            except ValueError:
                return None
        return None

    def save_to_file(self, filepath: str):
        """Save configuration to JSON file"""
        with open(filepath, "w") as f:
            json.dump(asdict(self.config), f, indent=2)

    def load_from_file(self, filepath: str):
        """Load configuration from JSON file"""
        with open(filepath, "r") as f:
            data = json.load(f)
            self.config = OrchestratorConfig(**data)

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value"""
        return getattr(self.config, key, default)

    def set(self, key: str, value: Any):
        """Set configuration value"""
        if hasattr(self.config, key):
            setattr(self.config, key, value)

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return asdict(self.config)
