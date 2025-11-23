"""LLM Provider Interface for supporting multiple LLM backends."""

import os
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
from dotenv import load_dotenv
import streamlit as st

load_dotenv()


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def invoke(
        self,
        messages: List[Tuple[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        seed: int = 42,
        **kwargs,
    ) -> str:
        """
        Invoke the LLM with the given messages.

        Args:
            messages: List of (role, content) tuples. Role is typically "system" or "human"
            model: Optional model name override
            temperature: Temperature for response generation
            **kwargs: Additional provider-specific parameters

        Returns:
            String response from the LLM
        """
        pass


class FeatherlessProvider(LLMProvider):
    """Featherless AI provider using langchain-featherless-ai."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        default_model: Optional[str] = None,
    ):
        """
        Initialize Featherless provider.

        Args:
            api_key: Featherless API key (defaults to FEATHERLESS_API_KEY env var)
            base_url: Featherless API URL (defaults to FEATHERLESS_API_URL env var or https://api.featherless.ai/v1)
            default_model: Default model to use
        """
        try:
            from langchain_featherless_ai import ChatFeatherlessAi
            from pydantic import SecretStr
        except ImportError:
            raise RuntimeError(
                "langchain-featherless-ai package is not installed. "
                "Install it with: pip install langchain-featherless-ai"
            )

        self.api_key = api_key or st.secrets["FEATHERLESS_API_KEYY"]
        if not self.api_key:
            raise ValueError(
                "FEATHERLESS_API_KEY not set. Pass it to the constructor or set it in the environment."
            )

        self.base_url = base_url or os.environ.get(
            "FEATHERLESS_API_URL", "https://api.featherless.ai/v1"
        )
        self.default_model = default_model
        self.llm = ChatFeatherlessAi(
            api_key=SecretStr(self.api_key), base_url=self.base_url
        )

    def invoke(
        self,
        messages: List[Tuple[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        seed: int = 42,
        **kwargs,
    ) -> str:
        """Invoke Featherless API."""
        model_to_use = model or self.default_model

        if model_to_use:
            response = self.llm.invoke(
                messages,
                model=model_to_use,
                temperature=temperature,
                seed=seed,
                **kwargs,
            )
        else:
            response = self.llm.invoke(
                messages, temperature=temperature, seed=seed, **kwargs
            )

        # Extract content from response
        if isinstance(response, str):
            return response
        elif hasattr(response, "content"):
            return response.content
        elif isinstance(response, dict):
            try:
                return response.get("choices", [])[0].get("message", {}).get("content")
            except (IndexError, KeyError, TypeError):
                return str(response)
        else:
            return str(response)


class GeminiProvider(LLMProvider):
    """Google Gemini provider using langchain-google-genai."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        default_model: str = "gemini-2.5-flash-lite",
    ):
        """
        Initialize Gemini provider.

        Args:
            api_key: Google API key (defaults to GOOGLE_API_KEY env var)
                    default_model: Default Gemini model to use (default: gemini-2.5-flash-lite)
        """
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError:
            raise RuntimeError(
                "langchain-google-genai package is not installed. "
                "Install it with: pip install langchain-google-genai"
            )

        self.api_key = api_key or st.secrets["GOOGLE_API_KEY"]
        if not self.api_key:
            raise ValueError(
                "GOOGLE_API_KEY not set. Pass it to the constructor or set it in the environment."
            )

        self.default_model = default_model
        self.ChatGoogleGenerativeAI = ChatGoogleGenerativeAI

    def invoke(
        self,
        messages: List[Tuple[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        seed: int = 42,
        **kwargs,
    ) -> str:
        """Invoke Google Gemini API."""
        model_to_use = self.default_model

        # Create LLM instance with model
        llm = self.ChatGoogleGenerativeAI(
            model=model_to_use,
            google_api_key=self.api_key,
            temperature=temperature,
            model_kwargs={"seed": 42},
            **kwargs,
        )

        response = llm.invoke(messages)

        # Extract content from response
        if isinstance(response, str):
            return response
        elif hasattr(response, "content"):
            return response.content
        elif isinstance(response, dict):
            try:
                return response.get("choices", [])[0].get("message", {}).get("content")
            except (IndexError, KeyError, TypeError):
                return str(response)
        else:
            return str(response)


def get_llm_provider(
    provider_name: Optional[str] = None, api_key: Optional[str] = None, **kwargs
) -> LLMProvider:
    """
    Factory function to get an LLM provider instance.

    Args:
        provider_name: Name of provider ("featherless" or "gemini").
                      Defaults to LLM_PROVIDER env var or "featherless"
        api_key: API key for the provider. If not provided, reads from environment
        **kwargs: Additional provider-specific arguments (base_url, default_model, etc.)

    Returns:
        Configured LLM provider instance

    Raises:
        ValueError: If provider_name is not supported

    Example:
        >>> # Use Featherless (default)
        >>> provider = get_llm_provider()

        >>> # Use Gemini
        >>> provider = get_llm_provider("gemini", api_key="your-key")

        >>> # Use environment variable
        >>> os.environ["LLM_PROVIDER"] = "gemini"
        >>> provider = get_llm_provider()
    """
    provider = provider_name or os.environ.get("LLM_PROVIDER", "featherless")
    provider = provider.lower().strip()

    if provider in ("gemini", "google"):
        return GeminiProvider(api_key=api_key, **kwargs)
    else:
        return FeatherlessProvider(api_key=api_key, **kwargs)
