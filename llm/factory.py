"""build_provider: single construction site for all LLM adapters."""

from __future__ import annotations

from exceptions import LLMResponseError
from llm.anthropic import AnthropicProvider
from llm.base import LLMProvider
from llm.gemini import GeminiProvider
from llm.openai import OpenAIProvider

_REGISTRY: dict[str, type] = {
    "gemini": GeminiProvider,
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
}


def build_provider(provider_name: str, api_key: str, model: str | None = None) -> LLMProvider:
    """Instantiate the named provider adapter with the given API key."""
    cls = _REGISTRY.get(provider_name.lower())
    if cls is None:
        raise LLMResponseError(f"unknown LLM provider: {provider_name!r}")
    return cls(api_key=api_key, model=model) if model else cls(api_key=api_key)
