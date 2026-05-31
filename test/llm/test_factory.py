"""Tests for llm/factory.py — build_provider dispatch."""

import pytest

from exceptions import LLMResponseError
from llm.anthropic import AnthropicProvider
from llm.factory import build_provider
from llm.gemini import GeminiProvider
from llm.openai import OpenAIProvider


@pytest.mark.parametrize(
    "name, cls",
    [
        ("gemini", GeminiProvider),
        ("openai", OpenAIProvider),
        ("anthropic", AnthropicProvider),
    ],
)
def test_dispatch(name, cls, monkeypatch):
    # Patch the SDK constructors so no real clients are built
    monkeypatch.setattr("llm.gemini.genai.Client", lambda api_key=None, **kw: None)  # noqa: ARG005
    monkeypatch.setattr("llm.openai.AsyncOpenAI", lambda **kw: None)
    monkeypatch.setattr("llm.anthropic.AsyncAnthropic", lambda **kw: None)
    provider = build_provider(name, "k", model="m")
    assert isinstance(provider, cls)


def test_dispatch_case_insensitive(monkeypatch):
    monkeypatch.setattr("llm.gemini.genai.Client", lambda api_key=None, **kw: None)  # noqa: ARG005
    provider = build_provider("GEMINI", "k")
    assert isinstance(provider, GeminiProvider)


def test_unknown_provider():
    with pytest.raises(LLMResponseError):
        build_provider("bedrock", "k")


def test_model_override(monkeypatch):
    monkeypatch.setattr("llm.openai.AsyncOpenAI", lambda **kw: None)
    provider = build_provider("openai", "k", model="gpt-4o")
    assert provider._model == "gpt-4o"
