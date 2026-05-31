"""AnthropicProvider: async Anthropic client, one instance per request."""

from __future__ import annotations

from collections.abc import AsyncIterator

from anthropic import APIStatusError as AnthropicStatusError
from anthropic import AsyncAnthropic, AuthenticationError, PermissionDeniedError, RateLimitError
from anthropic.types import TextBlock

from exceptions import LLMAuthError, LLMRateLimitError, LLMResponseError, LLMUnavailableError
from llm._prompts import Route, generation_prompt, normalize_decision, routing_prompt

_ANTHROPIC_OVERLOADED = 529  # Anthropic-specific "overloaded" status


class AnthropicProvider:
    def __init__(self, api_key: str, model: str = "claude-3-5-haiku-latest") -> None:
        self._client = AsyncAnthropic(api_key=api_key)
        self._model = model

    def __repr__(self) -> str:
        return f"AnthropicProvider(model={self._model!r})"

    def _map_error(self, exc: Exception) -> Exception:
        if isinstance(exc, (AuthenticationError, PermissionDeniedError)):
            return LLMAuthError()
        if isinstance(exc, RateLimitError):
            return LLMRateLimitError()
        if isinstance(exc, AnthropicStatusError) and exc.status_code in (
            500,
            503,
            _ANTHROPIC_OVERLOADED,
        ):
            return LLMUnavailableError()
        return LLMResponseError()

    async def _message(self, prompt: str, max_tokens: int = 1024) -> str:
        try:
            msg = await self._client.messages.create(
                model=self._model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            text_block = next((b for b in msg.content if isinstance(b, TextBlock)), None)
            if text_block is None:
                raise LLMResponseError("Anthropic response contained no text block")
            return text_block.text.strip()
        except (LLMAuthError, LLMRateLimitError, LLMUnavailableError, LLMResponseError):
            raise
        except Exception as e:  # noqa: BLE001
            raise self._map_error(e) from e

    async def route(self, query: str, *, has_documents: bool, web_allowed: bool) -> Route:
        text = await self._message(routing_prompt(query, has_documents, web_allowed), max_tokens=8)
        return normalize_decision(text)

    async def generate(self, query: str, context: str, decision: Route) -> str:
        return await self._message(generation_prompt(decision, query, context))

    async def stream(self, query: str, context: str, decision: Route) -> AsyncIterator[str]:  # type: ignore[override]
        prompt = generation_prompt(decision, query, context)
        try:
            async with self._client.messages.stream(
                model=self._model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            ) as stream_mgr:
                async for text in stream_mgr.text_stream:
                    yield text
        except (LLMAuthError, LLMRateLimitError, LLMUnavailableError, LLMResponseError):
            raise
        except Exception as e:  # noqa: BLE001
            raise self._map_error(e) from e
