"""AnthropicProvider: async Anthropic client, one instance per request.

Per-node model tiering (Phase 6): ``route()`` uses the cheap ``route_model`` and
``generate()``/``stream()`` use the strong ``synth_model``.

Prompt caching (Decision 9): the stable instruction — the routing rubric for ``route()`` and the
role+format contract for ``generate()``/``stream()`` — goes in a ``system`` block carrying
``cache_control={"type": "ephemeral"}`` (~90% off cached input tokens, up to 2x faster). The
variable query+context stays in the ``user`` message and is therefore NEVER cached. The system
block contains no per-request data, so it is byte-identical across requests and reliably hits.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from anthropic import APIStatusError as AnthropicStatusError
from anthropic import AsyncAnthropic, AuthenticationError, PermissionDeniedError, RateLimitError
from anthropic.types import TextBlock, TextBlockParam

from exceptions import LLMAuthError, LLMRateLimitError, LLMResponseError, LLMUnavailableError
from llm._prompts import (
    ROUTING_SYSTEM,
    Route,
    generation_system_user,
    normalize_decision,
    routing_user,
)

_ANTHROPIC_OVERLOADED = 529  # Anthropic-specific "overloaded" status
_DEFAULT_MODEL = "claude-3-5-haiku-latest"


def _cached_system(text: str) -> list[TextBlockParam]:
    """Wrap a stable instruction as a cache-eligible Anthropic system block."""
    return [
        TextBlockParam(type="text", text=text, cache_control={"type": "ephemeral"}),
    ]


class AnthropicProvider:
    def __init__(
        self,
        api_key: str,
        model: str | None = None,
        *,
        route_model: str | None = None,
        synth_model: str | None = None,
    ) -> None:
        # api_key is used only to construct the client; it is never stored on self.
        self._client = AsyncAnthropic(api_key=api_key)
        self._route_model = route_model or model or _DEFAULT_MODEL
        self._synth_model = synth_model or model or _DEFAULT_MODEL

    def __repr__(self) -> str:
        return (
            f"AnthropicProvider(route_model={self._route_model!r}, "
            f"synth_model={self._synth_model!r})"
        )

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

    async def _message(self, model: str, system: str, user: str, max_tokens: int = 1024) -> str:
        try:
            msg = await self._client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=_cached_system(system),  # stable prefix → ephemeral cache
                messages=[{"role": "user", "content": user}],  # variable suffix
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
        text = await self._message(
            self._route_model,
            ROUTING_SYSTEM,
            routing_user(query, has_documents, web_allowed),
            max_tokens=8,
        )
        return normalize_decision(text)

    async def generate(self, query: str, context: str, decision: Route) -> str:
        sys_msg, usr_msg = generation_system_user(decision, query, context)
        return await self._message(self._synth_model, sys_msg, usr_msg)

    async def stream(self, query: str, context: str, decision: Route) -> AsyncIterator[str]:  # type: ignore[override]
        sys_msg, usr_msg = generation_system_user(decision, query, context)
        try:
            async with self._client.messages.stream(
                model=self._synth_model,
                max_tokens=1024,
                system=_cached_system(sys_msg),  # stable prefix → ephemeral cache
                messages=[{"role": "user", "content": usr_msg}],  # variable suffix
            ) as stream_mgr:
                async for text in stream_mgr.text_stream:
                    yield text
        except (LLMAuthError, LLMRateLimitError, LLMUnavailableError, LLMResponseError):
            raise
        except Exception as e:  # noqa: BLE001
            raise self._map_error(e) from e
