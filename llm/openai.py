"""OpenAIProvider: async OpenAI client, one instance per request.

Per-node model tiering (Phase 6): ``route()`` uses the cheap ``route_model`` and
``generate()``/``stream()`` use the strong ``synth_model``.

Prompt caching: OpenAI does **automatic** prefix caching (no API flag) for stable leading
content. We keep the stable instruction (the routing rubric / the role+format contract) in the
``system`` message — always the prefix — and the variable query+context in the trailing ``user``
message. Never inject per-request data into the system message or the cache prefix breaks.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

from openai import APIStatusError as OpenAIStatusError
from openai import AsyncOpenAI, AuthenticationError, PermissionDeniedError, RateLimitError

from exceptions import LLMAuthError, LLMRateLimitError, LLMResponseError, LLMUnavailableError
from llm._prompts import (
    ROUTING_SYSTEM,
    Route,
    generation_system_user,
    normalize_decision,
    routing_user,
)

_DEFAULT_MODEL = "gpt-4o-mini"


class OpenAIProvider:
    def __init__(
        self,
        api_key: str,
        model: str | None = None,
        *,
        route_model: str | None = None,
        synth_model: str | None = None,
    ) -> None:
        # api_key is used only to construct the client; it is never stored on self.
        self._client = AsyncOpenAI(api_key=api_key)
        self._route_model = route_model or model or _DEFAULT_MODEL
        self._synth_model = synth_model or model or _DEFAULT_MODEL

    def __repr__(self) -> str:
        return (
            f"OpenAIProvider(route_model={self._route_model!r}, synth_model={self._synth_model!r})"
        )

    def _map_error(self, exc: Exception) -> Exception:
        if isinstance(exc, (AuthenticationError, PermissionDeniedError)):
            return LLMAuthError()
        if isinstance(exc, RateLimitError):
            return LLMRateLimitError()
        if isinstance(exc, OpenAIStatusError) and exc.status_code in (500, 502, 503):
            return LLMUnavailableError()
        return LLMResponseError()

    async def _chat(self, model: str, system: str, user: str) -> str:
        try:
            resp = await self._client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},  # stable prefix → auto-cached
                    {"role": "user", "content": user},  # variable suffix
                ],
            )
            return (resp.choices[0].message.content or "").strip()
        except (LLMAuthError, LLMRateLimitError, LLMUnavailableError, LLMResponseError):
            raise
        except Exception as e:  # noqa: BLE001
            raise self._map_error(e) from e

    async def route(self, query: str, *, has_documents: bool, web_allowed: bool) -> Route:
        text = await self._chat(
            self._route_model, ROUTING_SYSTEM, routing_user(query, has_documents, web_allowed)
        )
        return normalize_decision(text)

    async def generate(self, query: str, context: str, decision: Route) -> str:
        sys_msg, usr_msg = generation_system_user(decision, query, context)
        return await self._chat(self._synth_model, sys_msg, usr_msg)

    async def stream(self, query: str, context: str, decision: Route) -> AsyncIterator[str]:  # type: ignore[override]
        sys_msg, usr_msg = generation_system_user(decision, query, context)
        try:
            stream_resp = await self._client.chat.completions.create(
                model=self._synth_model,
                messages=[
                    {"role": "system", "content": sys_msg},  # stable prefix → auto-cached
                    {"role": "user", "content": usr_msg},  # variable suffix
                ],
                stream=True,
            )
            async for event in stream_resp:
                delta = event.choices[0].delta.content
                if delta:
                    yield delta
        except (LLMAuthError, LLMRateLimitError, LLMUnavailableError, LLMResponseError):
            raise
        except Exception as e:  # noqa: BLE001
            raise self._map_error(e) from e
