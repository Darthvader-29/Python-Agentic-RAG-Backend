"""GeminiProvider: instance-scoped Gemini client (no process-global configure).

Uses google-genai (google.genai.Client) — instance-scoped, not the deprecated
google.generativeai.configure() global that races under concurrent requests.

Per-node model tiering (Phase 6): ``route()`` uses the cheap ``route_model`` and
``generate()``/``stream()`` use the strong ``synth_model``; one client, two model ids.

Prompt caching: Gemini 2.5 does **implicit** caching keyed on a stable leading prefix. The
``routing_prompt``/``generation_prompt`` builders already lead with the byte-identical rubric /
format contract and put the variable query+context last, so no API flag is needed — the structure
is the cache key. (Explicit ``CachedContent`` is skipped: it adds storage TTL management for
prefixes well under the implicit-cache threshold.)
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import anyio
from google import genai
from google.api_core import exceptions as gexc

from exceptions import LLMAuthError, LLMRateLimitError, LLMResponseError, LLMUnavailableError
from llm._prompts import (
    Route,
    generation_prompt,
    normalize_decision,
    routing_prompt,
)

_DEFAULT_MODEL = "gemini-2.5-flash"


class GeminiProvider:
    def __init__(
        self,
        api_key: str,
        model: str | None = None,
        *,
        route_model: str | None = None,
        synth_model: str | None = None,
    ) -> None:
        # api_key is used only to construct the client; it is never stored on self.
        self._client = genai.Client(api_key=api_key)
        self._route_model = route_model or model or _DEFAULT_MODEL
        self._synth_model = synth_model or model or _DEFAULT_MODEL

    def __repr__(self) -> str:
        return (
            f"GeminiProvider(route_model={self._route_model!r}, synth_model={self._synth_model!r})"
        )

    def _map_error(self, exc: Exception) -> Exception:
        if isinstance(exc, (gexc.PermissionDenied, gexc.Unauthenticated)):
            return LLMAuthError()
        if isinstance(exc, gexc.ResourceExhausted):
            return LLMRateLimitError()
        if isinstance(exc, (gexc.ServiceUnavailable, gexc.DeadlineExceeded)):
            return LLMUnavailableError()
        return LLMResponseError()

    async def _complete(self, model: str, prompt: str) -> str:
        try:
            resp = await anyio.to_thread.run_sync(
                lambda: self._client.models.generate_content(model=model, contents=prompt)
            )
            return resp.text.strip()
        except (LLMAuthError, LLMRateLimitError, LLMUnavailableError, LLMResponseError):
            raise
        except Exception as e:  # noqa: BLE001
            raise self._map_error(e) from e

    async def route(self, query: str, *, has_documents: bool, web_allowed: bool) -> Route:
        text = await self._complete(
            self._route_model, routing_prompt(query, has_documents, web_allowed)
        )
        return normalize_decision(text)

    async def generate(self, query: str, context: str, decision: Route) -> str:
        return await self._complete(self._synth_model, generation_prompt(decision, query, context))

    async def stream(self, query: str, context: str, decision: Route) -> AsyncIterator[str]:  # type: ignore[override]
        prompt = generation_prompt(decision, query, context)
        try:
            chunks = await anyio.to_thread.run_sync(
                lambda: self._client.models.generate_content_stream(
                    model=self._synth_model, contents=prompt
                )
            )
            for chunk in chunks:  # SDK stream is a sync iterable
                if chunk.text:
                    yield chunk.text
        except (LLMAuthError, LLMRateLimitError, LLMUnavailableError, LLMResponseError):
            raise
        except Exception as e:  # noqa: BLE001
            raise self._map_error(e) from e
