"""GeminiProvider: instance-scoped Gemini client (no process-global configure).

Uses google-genai (google.genai.Client) — instance-scoped, not the deprecated
google.generativeai.configure() global that races under concurrent requests.
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


class GeminiProvider:
    def __init__(self, api_key: str, model: str = "gemini-2.5-flash") -> None:
        self._client = genai.Client(api_key=api_key)
        self._model = model

    def __repr__(self) -> str:
        return f"GeminiProvider(model={self._model!r})"

    def _map_error(self, exc: Exception) -> Exception:
        if isinstance(exc, (gexc.PermissionDenied, gexc.Unauthenticated)):
            return LLMAuthError()
        if isinstance(exc, gexc.ResourceExhausted):
            return LLMRateLimitError()
        if isinstance(exc, (gexc.ServiceUnavailable, gexc.DeadlineExceeded)):
            return LLMUnavailableError()
        return LLMResponseError()

    async def _complete(self, prompt: str) -> str:
        try:
            resp = await anyio.to_thread.run_sync(
                lambda: self._client.models.generate_content(model=self._model, contents=prompt)
            )
            return resp.text.strip()
        except (LLMAuthError, LLMRateLimitError, LLMUnavailableError, LLMResponseError):
            raise
        except Exception as e:  # noqa: BLE001
            raise self._map_error(e) from e

    async def route(self, query: str, *, has_documents: bool, web_allowed: bool) -> Route:
        text = await self._complete(routing_prompt(query, has_documents, web_allowed))
        return normalize_decision(text)

    async def generate(self, query: str, context: str, decision: Route) -> str:
        return await self._complete(generation_prompt(decision, query, context))

    async def stream(self, query: str, context: str, decision: Route) -> AsyncIterator[str]:  # type: ignore[override]
        prompt = generation_prompt(decision, query, context)
        try:
            chunks = await anyio.to_thread.run_sync(
                lambda: self._client.models.generate_content_stream(
                    model=self._model, contents=prompt
                )
            )
            for chunk in chunks:  # SDK stream is a sync iterable
                if chunk.text:
                    yield chunk.text
        except (LLMAuthError, LLMRateLimitError, LLMUnavailableError, LLMResponseError):
            raise
        except Exception as e:  # noqa: BLE001
            raise self._map_error(e) from e
