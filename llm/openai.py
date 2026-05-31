"""OpenAIProvider: async OpenAI client, one instance per request."""

from __future__ import annotations

from collections.abc import AsyncIterator

from openai import APIStatusError as OpenAIStatusError
from openai import AsyncOpenAI, AuthenticationError, PermissionDeniedError, RateLimitError

from exceptions import LLMAuthError, LLMRateLimitError, LLMResponseError, LLMUnavailableError
from llm._prompts import Route, generation_system_user, normalize_decision, routing_prompt

_ROUTING_SYSTEM = "You are a routing classifier. Reply with ONLY one word: RAG, WEB, or DIRECT."


class OpenAIProvider:
    def __init__(self, api_key: str, model: str = "gpt-4o-mini") -> None:
        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model

    def __repr__(self) -> str:
        return f"OpenAIProvider(model={self._model!r})"

    def _map_error(self, exc: Exception) -> Exception:
        if isinstance(exc, (AuthenticationError, PermissionDeniedError)):
            return LLMAuthError()
        if isinstance(exc, RateLimitError):
            return LLMRateLimitError()
        if isinstance(exc, OpenAIStatusError) and exc.status_code in (500, 502, 503):
            return LLMUnavailableError()
        return LLMResponseError()

    async def _chat(self, system: str, user: str) -> str:
        try:
            resp = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            return (resp.choices[0].message.content or "").strip()
        except (LLMAuthError, LLMRateLimitError, LLMUnavailableError, LLMResponseError):
            raise
        except Exception as e:  # noqa: BLE001
            raise self._map_error(e) from e

    async def route(self, query: str, *, has_documents: bool, web_allowed: bool) -> Route:
        text = await self._chat(_ROUTING_SYSTEM, routing_prompt(query, has_documents, web_allowed))
        return normalize_decision(text)

    async def generate(self, query: str, context: str, decision: Route) -> str:
        sys_msg, usr_msg = generation_system_user(decision, query, context)
        return await self._chat(sys_msg, usr_msg)

    async def stream(self, query: str, context: str, decision: Route) -> AsyncIterator[str]:  # type: ignore[override]
        sys_msg, usr_msg = generation_system_user(decision, query, context)
        try:
            stream_resp = await self._client.chat.completions.create(
                model=self._model,
                messages=[
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": usr_msg},
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
