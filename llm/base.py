from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Literal, Protocol, runtime_checkable

Route = Literal["RAG", "WEB", "DIRECT"]


@runtime_checkable
class LLMProvider(Protocol):
    """Provider-agnostic LLM interface. Exactly one instance per request."""

    async def route(self, query: str, *, has_documents: bool, web_allowed: bool) -> Route:
        """Classify a query into RAG / WEB / DIRECT."""
        ...

    async def generate(self, query: str, context: str, decision: Route) -> str:
        """Produce the final answer for the decided route."""
        ...

    def stream(self, query: str, context: str, decision: Route) -> AsyncIterator[str]:
        """Yield answer text deltas (consumed by SSE in Phase 6)."""
        ...
