"""Router module: delegates query classification to the injected LLM provider.

Phase 4: the Gemini process-global and GoogleAPIError ladder are removed.
Error mapping lives in the provider adapter; neutral LLM errors bubble to
app_exception_handler in exceptions.py.
"""

import structlog

from llm.base import LLMProvider, Route

logger = structlog.get_logger(__name__)


async def route_query(
    provider: LLMProvider,
    query: str,
    *,
    has_documents: bool,
    web_search_allowed: bool,
) -> Route:
    """Route query to RAG, WEB, or DIRECT using the injected provider."""
    decision = await provider.route(
        query, has_documents=has_documents, web_allowed=web_search_allowed
    )
    logger.info(
        "router_decision",
        query_preview=query[:50],
        decision=decision,
        has_documents=has_documents,
        web_search_allowed=web_search_allowed,
    )
    return decision
