"""Retrieval module for RAG system.

All external calls go through injected client instances; no module-level singletons.
"""

from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from database.db_manager import PineconeClient
    from integrations.duckduckgo.client import DuckDuckGoClient
    from integrations.huggingface.client import HuggingFaceClient

logger = structlog.get_logger(__name__)


async def retrieve_context(
    query: str,
    decision: str,
    session_id: str,
    web_search_allowed: bool,
    pinecone: "PineconeClient",
    embedder: "HuggingFaceClient",
    web: "DuckDuckGoClient",
) -> list[str]:
    """Retrieve context based on router decision."""
    context: list[str] = []

    if decision == "DIRECT":
        logger.info("retrieval_skip", reason="DIRECT route needs no context")
        return context

    elif decision == "RAG":
        logger.info("retrieval_rag", action="searching Pinecone")
        query_embedding = await embedder.embed_single(query)
        logger.debug("query_embedding", dims=len(query_embedding))

        results = await pinecone.search_vectors(
            query_vector=query_embedding, top_k=5, session_id=session_id
        )
        context = [result["text"] for result in results]
        logger.info("retrieval_rag_complete", chunks=len(context), session_id=session_id)

    elif decision == "WEB":
        if web_search_allowed:
            logger.info("retrieval_web", action="searching DuckDuckGo")
            web_results = await web.search_web(query, max_results=5)
            context = [result["snippet"] for result in web_results]
            logger.info("retrieval_web_complete", snippets=len(context))
        else:
            logger.info("retrieval_web_skipped", reason="web_search_allowed=False")

    return context


def format_context(context: list[str], max_tokens: int = 4000) -> str:
    """Format context for Gemini prompt (token-aware truncation)."""
    if not context:
        return "No relevant context found."

    formatted = "\n\n".join([f"CONTEXT {i + 1}:\n{chunk}" for i, chunk in enumerate(context)])

    max_chars = max_tokens * 3
    if len(formatted) > max_chars:
        formatted = formatted[:max_chars] + "\n\n[Context truncated...]"

    return formatted
