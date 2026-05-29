"""
Retrieval module for RAG system.
Handles RAG, WEB, and DIRECT retrieval decisions.
"""

import structlog

from database.db_manager import search_vectors
from integrations.duckduckgo.client import search_web  # Assuming you have this
from integrations.huggingface.client import embed_single

logger = structlog.get_logger(__name__)


async def retrieve_context(
    query: str, decision: str, session_id: str, web_search_allowed: bool = False
) -> list[str]:
    """
    Retrieve context based on router decision.

    Args:
        query: User query
        decision: "RAG", "WEB", or "DIRECT" from router
        session_id: For Pinecone filtering
        web_search_allowed: User toggle for web search

    Returns:
        List of context strings for generation
    """
    context: list[str] = []

    if decision == "DIRECT":
        logger.info("retrieval_skip", reason="DIRECT route needs no context")
        return context

    elif decision == "RAG":
        logger.info("retrieval_rag", action="searching Pinecone")
        # Embed query with SAME model used for ingestion
        query_embedding = embed_single(query)
        logger.debug("query_embedding", dims=len(query_embedding))

        # Search with session isolation
        results = search_vectors(query_vector=query_embedding, top_k=5, session_id=session_id)

        context = [result["text"] for result in results]
        logger.info("retrieval_rag_complete", chunks=len(context), session_id=session_id)

    elif decision == "WEB":
        if web_search_allowed:
            logger.info("retrieval_web", action="searching DuckDuckGo")
            web_results = search_web(query, max_results=5)
            context = [result["snippet"] for result in web_results]
            logger.info("retrieval_web_complete", snippets=len(context))
        else:
            logger.info("retrieval_web_skipped", reason="web_search_allowed=False")

    return context


def format_context(context: list[str], max_tokens: int = 4000) -> str:
    """
    Format context for Gemini prompt (token-aware truncation).
    """
    if not context:
        return "No relevant context found."

    formatted = "\n\n".join([f"CONTEXT {i + 1}:\n{chunk}" for i, chunk in enumerate(context)])

    # Rough token truncation (4 chars ≈ 1 token)
    max_chars = max_tokens * 3
    if len(formatted) > max_chars:
        formatted = formatted[:max_chars] + "\n\n[Context truncated...]"

    return formatted
