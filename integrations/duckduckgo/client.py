"""
DuckDuckGo search client for WEB routing.
Returns snippets for Gemini context.
"""

import structlog
from duckduckgo_search import DDGS

logger = structlog.get_logger(__name__)


def search_web(query: str, max_results: int = 5) -> list[dict[str, str]]:
    """Search DuckDuckGo and return title + snippet."""
    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=max_results)
            return [{"title": r["title"], "snippet": r["body"]} for r in results]
    except Exception:
        logger.error("duckduckgo_search_error", exc_info=True)
        return []
