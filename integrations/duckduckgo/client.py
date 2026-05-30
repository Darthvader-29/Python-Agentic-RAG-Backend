"""Async DuckDuckGo search client for WEB routing.

Sync DDGS call runs via asyncio.to_thread; tenacity retries on transient errors.
"""

import asyncio

import structlog
from duckduckgo_search import DDGS
from tenacity import retry, stop_after_attempt, wait_exponential

logger = structlog.get_logger(__name__)

_RETRY = dict(
    stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.5, max=8), reraise=True
)


class DuckDuckGoClient:
    @retry(**_RETRY)
    def _search_sync(self, query: str, max_results: int) -> list[dict[str, str]]:
        try:
            with DDGS() as ddgs:
                results = ddgs.text(query, max_results=max_results)
                return [{"title": r["title"], "snippet": r["body"]} for r in results]
        except Exception:
            logger.error("duckduckgo_search_error", exc_info=True)
            return []

    async def search_web(self, query: str, max_results: int = 5) -> list[dict[str, str]]:
        return await asyncio.to_thread(self._search_sync, query, max_results)
