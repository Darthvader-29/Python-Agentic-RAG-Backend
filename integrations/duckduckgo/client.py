"""
DuckDuckGo search client for WEB routing.
Returns snippets for Gemini context.
"""
from typing import List, Dict
from duckduckgo_search import DDGS

def search_web(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """Search DuckDuckGo and return title + snippet."""
    try:
        with DDGS() as ddgs:
            results = ddgs.text(query, max_results=max_results)
            return [{"title": r["title"], "snippet": r["body"]} for r in results]
    except Exception as e:
        print(f"[DuckDuckGo Error] {e}")
        return []
