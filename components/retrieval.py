import os
from typing import List, Dict, Any
from dotenv import load_dotenv
import google.generativeai as genai
from duckduckgo_search import DDGS

from database.db_manager import search_vectors
from components.router import RouteDecision

load_dotenv()

# Configure Gemini for query embeddings
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIM = 768  # MUST match Pinecone index dimension

# DuckDuckGo Search client
ddg = DDGS()


class RetrievalResult:
    def __init__(self):
        self.contexts: List[Dict[str, Any]] = []  # each: {"text": ..., "score": ...}
        self.sources: List[str] = []              # e.g. filenames or web titles


def retrieve_context(
    route_decision: RouteDecision,
    user_query: str,
    web_search_allowed: bool = True
) -> RetrievalResult:
    """
    Main retrieval function based on router decision and user web toggle.
    """
    result = RetrievalResult()
    decision = route_decision.decision

    if decision == "DIRECT":
        print("[Retrieval] DIRECT route: No context needed.")
        return result

    if decision == "RAG":
        print("[Retrieval] RAG route: Searching Pinecone...")
        return _retrieve_from_pinecone(user_query)

    if decision == "WEB":
        if web_search_allowed:
            print("[Retrieval] WEB route: Searching DuckDuckGo...")
            return _retrieve_from_web(user_query)
        else:
            print("[Retrieval] WEB route blocked by user preference. Returning empty context.")
            return result

    # Fallback
    print("[Retrieval] Unknown route. Returning empty context.")
    return result


def _embed_query_with_gemini(query: str) -> List[float]:
    """Create a Gemini embedding vector for the query."""
    try:
        res = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=query,
            task_type="retrieval_query",
            output_dimensionality=EMBEDDING_DIM,
        )
        # For single input, res.embedding is a single vector
        return res.embedding
    except Exception as e:
        print(f"[Retrieval] Query embedding failed: {e}")
        raise


def _retrieve_from_pinecone(query: str) -> RetrievalResult:
    """Search Pinecone for relevant document chunks."""
    result = RetrievalResult()

    try:
        query_vector = _embed_query_with_gemini(query)
        pinecone_results = search_vectors(query_vector, top_k=5)

        for doc in pinecone_results:
            result.contexts.append(
                {
                    "text": doc["text"],
                    "score": doc["score"],
                }
            )
            # doc["source"] was set in db_manager metadata
            result.sources.append(doc.get("source") or doc.get("filename", "unknown"))

    except Exception as e:
        print(f"[Retrieval] Error during Pinecone retrieval: {e}")

    return result


def _retrieve_from_web(query: str) -> RetrievalResult:
    """Search DuckDuckGo for web results."""
    result = RetrievalResult()

    try:
        web_results = ddg.text(query, max_results=3)
        for res in web_results:
            body = res.get("body") or res.get("snippet") or ""
            title = res.get("title") or res.get("href") or "web-result"

            result.contexts.append(
                {
                    "text": body,
                    "score": 1.0,  # no similarity score, treat as 1.0 baseline
                }
            )
            result.sources.append(title)

    except Exception as e:
        print(f"[Retrieval] Web search error: {e}")

    return result
