import os
from typing import List, Dict, Any
from dotenv import load_dotenv
import google.generativeai as genai
from duckduckgo_search import DDGS

from database.db_manager import search_vectors
from components.router import RouteDecision

load_dotenv()

# DuckDuckGo Search (Free web search)
ddg = DDGS()

class RetrievalResult:
    def __init__(self):
        self.contexts: List[Dict[str, Any]] = []
        self.sources: List[str] = []

def retrieve_context(route_decision: RouteDecision, user_query: str, web_search_allowed: bool = True) -> RetrievalResult:
    """
    Main retrieval function based on router decision.
    Respects user web search preference.
    """
    result = RetrievalResult()
    
    if route_decision.decision == "DIRECT":
        print("[Retrieval] DIRECT route: No context needed")
        return result
    
    elif route_decision.decision == "RAG":
        print("[Retrieval] RAG route: Searching Pinecone...")
        return _retrieve_from_pinecone(user_query)
    
    elif route_decision.decision == "WEB":
        if web_search_allowed:
            print("[Retrieval] WEB route: Searching DuckDuckGo...")
            return _retrieve_from_web(user_query)
        else:
            print("[Retrieval] WEB route blocked by user preference. Falling back to empty context.")
            return result  # Return empty result (no web search)
    
    return result

def _retrieve_from_pinecone(query: str) -> RetrievalResult:
    """Search Pinecone for relevant document chunks."""
    result = RetrievalResult()
    
    # 1. Generate embedding for the query
    embedding_result = genai.embed_content(
        model="models/text-embedding-004",
        content=query,
        task_type="retrieval_query"
    )
    query_embedding = embedding_result['embedding']
    
    # 2. Search Pinecone
    pinecone_results = search_vectors(query_embedding, top_k=5)
    
    for doc in pinecone_results:
        result.contexts.append({
            "text": doc["text"],
            "score": doc["score"]
        })
        result.sources.append(doc["source"])
    
    return result

def _retrieve_from_web(query: str) -> RetrievalResult:
    """Search DuckDuckGo for web results."""
    result = RetrievalResult()
    
    try:
        # Get top 3 web results
        web_results = ddg.text(query, max_results=3)
        
        for res in web_results:
            result.contexts.append({
                "text": res["body"],
                "score": 1.0  # Web results don't have similarity scores
            })
            result.sources.append(res["title"])
            
    except Exception as e:
        print(f"[Web Search Error] {e}")
        # Graceful fallback - don't crash the app
    
    return result
