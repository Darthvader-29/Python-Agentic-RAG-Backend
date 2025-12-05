"""
Retrieval module for RAG system.
Handles RAG, WEB, and DIRECT retrieval decisions.
"""
import os
from typing import List
from database.db_manager import search_vectors
from integrations.huggingface.client import embed_single
from integrations.duckduckgo.client import search_web  # Assuming you have this

async def retrieve_context(
    query: str, 
    decision: str, 
    session_id: str, 
    web_search_allowed: bool = False
) -> List[str]:
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
    context = []
    
    if decision == "DIRECT":
        print("[Retrieval] DIRECT: No context needed")
        return context
    
    elif decision == "RAG":
        print("[Retrieval] RAG: Searching Pinecone...")
        # Embed query with SAME model used for ingestion
        query_embedding = embed_single(query)
        print(f"Query embedding shape: {len(query_embedding)} dims")
        
        # Search with session isolation
        results = search_vectors(
            query_vector=query_embedding,
            top_k=5,
            session_id=session_id
        )
        
        context = [result["text"] for result in results]
        print(f"[RAG] Retrieved {len(context)} chunks from session {session_id}")
    
    elif decision == "WEB":
        if web_search_allowed:
            print("[Retrieval] WEB: Searching DuckDuckGo...")
            web_results = search_web(query, max_results=5)
            context = [result["snippet"] for result in web_results]
            print(f"[WEB] Retrieved {len(context)} web snippets")
        else:
            print("[Retrieval] WEB requested but web_search_allowed=False")
    
    return context

def format_context(context: List[str], max_tokens: int = 4000) -> str:
    """
    Format context for Gemini prompt (token-aware truncation).
    """
    if not context:
        return "No relevant context found."
    
    formatted = "\n\n".join([f"CONTEXT {i+1}:\n{chunk}" for i, chunk in enumerate(context)])
    
    # Rough token truncation (4 chars ≈ 1 token)
    max_chars = max_tokens * 3
    if len(formatted) > max_chars:
        formatted = formatted[:max_chars] + "\n\n[Context truncated...]"
    
    return formatted
