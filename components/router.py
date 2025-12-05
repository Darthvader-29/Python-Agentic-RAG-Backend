"""
Router module: Uses Gemini to classify queries into RAG, WEB, or DIRECT.
Considers user web_search_allowed toggle.
"""
from typing import Literal
import google.generativeai as genai
from config import GOOGLE_API_KEY

genai.configure(api_key=GOOGLE_API_KEY)
# Free tier Gemini model
gemini_model = genai.GenerativeModel(
    model_name="gemini-2.5-flash",
    generation_config={
        "temperature": 0.1,  # Low temp for consistent routing
        "max_output_tokens": 20,
    }
)

async def route_query(
    query: str, 
    session_id: str, 
    web_search_allowed: bool
) -> Literal["RAG", "WEB", "DIRECT"]:
    """
    Route query to RAG, WEB, or DIRECT using Gemini.
    
    Args:
        query: User question
        session_id: For context about uploaded docs
        web_search_allowed: User toggle
    
    Returns:
        "RAG", "WEB", or "DIRECT"
    """
    # Check if user has uploaded documents for this session
    has_documents = await has_session_documents(session_id)
    
    # Build routing prompt
    prompt = _build_routing_prompt(query, has_documents, web_search_allowed)
    
    try:
        response = await gemini_model.generate_content_async(prompt)
        decision = response.text.strip().upper()
        
        # Normalize response
        decision = _normalize_decision(decision)
        
        print(f"[Router] Query: '{query[:50]}...' -> {decision} "
              f"(docs: {has_documents}, web: {web_search_allowed})")
        
        return decision
        
    except Exception as e:
        print(f"[Router] Gemini error, defaulting to RAG: {e}")
        return "RAG" if has_documents else "DIRECT"

def _build_routing_prompt(query: str, has_documents: bool, web_allowed: bool) -> str:
    """Dynamic prompt based on session state."""
    doc_status = "YES (user uploaded documents)" if has_documents else "NO"
    web_status = "ALLOWED" if web_allowed else "DISABLED"
    
    return f"""\
            You are a RAG system router. Classify this query into exactly ONE category:

            - RAG: Needs info from user's PRIVATE DOCUMENTS (legal, contracts, reports, etc.)
            - WEB: Needs CURRENT EVENTS, FACTS, or GENERAL KNOWLEDGE (news, stats, definitions)
            - DIRECT: Simple chat, opinions, code help, greetings, or DOCUMENTS NOT NEEDED

            Query: "{query}"
            Documents available: {doc_status}
            Web search: {web_status}

            Respond with ONLY: RAG, WEB, or DIRECT
            Example:
            Query: "What does section 3.2 say about termination?"
            -> RAG

            Query: "What's the latest on US elections?"
            -> WEB

            Query: "Write a Python function to sort arrays"
            -> DIRECT
            """

def _normalize_decision(decision: str) -> Literal["RAG", "WEB", "DIRECT"]:
    """Handle Gemini variations."""
    decision = decision.strip().upper()
    if decision.startswith("RAG"):
        return "RAG"
    elif decision.startswith("WEB"):
        return "WEB"
    else:
        return "DIRECT"

async def has_session_documents(session_id: str) -> bool:
    """
    Quick check: Does this session have vectors in Pinecone?
    (Could be cached or use Pinecone stats)
    """
    from database.db_manager import get_index
    
    try:
        index = get_index()
        # Query for one vector in the session namespace
        # This is faster than describe_index_stats if the index is large
        response = index.query(
            vector=[0.0] * 384,  # Dummy vector, content doesn't matter
            top_k=1,
            filter={"session_id": {"$eq": session_id}}
        )
        return len(response.matches) > 0
        
    except:
        return False
