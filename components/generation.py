import os
import google.generativeai as genai
from dotenv import load_dotenv
from components.router import RouteDecision
from components.retrieval import RetrievalResult

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Load prompts from .env
RAG_PROMPT_TEMPLATE = os.getenv("RAG_PROMPT", "")
WEB_PROMPT_TEMPLATE = os.getenv("WEB_PROMPT", "")
DIRECT_PROMPT_TEMPLATE = os.getenv("DIRECT_PROMPT", "")

def _call_gemini(prompt: str) -> str:
    """Private helper to call the Gemini API and return the text."""
    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"[Generation Error] Gemini API call failed: {e}")
        return "Sorry, I encountered an error while generating the response."

def _generate_rag_answer(query: str, context: str) -> str:
    """Generates an answer based on RAG context."""
    prompt = RAG_PROMPT_TEMPLATE.format(context=context, query=query)
    return _call_gemini(prompt)

def _generate_web_answer(query: str, context: str) -> str:
    """Generates an answer based on web context."""
    prompt = WEB_PROMPT_TEMPLATE.format(context=context, query=query)
    return _call_gemini(prompt)

def _generate_direct_answer(query: str) -> str:
    """Generates a direct answer from the model's knowledge."""
    prompt = DIRECT_PROMPT_TEMPLATE.format(query=query)
    return _call_gemini(prompt)

def generate_final_response(route_decision: RouteDecision, query: str, retrieval_result: RetrievalResult) -> str:
    """
    Main orchestrator function for generation.
    It calls the appropriate specialized function based on the route.
    """
    decision = route_decision.decision
    
    if decision == "DIRECT":
        print("[Generator] Calling DIRECT answer function.")
        return _generate_direct_answer(query)
        
    # For RAG or WEB, we need context.
    if not retrieval_result.contexts:
        print("[Generator] Route was RAG/WEB but no context found. Falling back to DIRECT.")
        return _generate_direct_answer(query)
        
    # Combine all context chunks into a single string
    combined_context = "\n\n---\n\n".join([item['text'] for item in retrieval_result.contexts])

    if decision == "RAG":
        print("[Generator] Calling RAG answer function.")
        return _generate_rag_answer(query, combined_context)
    
    elif decision == "WEB":
        print("[Generator] Calling WEB answer function.")
        return _generate_web_answer(query, combined_context)
        
    # Fallback case
    return _generate_direct_answer(query)
