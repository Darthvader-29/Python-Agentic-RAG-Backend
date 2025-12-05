"""
Generation module: Final answer creation based on router decision.
Uses Gemini with context-appropriate prompts.
"""
from typing import Literal
from google.generativeai import GenerativeModel
import os
from dotenv import load_dotenv
from components.retrieval import format_context

load_dotenv()

# Gemini model for generation (free tier)
gemini_model = GenerativeModel(
    model_name="gemini-2.5-flash",
    generation_config={
        "temperature": 0.3,  # Balanced creativity
        "max_output_tokens": 2048,
        "top_p": 0.9,
    }
)

async def generate_final_response(
    query: str, 
    context: list[str], 
    decision: Literal["RAG", "WEB", "DIRECT"]
) -> str:
    """
    Generate final answer based on routing decision.
    
    Args:
        query: Original user question
        context: Retrieved documents/web snippets
        decision: RAG, WEB, or DIRECT
    
    Returns:
        Final answer string
    """
    formatted_context = format_context(context)
    
    if decision == "RAG":
        response = await _generate_rag_response(query, formatted_context)
    elif decision == "WEB":
        response = await _generate_web_response(query, formatted_context)
    else:  # DIRECT
        response = await _generate_direct_response(query)
    
    print(f"[Generation] {decision}: Generated {len(response)} chars")
    return response

async def _generate_rag_response(query: str, context: str) -> str:
    """RAG-specific prompt: Grounded in user documents."""
    prompt = f"""\
            You are a helpful assistant answering questions about PRIVATE DOCUMENTS.

            CONTEXT FROM USER DOCUMENTS:
            {context}

            USER QUESTION: {query}

            Answer ONLY based on the document context above. If the answer isn't in the context, say "I don't have that information in the uploaded documents."
            Format naturally, cite section/chunk numbers when possible."""
    
    response = await gemini_model.generate_content_async(prompt)
    return response.text.strip()

async def _generate_web_response(query: str, context: str) -> str:
    """WEB-specific prompt: Current info from search results."""
    prompt = f"""\
        You are a helpful assistant using WEB SEARCH RESULTS.

        WEB SEARCH RESULTS:
        {context}

        USER QUESTION: {query}

        Answer using ONLY the web results above. Summarize key facts. If results don't answer the question, say "Web results don't contain this information."
        Be concise and factual."""
    
    response = await gemini_model.generate_content_async(prompt)
    return response.text.strip()

async def _generate_direct_response(query: str) -> str:
    """DIRECT: Pure chat, no context needed."""
    prompt = f"""\
        You are a helpful AI assistant.

        USER: {query}

        Answer naturally and helpfully."""
    
    response = await gemini_model.generate_content_async(prompt)
    return response.text.strip()
