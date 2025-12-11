"""
Generation module: Final answer creation based on router decision.
Uses Gemini with context-appropriate prompts.
"""
from typing import Literal
from google.generativeai import GenerativeModel
from google.api_core.exceptions import GoogleAPIError
from exceptions import AppException
import os
from dotenv import load_dotenv
from components.retrieval import format_context


load_dotenv()

# Gemini model for generation (free tier)
gemini_model = GenerativeModel(
    model_name="gemini-2.5-flash",
    generation_config={
        "temperature": 0.3,  # Balanced creativity
        "max_output_tokens": 65535,
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
    """
    formatted_context = format_context(context)

    try:
        if decision == "RAG":
            response = await _generate_rag_response(query, formatted_context)
        elif decision == "WEB":
            response = await _generate_web_response(query, formatted_context)
        else:  # DIRECT
            response = await _generate_direct_response(query)

        print(f"[Generation] {decision}: Generated {len(response)} chars")
        return response

    except GoogleAPIError as e:
        # Map Gemini HTTP / gRPC codes to clear frontend messages
        code = getattr(e, "code", None)
        status_name = getattr(code, "name", "") if code else ""
        http_status = getattr(code, "value", 500) if code else 500

        if http_status == 403:
            msg = "The AI service is not authorized. Please check the Gemini API key and permissions."
        elif http_status == 404:
            msg = "The AI service could not find a required resource. Please try again later."
        elif http_status == 429:
            msg = "Gemini free-tier daily or per-minute limit has been reached. Please try again later."
        elif http_status == 500:
            msg = "The AI service encountered an internal error. Please retry after some time."
        elif http_status == 503:
            msg = "The AI service is temporarily unavailable. Please retry after some time."
        elif http_status == 504:
            msg = "The AI service timed out while processing this request. Try a shorter question."
        else:
            msg = f"The AI service returned an unexpected error ({status_name}). Please try again."

        print(f"[Generation] Gemini API error {http_status}: {status_name} -> {msg}")
        raise AppException(status_code=http_status, detail=msg) from e

    except Exception as e:
        print(f"[Generation] Unexpected error: {e}")
        raise AppException(status_code=500, detail="free tier Limit Reached for API please try again later.") from e


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
