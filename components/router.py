"""Router module: Uses Gemini to classify queries into RAG, WEB, or DIRECT.

Gemini client stays module-level global — per-user BYOK keys and provider
abstraction are deferred to Phase 4. Only the Pinecone session-check is injected.
"""

from typing import TYPE_CHECKING, Literal

import google.generativeai as genai
import structlog
from google.api_core.exceptions import GoogleAPIError

from config import settings
from exceptions import AppException

if TYPE_CHECKING:
    from database.db_manager import PineconeClient

genai.configure(api_key=settings.GOOGLE_API_KEY)
# Free tier Gemini model (global; BYOK + provider abstraction deferred to Phase 4)
gemini_model = genai.GenerativeModel(
    model_name="gemini-2.5-flash",
    generation_config={
        "temperature": 0.1,
        "max_output_tokens": 20,
    },
)

logger = structlog.get_logger(__name__)


async def route_query(
    query: str,
    session_id: str,
    web_search_allowed: bool,
    pinecone: "PineconeClient",
) -> Literal["RAG", "WEB", "DIRECT"]:
    """Route query to RAG, WEB, or DIRECT using Gemini."""
    has_documents = await pinecone.has_session_documents(session_id)

    prompt = _build_routing_prompt(query, has_documents, web_search_allowed)

    try:
        response = await gemini_model.generate_content_async(prompt)
        decision = _normalize_decision(response.text.strip().upper())

        logger.info(
            "router_decision",
            query_preview=query[:50],
            decision=decision,
            has_documents=has_documents,
            web_search_allowed=web_search_allowed,
        )

        return decision

    except GoogleAPIError as e:
        code = getattr(e, "code", None)
        status_name = getattr(code, "name", "") if code else ""
        http_status = getattr(code, "value", 500) if code else 500

        if http_status == 403:
            msg = (
                "The AI service is not authorized. Please check the Gemini API key and permissions."
            )
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

        logger.error(
            "gemini_api_error",
            component="router",
            http_status=http_status,
            status_name=status_name,
            message=msg,
        )
        raise AppException(status_code=http_status, detail=msg) from e

    except Exception:
        logger.error("router_gemini_error", exc_info=True)
        return "RAG" if has_documents else "DIRECT"


def _build_routing_prompt(query: str, has_documents: bool, web_allowed: bool) -> str:
    doc_status = "YES (user uploaded documents)" if has_documents else "NO"
    web_status = "ALLOWED" if web_allowed else "DISABLED"

    return f"""
            You are a routing classifier for a Retrieval-Augmented Generation system.

            Classify the user query into EXACTLY ONE of these categories:

            - RAG: Requires information that is likely to be found ONLY in the user's PRIVATE DOCUMENTS
            (contracts, policies, internal reports, PDFs, local notes).
            - WEB: Asks about GENERAL KNOWLEDGE, PUBLIC FACTS, DEFINITIONS, NEWS, PEOPLE, COMPANIES, OR TECHNOLOGY.
            - DIRECT: Simple chat, opinions, greetings, or coding questions that do NOT require either
            documents or the web (you can answer from general model knowledge alone).

            IMPORTANT:
            - If the question is about a programming language, framework, library, famous person, company,
            or public concept, choose WEB (if web is allowed), otherwise DIRECT.
            - ONLY choose RAG when the question clearly refers to "my document", "the PDF", "the contract",
            "this report", or similar private content.
            - NEVER choose RAG for generic trivia or public facts.

            Query: "{query}"
            Documents available: {doc_status}
            Web search: {web_status}

            Respond with ONLY one word: RAG, WEB, or DIRECT.
            """


def _normalize_decision(decision: str) -> Literal["RAG", "WEB", "DIRECT"]:
    """Handle Gemini response variations."""
    decision = decision.strip().upper()
    if decision.startswith("RAG"):
        return "RAG"
    elif decision.startswith("WEB"):
        return "WEB"
    else:
        return "DIRECT"
