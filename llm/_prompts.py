"""Shared prompt builders used by all LLM adapters.

Prompts are preserved verbatim from the original router/generation modules so
routing vocabulary and generation style remain unchanged after Phase 4.
"""

from __future__ import annotations

from typing import Literal

Route = Literal["RAG", "WEB", "DIRECT"]


# ── Routing ───────────────────────────────────────────────────────────────────


def routing_prompt(query: str, has_documents: bool, web_allowed: bool) -> str:
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


def normalize_decision(text: str) -> Route:
    """Normalize a provider's routing response to one of RAG/WEB/DIRECT."""
    t = text.strip().upper()
    if t.startswith("RAG"):
        return "RAG"
    if t.startswith("WEB"):
        return "WEB"
    return "DIRECT"


# ── Generation ────────────────────────────────────────────────────────────────


def generation_prompt(decision: str, query: str, context: str) -> str:
    """Single-string generation prompt for Gemini and Anthropic."""
    d = decision.upper()
    if "RAG" in d:
        return (
            f"You are a helpful assistant answering questions about PRIVATE DOCUMENTS.\n\n"
            f"CONTEXT FROM USER DOCUMENTS:\n{context}\n\n"
            f"USER QUESTION: {query}\n\n"
            "Answer ONLY based on the document context above. "
            "If the answer isn't in the context, say "
            '"I don\'t have that information in the uploaded documents."\n'
            "Format naturally, cite section/chunk numbers when possible."
        )
    if "WEB" in d:
        return (
            f"You are a helpful assistant using WEB SEARCH RESULTS.\n\n"
            f"WEB SEARCH RESULTS:\n{context}\n\n"
            f"USER QUESTION: {query}\n\n"
            "Answer using ONLY the web results above. Summarize key facts. "
            "If results don't answer the question, say "
            '"Web results don\'t contain this information."\n'
            "Be concise and factual."
        )
    # DIRECT (or DIRECT+WEB/DIRECT+RAG with empty context)
    return f"You are a helpful AI assistant.\n\nUSER: {query}\n\nAnswer naturally and helpfully."


def generation_system_user(decision: str, query: str, context: str) -> tuple[str, str]:
    """Split generation prompt into (system, user) for OpenAI chat format."""
    d = decision.upper()
    if "RAG" in d:
        system = "You are a helpful assistant answering questions about PRIVATE DOCUMENTS."
        user = (
            f"CONTEXT FROM USER DOCUMENTS:\n{context}\n\n"
            f"USER QUESTION: {query}\n\n"
            "Answer ONLY based on the document context above. "
            "If the answer isn't in the context, say "
            '"I don\'t have that information in the uploaded documents." '
            "Format naturally, cite section/chunk numbers when possible."
        )
    elif "WEB" in d:
        system = "You are a helpful assistant using WEB SEARCH RESULTS."
        user = (
            f"WEB SEARCH RESULTS:\n{context}\n\n"
            f"USER QUESTION: {query}\n\n"
            "Answer using ONLY the web results above. Summarize key facts. "
            "If results don't answer the question, say "
            '"Web results don\'t contain this information." Be concise and factual.'
        )
    else:
        system = "You are a helpful AI assistant."
        user = query
    return system, user
