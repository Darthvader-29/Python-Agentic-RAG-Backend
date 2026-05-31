"""Generation module: delegates final answer creation to the injected LLM provider.

Phase 4: the Gemini process-global and GoogleAPIError ladder are removed.
Error mapping lives in the provider adapter; neutral LLM errors bubble to
app_exception_handler in exceptions.py.
"""

import structlog

from components.retrieval import format_context
from llm.base import LLMProvider, Route

logger = structlog.get_logger(__name__)


async def generate_final_response(
    provider: LLMProvider,
    query: str,
    context: list[str],
    decision: Route,
) -> str:
    """Generate the final answer using the injected provider."""
    answer = await provider.generate(query, format_context(context), decision)
    logger.info("generation_complete", decision=decision, response_chars=len(answer))
    return answer
