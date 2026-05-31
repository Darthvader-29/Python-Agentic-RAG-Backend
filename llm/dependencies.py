"""Per-request LLM provider dependency.

Kept separate from dependencies.py to avoid a circular import:
  auth/dependencies.py → dependencies.py → (would be circular if we imported auth here)
"""

from __future__ import annotations

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from auth.crypto import decrypt_key
from auth.dependencies import get_current_user
from config import settings
from database.models import User
from database.repository import get_user_llm_key
from dependencies import get_db_session
from exceptions import LLMAuthError
from llm.base import LLMProvider
from llm.factory import build_provider


async def get_llm_provider(
    user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db_session),
) -> LLMProvider:
    """Build a per-request LLM provider from the authenticated user's decrypted BYOK key.

    The decrypted api_key is a local variable only — never logged, never cached,
    never placed on app.state, and gone when the function returns.
    """
    row = await get_user_llm_key(db, user_id=user.id)
    if row is not None:
        api_key = decrypt_key(row.ciphertext)  # plaintext: local only, never persisted
        return build_provider(
            row.provider or settings.DEFAULT_LLM_PROVIDER,
            api_key,
            model=settings.DEFAULT_LLM_MODEL,
        )
    fallback = settings.LLM_FALLBACK_API_KEY.get_secret_value()
    if fallback:
        return build_provider(
            settings.DEFAULT_LLM_PROVIDER,
            fallback,
            model=settings.DEFAULT_LLM_MODEL,
        )
    raise LLMAuthError("No LLM key on file and no server fallback configured.")
