"""Tests for BYOK LLM key CRUD + ciphertext-at-rest gate (Phase 3).

Headline CI gate: asserts that the stored value is NOT the plaintext key
and that no plaintext key appears in captured log output.
"""

import io
import logging

import pytest
import structlog
from fastapi import HTTPException
from sqlalchemy import select

from auth.crypto import decrypt_key
from auth.keys_router import add_key, delete_key, rotate_key
from auth.schemas import KeyIn
from auth.security import hash_password
from database.models import UserLLMKey
from database.repository import LLMKeyRepository, UserRepository

pytestmark = pytest.mark.asyncio

_PLAINTEXT_KEY = "sk-supersecret-api-key-do-not-log"


async def _make_user(db, email: str, username: str):
    return await UserRepository(db).create(
        email=email,
        username=username,
        hashed_password=hash_password("pass"),
    )


async def test_add_key_stores_ciphertext_not_plaintext(db_session):
    user = await _make_user(db_session, "keytest@test.com", "keytest")

    await add_key(
        KeyIn(provider="gemini", api_key=_PLAINTEXT_KEY),
        current_user=user,
        db=db_session,
    )

    # Query the raw DB row — must NOT contain the plaintext
    result = await db_session.execute(
        select(UserLLMKey).where(UserLLMKey.user_id == user.id, UserLLMKey.provider == "gemini")
    )
    row = result.scalar_one()
    assert row.ciphertext != _PLAINTEXT_KEY
    assert _PLAINTEXT_KEY not in row.ciphertext
    # But must decrypt to the original key
    assert decrypt_key(row.ciphertext) == _PLAINTEXT_KEY


async def test_rotate_key_changes_ciphertext(db_session):
    user = await _make_user(db_session, "rotate@test.com", "rotateuser")

    await add_key(KeyIn(provider="openai", api_key="first-key"), current_user=user, db=db_session)

    result1 = await db_session.execute(
        select(UserLLMKey).where(UserLLMKey.user_id == user.id, UserLLMKey.provider == "openai")
    )
    ct1 = result1.scalar_one().ciphertext

    await rotate_key(
        "openai",
        KeyIn(provider="openai", api_key="second-key"),
        current_user=user,
        db=db_session,
    )

    result2 = await db_session.execute(
        select(UserLLMKey).where(UserLLMKey.user_id == user.id, UserLLMKey.provider == "openai")
    )
    row2 = result2.scalar_one()
    assert row2.ciphertext != ct1
    assert decrypt_key(row2.ciphertext) == "second-key"


async def test_delete_key_removes_row(db_session):
    user = await _make_user(db_session, "del@test.com", "deluser")

    await add_key(
        KeyIn(provider="anthropic", api_key="some-key"),
        current_user=user,
        db=db_session,
    )

    await delete_key("anthropic", current_user=user, db=db_session)

    remaining = await LLMKeyRepository(db_session).get(user_id=user.id, provider="anthropic")
    assert remaining is None


async def test_delete_nonexistent_key_raises_404(db_session):
    user = await _make_user(db_session, "nd@test.com", "nduser")
    with pytest.raises(HTTPException) as exc_info:
        await delete_key("gemini", current_user=user, db=db_session)
    assert exc_info.value.status_code == 404


async def test_user_scoping_different_users_independent(db_session):
    """Two users can each have their own keys for the same provider."""
    user_a = await _make_user(db_session, "ka@test.com", "kauser")
    user_b = await _make_user(db_session, "kb@test.com", "kbuser")

    await add_key(KeyIn(provider="gemini", api_key="key-a"), current_user=user_a, db=db_session)
    await add_key(KeyIn(provider="gemini", api_key="key-b"), current_user=user_b, db=db_session)

    row_a = await LLMKeyRepository(db_session).get(user_id=user_a.id, provider="gemini")
    row_b = await LLMKeyRepository(db_session).get(user_id=user_b.id, provider="gemini")

    assert row_a is not None and row_b is not None
    assert row_a.ciphertext != row_b.ciphertext
    assert decrypt_key(row_a.ciphertext) == "key-a"
    assert decrypt_key(row_b.ciphertext) == "key-b"


async def test_plaintext_key_never_appears_in_logs(db_session, capfd):
    """Log-scan gate: structlog output must never contain the plaintext API key."""
    user = await _make_user(db_session, "logscan@test.com", "logscanuser")

    # Capture stdout (structlog in dev mode writes there)
    await add_key(
        KeyIn(provider="gemini", api_key=_PLAINTEXT_KEY),
        current_user=user,
        db=db_session,
    )
    await rotate_key(
        "gemini",
        KeyIn(provider="gemini", api_key=_PLAINTEXT_KEY + "-rotated"),
        current_user=user,
        db=db_session,
    )
    await delete_key("gemini", current_user=user, db=db_session)

    captured = capfd.readouterr()
    all_output = captured.out + captured.err
    assert _PLAINTEXT_KEY not in all_output, "Plaintext API key leaked into log output"
    assert (_PLAINTEXT_KEY + "-rotated") not in all_output
