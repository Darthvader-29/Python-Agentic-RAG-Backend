"""Tests for llm/dependencies.py — get_llm_provider DI."""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from auth.crypto import encrypt_key
from database.models import User, UserLLMKey
from exceptions import LLMAuthError
from llm.dependencies import get_llm_provider


def _fake_user() -> User:
    u = MagicMock(spec=User)
    u.id = uuid.uuid4()
    return u


def _fake_key_row(
    user_id: uuid.UUID, provider: str = "openai", plaintext: str = "sk-test"
) -> UserLLMKey:
    row = MagicMock(spec=UserLLMKey)
    row.provider = provider
    row.ciphertext = encrypt_key(plaintext)
    return row


@pytest.mark.asyncio
@patch("llm.dependencies.get_user_llm_key", new_callable=AsyncMock)
@patch("llm.openai.AsyncOpenAI")
async def test_provider_built_from_user_key(mock_openai_cls, mock_get_key):
    user = _fake_user()
    mock_get_key.return_value = _fake_key_row(user.id, provider="openai")
    db = AsyncMock()

    provider = await get_llm_provider(user=user, db=db)

    assert provider.__class__.__name__ == "OpenAIProvider"
    assert "sk-test" not in repr(provider)


@pytest.mark.asyncio
@patch("llm.dependencies.get_user_llm_key", new_callable=AsyncMock)
async def test_no_key_no_fallback_raises(mock_get_key, monkeypatch):
    mock_get_key.return_value = None
    monkeypatch.setenv("LLM_FALLBACK_API_KEY", "")

    user = _fake_user()
    db = AsyncMock()

    with pytest.raises(LLMAuthError):
        await get_llm_provider(user=user, db=db)


@pytest.mark.asyncio
@patch("llm.dependencies.get_user_llm_key", new_callable=AsyncMock)
@patch("llm.gemini.genai.Client")
async def test_server_fallback_used_when_no_user_key(mock_gemini_cls, mock_get_key, monkeypatch):
    mock_get_key.return_value = None
    monkeypatch.setenv("LLM_FALLBACK_API_KEY", "sk-fallback-server-key")

    # Reload settings so the new env var is picked up
    import importlib

    import config

    importlib.reload(config)

    user = _fake_user()
    db = AsyncMock()

    with patch("llm.dependencies.settings") as mock_settings:
        mock_settings.DEFAULT_LLM_PROVIDER = "gemini"
        mock_settings.DEFAULT_LLM_MODEL = "gemini-2.5-flash"
        mock_settings.LLM_FALLBACK_API_KEY = MagicMock()
        mock_settings.LLM_FALLBACK_API_KEY.get_secret_value.return_value = "sk-fallback"

        provider = await get_llm_provider(user=user, db=db)

    assert provider.__class__.__name__ == "GeminiProvider"


@pytest.mark.asyncio
@patch("llm.dependencies.get_user_llm_key", new_callable=AsyncMock)
@patch("llm.gemini.genai.Client")
async def test_provider_uses_row_provider_field(mock_gemini_cls, mock_get_key):
    user = _fake_user()
    mock_get_key.return_value = _fake_key_row(user.id, provider="gemini", plaintext="sk-gemini")
    db = AsyncMock()

    provider = await get_llm_provider(user=user, db=db)

    assert provider.__class__.__name__ == "GeminiProvider"
