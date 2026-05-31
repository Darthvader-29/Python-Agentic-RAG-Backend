"""Test that /api/chat resolves the LLM provider via DI and uses it."""

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from app import app
from auth.dependencies import get_current_user
from database.models import User
from llm.dependencies import get_llm_provider


class _FakeProvider:
    canned_answer = "fake-provider-answer"

    async def route(self, query, *, has_documents, web_allowed):
        return "DIRECT"

    async def generate(self, query, context, decision):
        return self.canned_answer

    def stream(self, query, context, decision):
        return iter([])


def _fake_user():
    user = MagicMock(spec=User)
    user.id = uuid.uuid4()
    return user


@pytest.mark.asyncio
async def test_chat_uses_injected_provider():
    fake_provider = _FakeProvider()
    fake_user = _fake_user()

    app.dependency_overrides[get_llm_provider] = lambda: fake_provider
    app.dependency_overrides[get_current_user] = lambda: fake_user
    try:
        with (
            patch("app.repo.get_session", new_callable=AsyncMock, return_value=None),
            patch("app.repo.create_session", new_callable=AsyncMock),
            patch("app.repo.session_has_documents", new_callable=AsyncMock, return_value=False),
            patch("app.check_docs_relevant", new_callable=AsyncMock, return_value=(False, False)),
            patch("app.retrieve_context", new_callable=AsyncMock, return_value=[]),
        ):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                resp = await client.post(
                    "/api/chat",
                    json={"message": "hello", "web_search_allowed": False},
                    headers={"Authorization": "Bearer test-token"},
                )
        assert resp.status_code == 200
        assert resp.json()["answer"] == fake_provider.canned_answer
    finally:
        app.dependency_overrides.clear()
