from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from components.router import route_query


def _fake_db() -> AsyncMock:
    """Mock AsyncSession for route_query tests."""
    return AsyncMock()


@pytest.mark.asyncio
@patch("components.router.repo.session_has_documents", new_callable=AsyncMock)
@patch("components.router.gemini_model.generate_content_async")
async def test_route_query_rag(mock_gemini, mock_has_docs):
    mock_has_docs.return_value = False
    mock_gemini.return_value = MagicMock(text="RAG")
    decision = await route_query(
        "Summarize the uploaded PDF for me.", "session123", False, _fake_db()
    )
    assert decision == "RAG"


@pytest.mark.asyncio
@patch("components.router.repo.session_has_documents", new_callable=AsyncMock)
@patch("components.router.gemini_model.generate_content_async")
async def test_route_query_direct(mock_gemini, mock_has_docs):
    mock_has_docs.return_value = False
    mock_gemini.return_value = MagicMock(text="DIRECT")
    decision = await route_query(
        "Write a python script to scrape google.", "session123", False, _fake_db()
    )
    assert decision == "DIRECT"


@pytest.mark.asyncio
@patch("components.router.repo.session_has_documents", new_callable=AsyncMock)
@patch("components.router.gemini_model.generate_content_async")
async def test_route_query_web(mock_gemini, mock_has_docs):
    mock_has_docs.return_value = False
    mock_gemini.return_value = MagicMock(text="WEB")
    decision = await route_query(
        "Who is the president of France in 2025?", "session123", True, _fake_db()
    )
    assert decision == "WEB"


@pytest.mark.asyncio
@patch("components.router.repo.session_has_documents", new_callable=AsyncMock)
@patch("components.router.gemini_model.generate_content_async")
async def test_route_query_uses_db_for_has_docs(mock_gemini, mock_has_docs):
    """route_query reads has_documents from the repository, not from Pinecone."""
    mock_has_docs.return_value = True
    mock_gemini.return_value = MagicMock(text="RAG")
    db = _fake_db()
    await route_query("Tell me about my contract.", "sess1", False, db)
    mock_has_docs.assert_awaited_once_with(db, "sess1")
