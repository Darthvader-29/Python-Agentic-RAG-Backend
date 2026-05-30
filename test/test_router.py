from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from components.router import route_query
from database.db_manager import PineconeClient


def _fake_pinecone(has_docs: bool = False) -> AsyncMock:
    """AsyncMock PineconeClient with has_session_documents pre-configured."""
    pc = AsyncMock(spec=PineconeClient)
    pc.has_session_documents.return_value = has_docs
    return pc


@pytest.mark.asyncio
@patch("components.router.gemini_model.generate_content_async")
async def test_route_query_rag(mock_gemini):
    mock_gemini.return_value = MagicMock(text="RAG")
    decision = await route_query(
        "Summarize the uploaded PDF for me.", "session123", False, _fake_pinecone()
    )
    assert decision == "RAG"


@pytest.mark.asyncio
@patch("components.router.gemini_model.generate_content_async")
async def test_route_query_direct(mock_gemini):
    mock_gemini.return_value = MagicMock(text="DIRECT")
    decision = await route_query(
        "Write a python script to scrape google.", "session123", False, _fake_pinecone()
    )
    assert decision == "DIRECT"


@pytest.mark.asyncio
@patch("components.router.gemini_model.generate_content_async")
async def test_route_query_web(mock_gemini):
    mock_gemini.return_value = MagicMock(text="WEB")
    decision = await route_query(
        "Who is the president of France in 2025?", "session123", True, _fake_pinecone()
    )
    assert decision == "WEB"


@pytest.mark.asyncio
async def test_has_session_documents_true():
    """PineconeClient.has_session_documents returns True when Pinecone has matches."""
    client = PineconeClient(api_key="test", index_name="test-index")
    mock_index = MagicMock()
    mock_index.query.return_value = MagicMock(matches=[1])
    client._index = mock_index

    result = await client.has_session_documents("session_with_docs")

    assert result is True
    mock_index.query.assert_called_once_with(
        vector=[0.0] * 384, top_k=1, filter={"session_id": {"$eq": "session_with_docs"}}
    )


@pytest.mark.asyncio
async def test_has_session_documents_false():
    """PineconeClient.has_session_documents returns False when no matches."""
    client = PineconeClient(api_key="test", index_name="test-index")
    mock_index = MagicMock()
    mock_index.query.return_value = MagicMock(matches=[])
    client._index = mock_index

    result = await client.has_session_documents("session_without_docs")

    assert result is False
    mock_index.query.assert_called_once_with(
        vector=[0.0] * 384, top_k=1, filter={"session_id": {"$eq": "session_without_docs"}}
    )
