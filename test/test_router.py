import pytest
from unittest.mock import MagicMock, patch
from components.router import route_query, has_session_documents

@pytest.mark.asyncio
@patch('components.router.gemini_model.generate_content_async')
async def test_route_query_rag(mock_generate_content_async):
    mock_response = MagicMock()
    mock_response.text = "RAG"
    mock_generate_content_async.return_value = mock_response
    decision = await route_query("Summarize the uploaded PDF for me.", "session123", False)
    assert decision == "RAG"

@pytest.mark.asyncio
@patch('components.router.gemini_model.generate_content_async')
async def test_route_query_direct(mock_generate_content_async):
    mock_response = MagicMock()
    mock_response.text = "DIRECT"
    mock_generate_content_async.return_value = mock_response
    decision = await route_query("Write a python script to scrape google.", "session123", False)
    assert decision == "DIRECT"

@pytest.mark.asyncio
@patch('components.router.gemini_model.generate_content_async')
async def test_route_query_web(mock_generate_content_async):
    mock_response = MagicMock()
    mock_response.text = "WEB"
    mock_generate_content_async.return_value = mock_response
    decision = await route_query("Who is the president of France in 2025?", "session123", True)
    assert decision == "WEB"

@pytest.mark.asyncio
@patch('database.db_manager.get_index')
async def test_has_session_documents_true(mock_get_index):
    mock_index = MagicMock()
    mock_query_response = MagicMock()
    mock_query_response.matches = [1]  # Simulate at least one match
    mock_index.query.return_value = mock_query_response
    mock_get_index.return_value = mock_index

    result = await has_session_documents("session_with_docs")
    assert result is True
    mock_index.query.assert_called_once_with(
        vector=[0.0] * 384,
        top_k=1,
        filter={"session_id": {"$eq": "session_with_docs"}}
    )

@pytest.mark.asyncio
@patch('database.db_manager.get_index')
async def test_has_session_documents_false(mock_get_index):
    mock_index = MagicMock()
    mock_query_response = MagicMock()
    mock_query_response.matches = []  # Simulate no matches
    mock_index.query.return_value = mock_query_response
    mock_get_index.return_value = mock_index

    result = await has_session_documents("session_without_docs")
    assert result is False
    mock_index.query.assert_called_once_with(
        vector=[0.0] * 384,
        top_k=1,
        filter={"session_id": {"$eq": "session_without_docs"}}
    )