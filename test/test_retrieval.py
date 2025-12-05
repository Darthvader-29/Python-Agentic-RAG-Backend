import pytest
from unittest.mock import patch, MagicMock
from components.retrieval import retrieve_context

@pytest.mark.asyncio
@patch('components.retrieval.embed_single')
@patch('components.retrieval.search_vectors')
async def test_retrieve_context_rag(mock_search_vectors, mock_embed_single):
    mock_embed_single.return_value = [0.1] * 384
    mock_search_vectors.return_value = [{"text": "RAG context"}]
    
    context = await retrieve_context("test query", "RAG", "session123")
    
    assert context == ["RAG context"]
    mock_embed_single.assert_called_once_with("test query")
    mock_search_vectors.assert_called_once_with(
        query_vector=[0.1] * 384,
        top_k=5,
        session_id="session123"
    )

@pytest.mark.asyncio
@patch('components.retrieval.search_web')
async def test_retrieve_context_web(mock_search_web):
    mock_search_web.return_value = [{"snippet": "WEB context"}]
    
    context = await retrieve_context("test query", "WEB", "session123", web_search_allowed=True)
    
    assert context == ["WEB context"]
    mock_search_web.assert_called_once_with("test query", max_results=5)

@pytest.mark.asyncio
async def test_retrieve_context_direct():
    context = await retrieve_context("test query", "DIRECT", "session123")
    assert context == []

@pytest.mark.asyncio
@patch('components.retrieval.search_web')
async def test_retrieve_context_web_disabled(mock_search_web):
    context = await retrieve_context("test query", "WEB", "session123", web_search_allowed=False)
    assert context == []
    mock_search_web.assert_not_called()
