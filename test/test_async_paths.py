"""DI-override integration tests: prove every endpoint pulls clients from DI."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession

from app import app
from database.db_manager import PineconeClient
from dependencies import (
    get_db_session,
    get_db_sessionmaker,
    get_embedding_client,
    get_pinecone_client,
    get_s3_client,
    get_web_search_client,
)


@pytest.fixture
def fake_pinecone():
    pc = AsyncMock(spec=PineconeClient)
    pc.search_vectors.return_value = []
    pc.save_vectors.return_value = None
    pc.delete_vectors_by_session.return_value = None
    return pc


@pytest.fixture
def fake_embedder():
    emb = AsyncMock()
    emb.embed_batch.return_value = [[0.1] * 384]
    emb.embed_single.return_value = [0.1] * 384
    return emb


@pytest.fixture
def fake_s3():
    s3 = AsyncMock()
    s3.upload_fileobj.return_value = "uploads/test-key.pdf"
    s3.delete_objects.return_value = None
    return s3


@pytest.fixture
def fake_web():
    web = AsyncMock()
    web.search_web.return_value = []
    return web


@pytest.fixture
def fake_db():
    """Mock AsyncSession — returned by get_db_session override."""
    return AsyncMock(spec=AsyncSession)


@pytest.fixture
def di_client(fake_pinecone, fake_embedder, fake_s3, fake_web, fake_db):
    """TestClient with DI overrides and a patched lifespan to avoid real network calls."""
    fake_sessionmaker = AsyncMock()

    async def _db_session_override():
        yield fake_db

    app.dependency_overrides[get_pinecone_client] = lambda: fake_pinecone
    app.dependency_overrides[get_embedding_client] = lambda: fake_embedder
    app.dependency_overrides[get_s3_client] = lambda: fake_s3
    app.dependency_overrides[get_web_search_client] = lambda: fake_web
    app.dependency_overrides[get_db_session] = _db_session_override
    app.dependency_overrides[get_db_sessionmaker] = lambda: fake_sessionmaker

    with patch.object(PineconeClient, "ensure_index", new_callable=AsyncMock):
        with TestClient(app, raise_server_exceptions=False) as client:
            yield client

    app.dependency_overrides.clear()


# ── cleanup endpoint ──────────────────────────────────────────────────────────


def test_cleanup_uses_di_pinecone_s3_and_db(di_client, fake_pinecone, fake_s3):
    """cleanup_session resolves pinecone, s3, and db from DI; keys come from Postgres."""
    with (
        patch("app.repo.list_s3_keys_for_session", new_callable=AsyncMock) as mock_keys,
        patch("app.repo.delete_session", new_callable=AsyncMock),
    ):
        mock_keys.return_value = ["key1", "key2"]

        resp = di_client.post("/api/cleanup", json={"session_id": "test-session"})

    assert resp.status_code == 200
    data = resp.json()
    assert data["session_id"] == "test-session"
    assert data["deleted_files"] == 2

    fake_pinecone.delete_vectors_by_session.assert_awaited_once_with("test-session")
    fake_s3.delete_objects.assert_awaited_once_with(["key1", "key2"])


def test_cleanup_no_files_skips_s3_delete(di_client, fake_pinecone, fake_s3):
    """When session has no documents, s3.delete_objects is not called."""
    with (
        patch("app.repo.list_s3_keys_for_session", new_callable=AsyncMock) as mock_keys,
        patch("app.repo.delete_session", new_callable=AsyncMock),
    ):
        mock_keys.return_value = []

        resp = di_client.post("/api/cleanup", json={"session_id": "empty-session"})

    assert resp.status_code == 200
    assert resp.json()["deleted_files"] == 0
    fake_s3.delete_objects.assert_not_awaited()


# ── chat endpoint ─────────────────────────────────────────────────────────────


def test_chat_uses_di_clients(di_client, fake_pinecone, fake_embedder, fake_web):
    """chat endpoint resolves all clients from DI; route_query reads has_docs from DB."""
    with (
        patch("components.router.repo.session_has_documents", new_callable=AsyncMock) as mock_hd,
        patch("components.router.gemini_model.generate_content_async") as mock_gemini,
        patch("app.generate_final_response", new_callable=AsyncMock) as mock_gen,
    ):
        mock_hd.return_value = False
        mock_gemini.return_value = MagicMock(text="DIRECT")
        mock_gen.return_value = "Test answer"

        resp = di_client.post(
            "/api/chat",
            json={
                "message": "Hello",
                "session_id": "test-session",
                "web_search_allowed": False,
            },
        )

    assert resp.status_code == 200
    # Postgres (not Pinecone) is queried for has_documents
    mock_hd.assert_awaited()
    # Vector search still goes through Pinecone for relevance
    fake_pinecone.search_vectors.assert_awaited()


# ── health endpoint ───────────────────────────────────────────────────────────


def test_health_returns_healthy(di_client):
    resp = di_client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"
