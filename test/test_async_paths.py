"""DI-override integration tests: prove every endpoint pulls clients from DI."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app import app
from database.db_manager import PineconeClient
from dependencies import (
    get_embedding_client,
    get_pinecone_client,
    get_s3_client,
    get_web_search_client,
)


@pytest.fixture
def fake_pinecone():
    pc = AsyncMock(spec=PineconeClient)
    pc.has_session_documents.return_value = False
    pc.search_vectors.return_value = []
    pc.save_vectors.return_value = None
    pc.delete_vectors_by_session.return_value = None
    pc.list_s3_keys_for_session.return_value = []
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
def di_client(fake_pinecone, fake_embedder, fake_s3, fake_web):
    """TestClient with DI overrides and a patched lifespan to avoid real network calls."""
    app.dependency_overrides[get_pinecone_client] = lambda: fake_pinecone
    app.dependency_overrides[get_embedding_client] = lambda: fake_embedder
    app.dependency_overrides[get_s3_client] = lambda: fake_s3
    app.dependency_overrides[get_web_search_client] = lambda: fake_web

    # Prevent lifespan from making real external calls
    with patch.object(PineconeClient, "ensure_index", new_callable=AsyncMock):
        with TestClient(app, raise_server_exceptions=False) as client:
            yield client

    app.dependency_overrides.clear()


# ── cleanup endpoint ──────────────────────────────────────────────────────────


def test_cleanup_uses_di_pinecone_and_s3(di_client, fake_pinecone, fake_s3):
    """cleanup_session resolves pinecone and s3 from DI, not module imports."""
    resp = di_client.post(
        "/api/cleanup",
        json={"session_id": "test-session", "file_keys": ["key1", "key2"]},
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["session_id"] == "test-session"
    assert data["deleted_files"] == 2

    fake_pinecone.delete_vectors_by_session.assert_awaited_once_with("test-session")
    fake_s3.delete_objects.assert_awaited_once_with(["key1", "key2"])


def test_cleanup_fallback_to_pinecone_s3_keys(di_client, fake_pinecone, fake_s3):
    """When file_keys is empty, cleanup fetches keys from Pinecone."""
    fake_pinecone.list_s3_keys_for_session.return_value = ["pinecone-key"]

    resp = di_client.post(
        "/api/cleanup",
        json={"session_id": "test-session", "file_keys": []},
    )

    assert resp.status_code == 200
    fake_pinecone.list_s3_keys_for_session.assert_awaited_once_with("test-session")
    fake_s3.delete_objects.assert_awaited_once_with(["pinecone-key"])


# ── chat endpoint ─────────────────────────────────────────────────────────────


def test_chat_uses_di_clients(di_client, fake_pinecone, fake_embedder, fake_web):
    """chat endpoint resolves all three clients from DI."""
    with (
        patch("components.router.gemini_model.generate_content_async") as mock_gemini,
        patch("app.generate_final_response", new_callable=AsyncMock) as mock_gen,
    ):
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
    # Verify injected clients were actually called (not module-level singletons)
    fake_pinecone.has_session_documents.assert_awaited()
    fake_pinecone.search_vectors.assert_awaited()


# ── health endpoint ───────────────────────────────────────────────────────────


def test_health_returns_healthy(di_client):
    resp = di_client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "healthy"
