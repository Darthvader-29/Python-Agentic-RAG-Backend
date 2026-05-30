"""Tests for the DI seam: provider functions, async offloading, and tenacity retry."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from database.db_manager import PineconeClient
from integrations.huggingface.client import HuggingFaceClient
from integrations.s3.client import S3Client

# ── from_settings endpoint ────────────────────────────────────────────────────


def _dev_settings(**overrides):
    defaults = dict(
        S3_ENDPOINT_URL=None,
        ENVIRONMENT="development",
        S3_BUCKET_NAME="test-bucket",
        AWS_REGION="us-east-1",
        AWS_ACCESS_KEY_ID="ak",
        AWS_SECRET_ACCESS_KEY="sk",
    )
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_s3_from_settings_dev_targets_minio():
    """Development with no explicit S3_ENDPOINT_URL auto-targets MinIO."""
    with patch("integrations.s3.client.boto3.client") as mock_boto:
        mock_boto.return_value = MagicMock()
        S3Client.from_settings(_dev_settings())
        _, kwargs = mock_boto.call_args
        assert kwargs["endpoint_url"] == "http://localhost:9000"


def test_s3_from_settings_dev_explicit_endpoint():
    """Explicit S3_ENDPOINT_URL takes priority over the MinIO default."""
    with patch("integrations.s3.client.boto3.client") as mock_boto:
        mock_boto.return_value = MagicMock()
        S3Client.from_settings(_dev_settings(S3_ENDPOINT_URL="http://custom:9000"))
        _, kwargs = mock_boto.call_args
        assert kwargs["endpoint_url"] == "http://custom:9000"


def test_s3_from_settings_prod_uses_real_s3():
    """Production ENVIRONMENT leaves endpoint_url=None (standard AWS S3)."""
    with patch("integrations.s3.client.boto3.client") as mock_boto:
        mock_boto.return_value = MagicMock()
        S3Client.from_settings(_dev_settings(ENVIRONMENT="production"))
        _, kwargs = mock_boto.call_args
        assert kwargs["endpoint_url"] is None


def test_di_providers_read_app_state():
    """get_*_client provider functions return the object stored on app.state."""
    from unittest.mock import MagicMock

    from fastapi import Request

    from dependencies import (
        get_embedding_client,
        get_pinecone_client,
        get_s3_client,
        get_web_search_client,
    )

    fake_pc = MagicMock()
    fake_s3 = MagicMock()
    fake_emb = MagicMock()
    fake_web = MagicMock()

    mock_request = MagicMock(spec=Request)
    mock_request.app.state.pinecone = fake_pc
    mock_request.app.state.s3 = fake_s3
    mock_request.app.state.embedder = fake_emb
    mock_request.app.state.web = fake_web

    assert get_pinecone_client(mock_request) is fake_pc
    assert get_s3_client(mock_request) is fake_s3
    assert get_embedding_client(mock_request) is fake_emb
    assert get_web_search_client(mock_request) is fake_web


# ── asyncio.to_thread offload ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_pinecone_search_offloads_via_to_thread():
    """search_vectors passes its blocking call through asyncio.to_thread."""
    client = PineconeClient(api_key="test", index_name="test-index")
    mock_index = MagicMock()
    mock_index.query.return_value = MagicMock(matches=[])
    client._index = mock_index

    with patch("asyncio.to_thread", wraps=asyncio.to_thread) as spy:
        await client.search_vectors([0.0] * 384, top_k=1)
        assert spy.called


@pytest.mark.asyncio
async def test_hf_embed_batch_offloads_via_to_thread():
    """embed_batch passes its HF inference call through asyncio.to_thread."""
    client = HuggingFaceClient(token="test-token")
    with patch.object(client._client, "feature_extraction", return_value=[[0.1] * 384]):
        with patch("asyncio.to_thread", wraps=asyncio.to_thread) as spy:
            await client.embed_batch(["hello"])
            assert spy.called


# ── tenacity retry ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_pinecone_search_retries_three_times():
    """Transient failures are retried up to 3 times before succeeding."""
    client = PineconeClient(api_key="test", index_name="test-index")
    mock_index = MagicMock()
    mock_index.query.side_effect = [
        RuntimeError("transient"),
        RuntimeError("transient"),
        MagicMock(matches=[]),
    ]
    client._index = mock_index

    with patch("time.sleep"):  # skip tenacity backoff waits
        result = await client.search_vectors([0.0] * 384, top_k=1)

    assert result == []
    assert mock_index.query.call_count == 3


@pytest.mark.asyncio
async def test_pinecone_search_reraises_after_max_retries():
    """After 3 failed attempts, the original exception is reraised (reraise=True)."""
    client = PineconeClient(api_key="test", index_name="test-index")
    mock_index = MagicMock()
    mock_index.query.side_effect = RuntimeError("permanent error")
    client._index = mock_index

    with patch("time.sleep"):
        with pytest.raises(RuntimeError, match="permanent error"):
            await client.search_vectors([0.0] * 384, top_k=1)

    assert mock_index.query.call_count == 3
