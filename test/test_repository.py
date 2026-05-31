"""Repository-layer tests against the real NeonDB TestDB.

Requires TEST_DATABASE_URL in the environment. The conftest.py _engine fixture
creates/drops tables around the session; each test rolls back its own transaction.
"""

import pytest

from database import repository as repo
from database.models import DocumentStatus

pytestmark = pytest.mark.asyncio


async def test_session_has_documents_false_when_empty(db_session):
    await repo.get_or_create_session(db_session, "s1")
    assert await repo.session_has_documents(db_session, "s1") is False


async def test_create_document_and_has_documents(db_session):
    await repo.get_or_create_session(db_session, "s1")
    await repo.create_document(db_session, session_id="s1", s3_key="uploads/a", filename="a.pdf")
    assert await repo.session_has_documents(db_session, "s1") is True
    assert await repo.list_s3_keys_for_session(db_session, "s1") == ["uploads/a"]


async def test_set_document_status_to_ready(db_session):
    await repo.get_or_create_session(db_session, "s1")
    await repo.create_document(db_session, session_id="s1", s3_key="uploads/a", filename="a.pdf")
    await repo.set_document_status(db_session, s3_key="uploads/a", status=DocumentStatus.READY)
    keys = await repo.list_s3_keys_for_session(db_session, "s1")
    assert keys == ["uploads/a"]


async def test_set_document_status_to_failed(db_session):
    await repo.get_or_create_session(db_session, "s1")
    await repo.create_document(db_session, session_id="s1", s3_key="uploads/b", filename="b.pdf")
    await repo.set_document_status(db_session, s3_key="uploads/b", status=DocumentStatus.FAILED)
    keys = await repo.list_s3_keys_for_session(db_session, "s1")
    assert keys == ["uploads/b"]


async def test_list_s3_keys_multiple_docs(db_session):
    await repo.get_or_create_session(db_session, "s1")
    await repo.create_document(db_session, session_id="s1", s3_key="uploads/c", filename="c.pdf")
    await repo.create_document(db_session, session_id="s1", s3_key="uploads/d", filename="d.pdf")
    keys = await repo.list_s3_keys_for_session(db_session, "s1")
    assert sorted(keys) == ["uploads/c", "uploads/d"]


async def test_delete_session_cascades_documents(db_session):
    await repo.get_or_create_session(db_session, "s1")
    await repo.create_document(db_session, session_id="s1", s3_key="uploads/a", filename="a.pdf")
    await repo.delete_session(db_session, "s1")
    assert await repo.session_has_documents(db_session, "s1") is False
    assert await repo.list_s3_keys_for_session(db_session, "s1") == []


async def test_get_or_create_session_is_idempotent(db_session):
    """Calling get_or_create_session twice must not raise."""
    await repo.get_or_create_session(db_session, "idempotent-session")
    await repo.get_or_create_session(db_session, "idempotent-session")
    assert await repo.session_has_documents(db_session, "idempotent-session") is False
