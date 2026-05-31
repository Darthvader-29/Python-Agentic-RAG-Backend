"""Async data-access layer for session and document state.

Each function accepts an AsyncSession and performs a single focused query.
The caller (endpoint or background task) owns the transaction boundary.
"""

from sqlalchemy import delete, exists, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from database.models import Document, DocumentStatus, Session


async def get_or_create_session(db: AsyncSession, session_id: str) -> None:
    """Idempotent upsert — safe to call multiple times for the same session_id."""
    stmt = pg_insert(Session).values(id=session_id).on_conflict_do_nothing(index_elements=["id"])
    await db.execute(stmt)


async def create_document(
    db: AsyncSession, *, session_id: str, s3_key: str, filename: str
) -> Document:
    doc = Document(
        session_id=session_id,
        s3_key=s3_key,
        filename=filename,
        status=DocumentStatus.PENDING,
    )
    db.add(doc)
    await db.flush()  # populate doc.id without ending the request transaction
    return doc


async def set_document_status(db: AsyncSession, *, s3_key: str, status: DocumentStatus) -> None:
    await db.execute(update(Document).where(Document.s3_key == s3_key).values(status=status))


async def session_has_documents(db: AsyncSession, session_id: str) -> bool:
    stmt = select(exists().where(Document.session_id == session_id))
    return bool(await db.scalar(stmt))


async def list_s3_keys_for_session(db: AsyncSession, session_id: str) -> list[str]:
    stmt = (
        select(Document.s3_key).where(Document.session_id == session_id).order_by(Document.s3_key)
    )
    return list(await db.scalars(stmt))


async def delete_session(db: AsyncSession, session_id: str) -> None:
    # FK ON DELETE CASCADE removes the session's documents atomically
    await db.execute(delete(Session).where(Session.id == session_id))
