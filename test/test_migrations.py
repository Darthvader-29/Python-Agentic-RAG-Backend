"""Alembic migration integration test.

Runs against the real NeonDB TestDB (TEST_DATABASE_URL). Verifies that
`alembic upgrade head` creates the sessions and documents tables, then
runs `downgrade base` to leave the DB clean for the next test run.

This test is deliberately a sync function so that alembic's internal
asyncio.run() does not conflict with the test event loop.
"""

import asyncio
import os

import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import text
from sqlalchemy.pool import NullPool


def test_alembic_upgrade_head_creates_schema():
    raw = os.environ.get("TEST_DATABASE_URL", "")
    if not raw:
        pytest.skip("TEST_DATABASE_URL not set — skipping migration test")

    from sqlalchemy.ext.asyncio import create_async_engine

    from database.session import _to_asyncpg_url

    url, connect_args = _to_asyncpg_url(raw)

    # Point alembic at the test DB for this run
    cfg = Config("alembic.ini")
    # Override DATABASE_URL env so migrations/env.py picks up the test DB
    os.environ["DATABASE_URL"] = raw

    try:
        command.downgrade(cfg, "base")
        command.upgrade(cfg, "head")

        async def _check() -> set:
            engine = create_async_engine(url, connect_args=connect_args, poolclass=NullPool)
            async with engine.connect() as conn:
                result = await conn.execute(
                    text(
                        "SELECT table_name FROM information_schema.tables "
                        "WHERE table_schema = 'public' "
                        "AND table_name IN ('sessions', 'documents')"
                    )
                )
                tables = {row[0] for row in result}
            await engine.dispose()
            return tables

        tables = asyncio.run(_check())
        assert {"sessions", "documents"} <= tables, f"Missing tables: {tables}"

    finally:
        # Clean up: downgrade back to base so the _engine fixture can start fresh
        try:
            command.downgrade(cfg, "base")
        except Exception:
            pass
        # Restore DATABASE_URL to the prod value (conftest dummy wins if no prod .env)
        os.environ.pop("DATABASE_URL", None)
