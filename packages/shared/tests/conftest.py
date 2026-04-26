"""Shared pytest fixtures for the BUF-6 integration tests.

The integration fixtures are skipped automatically when ``TEST_DATABASE_URL``
is not set, so a fresh clone with no Postgres running still has a green
``uv run pytest``. Local devs typically run::

    docker compose up -d --wait postgres
    TEST_DATABASE_URL=postgresql+psycopg://nexus:nexus@localhost:5432/nexus uv run pytest

CI provides a Postgres service container and exports the URL.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Generator
from pathlib import Path

import pytest
from esports_sim.db.base import Base
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.orm import Session, sessionmaker


def _shared_pkg_root() -> Path:
    """Path to ``packages/shared`` so we can invoke alembic from there."""
    # tests/ is a sibling of alembic/ inside packages/shared.
    return Path(__file__).resolve().parent.parent


def _resolve_test_url() -> str | None:
    """Return the TEST_DATABASE_URL with a sync driver prefix, or None."""
    url = os.getenv("TEST_DATABASE_URL")
    if not url:
        return None
    # alembic + the sync engine want psycopg, even if a service config nearby
    # uses asyncpg. Normalise so callers don't have to maintain two URLs.
    if url.startswith("postgresql+asyncpg://"):
        return url.replace("postgresql+asyncpg://", "postgresql+psycopg://", 1)
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url


@pytest.fixture(scope="session")
def db_engine() -> Generator[Engine, None, None]:
    """Session-scoped engine with migrations applied to head.

    Skips the test when ``TEST_DATABASE_URL`` is not configured. After the
    session ends, migrations are rolled back to leave the DB clean for the
    next run.
    """
    url = _resolve_test_url()
    if not url:
        pytest.skip("TEST_DATABASE_URL not set; skipping Postgres integration tests")

    # Run alembic via subprocess so we exercise the same CLI path operators
    # use, not just programmatic upgrades. The env vars override alembic.ini.
    env = {**os.environ, "DATABASE_URL": url}
    upgrade = subprocess.run(
        [sys.executable, "-m", "alembic", "upgrade", "head"],
        cwd=_shared_pkg_root(),
        env=env,
        capture_output=True,
        text=True,
    )
    if upgrade.returncode != 0:
        pytest.fail(
            "alembic upgrade head failed:\n"
            f"stdout:\n{upgrade.stdout}\n"
            f"stderr:\n{upgrade.stderr}"
        )

    engine = create_engine(url, future=True)
    try:
        yield engine
    finally:
        # Roll back to a clean DB so re-runs are idempotent. Drop the
        # alembic_version table too — otherwise the next run thinks it's
        # already migrated.
        with engine.begin() as conn:
            Base.metadata.drop_all(bind=conn)
            conn.execute(text("DROP TABLE IF EXISTS alembic_version"))
            for typename in (
                "review_status",
                "staging_status",
                "platform",
                "entity_type",
            ):
                conn.execute(text(f"DROP TYPE IF EXISTS {typename}"))
        engine.dispose()


@pytest.fixture
def db_session(db_engine: Engine) -> Generator[Session, None, None]:
    """Function-scoped session with savepoint isolation.

    Each test gets a fresh outer transaction; SQLAlchemy 2.0's
    ``join_transaction_mode="create_savepoint"`` makes ``session.commit()``
    inside the test create a savepoint instead of really committing, so a
    deliberate ``IntegrityError`` doesn't leak across tests.
    """
    connection = db_engine.connect()
    transaction = connection.begin()
    SessionLocal = sessionmaker(
        bind=connection,
        expire_on_commit=False,
        join_transaction_mode="create_savepoint",
    )
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
        if transaction.is_active:
            transaction.rollback()
        connection.close()
