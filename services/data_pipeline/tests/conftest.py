"""Pytest fixtures for the data-pipeline tests (BUF-9).

Mirrors the ``packages/shared`` fixture layout: an integration-marked
``db_session`` that runs alembic against ``TEST_DATABASE_URL`` and yields
a savepoint-isolated session. Skipped automatically when
``TEST_DATABASE_URL`` is unset, so a fresh clone with no Postgres still
has a green ``uv run pytest``.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Generator
from pathlib import Path

# Sibling-import path: pytest's ``--import-mode=importlib`` skips the
# sys.path manipulation that ``prepend`` mode would otherwise do, so
# adding the test directory here keeps ``from pipeline_fixtures import
# ...`` working in the per-test files. The data-pipeline helper is
# *uniquely* named (``pipeline_fixtures``) so that even though both
# test directories end up on sys.path, neither shadows the other.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pytest
from esports_sim.db.base import Base
from sqlalchemy import Engine, create_engine, text
from sqlalchemy.orm import Session, sessionmaker


def _shared_pkg_root() -> Path:
    """Path to ``packages/shared`` so we can invoke alembic from there.

    The migrations live under the shared package, not under data_pipeline
    — alembic is run from the workspace root in production but tests
    spin it up programmatically.
    """
    # tests/ -> data_pipeline -> services -> repo root
    return Path(__file__).resolve().parents[3] / "packages" / "shared"


def _resolve_test_url() -> str | None:
    url = os.getenv("TEST_DATABASE_URL")
    if not url:
        return None
    if url.startswith("postgresql+asyncpg://"):
        return url.replace("postgresql+asyncpg://", "postgresql+psycopg://", 1)
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url


@pytest.fixture(scope="session")
def db_engine() -> Generator[Engine, None, None]:
    url = _resolve_test_url()
    if not url:
        pytest.skip("TEST_DATABASE_URL not set; skipping Postgres integration tests")

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
    """Function-scoped session with savepoint isolation per test."""
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
