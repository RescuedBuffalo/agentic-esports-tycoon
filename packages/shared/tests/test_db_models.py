"""Integration tests against a real Postgres for the BUF-6 schema.

Skipped automatically when ``TEST_DATABASE_URL`` is not set — see
``conftest.py``. CI provides a Postgres service container; locally:

    docker compose up -d --wait postgres
    TEST_DATABASE_URL=postgresql+psycopg://nexus:nexus@localhost:5432/nexus \\
        uv run pytest -m integration
"""

from __future__ import annotations

import uuid

import pytest
from esports_sim.db.enums import EntityType, Platform, ReviewStatus, StagingStatus
from esports_sim.db.models import (
    Entity,
    EntityAlias,
)
from sqlalchemy import inspect, select, text
from sqlalchemy.exc import IntegrityError

from tests.fixtures import (
    make_entity,
    make_entity_alias,
    make_raw_record,
    make_review_queue_item,
    make_staging_record,
)

pytestmark = pytest.mark.integration


def test_alembic_upgrade_creates_all_tables(db_engine) -> None:
    """`alembic upgrade head` produces every table the application expects."""
    insp = inspect(db_engine)
    tables = set(insp.get_table_names())
    expected = {
        "entity",
        "entity_alias",
        "staging_record",
        "raw_record",
        "alias_review_queue",
        "alembic_version",  # bookkeeping table that proves the migration ran
    }
    missing = expected - tables
    assert not missing, f"missing tables: {missing}"


def test_pgvector_extension_enabled(db_engine) -> None:
    """The `vector` extension must be live so BUF-28 can land vector tables."""
    with db_engine.connect() as conn:
        row = conn.execute(
            text("SELECT extname FROM pg_extension WHERE extname = 'vector'")
        ).first()
    assert row is not None, "pgvector extension was not created by the migration"


def test_player_entity_with_two_aliases(db_session) -> None:
    """Acceptance scenario from BUF-6: insert a player + two aliases.

    Two aliases with different ``(platform, platform_id)`` pairs are legal —
    that's exactly how a player tracked on both VLR and Liquipedia ends up
    with one canonical id and two alias rows.
    """
    e = make_entity(entity_type=EntityType.PLAYER)
    make_entity_alias(entity=e, platform=Platform.VLR, platform_id="vlr-1234", platform_name="TenZ")
    make_entity_alias(
        entity=e,
        platform=Platform.LIQUIPEDIA,
        platform_id="liq-tenz",
        platform_name="TenZ",
        confidence=0.95,
    )
    db_session.add(e)
    db_session.flush()

    stored = db_session.get(Entity, e.canonical_id)
    assert stored is not None
    assert stored.entity_type is EntityType.PLAYER
    assert stored.is_active is True
    assert len(stored.aliases) == 2
    assert {a.platform for a in stored.aliases} == {Platform.VLR, Platform.LIQUIPEDIA}


def test_alias_unique_constraint_prevents_duplicate_platform_id(db_session) -> None:
    """`(platform, platform_id)` is the schema-level guarantee no scraper can
    silently split a player into two canonical rows by re-indexing the same
    handle under a fresh entity.
    """
    e1 = make_entity(entity_type=EntityType.PLAYER)
    make_entity_alias(
        entity=e1, platform=Platform.VLR, platform_id="vlr-shared", platform_name="TenZ"
    )
    db_session.add(e1)
    db_session.flush()

    e2 = make_entity(entity_type=EntityType.PLAYER)
    make_entity_alias(
        entity=e2,
        platform=Platform.VLR,
        platform_id="vlr-shared",  # same (platform, platform_id) — should fail
        platform_name="Imposter",
        confidence=0.5,
    )
    db_session.add(e2)
    with pytest.raises(IntegrityError):
        db_session.flush()
    db_session.rollback()


def test_alias_same_platform_id_different_platform_is_legal(db_session) -> None:
    """Same string is fine across two platforms — uniqueness is per platform."""
    e = make_entity(entity_type=EntityType.PLAYER)
    make_entity_alias(
        entity=e,
        platform=Platform.VLR,
        platform_id="shared-string",
        platform_name="TenZ",
    )
    make_entity_alias(
        entity=e,
        platform=Platform.LIQUIPEDIA,
        platform_id="shared-string",
        platform_name="TenZ",
    )
    db_session.add(e)
    db_session.flush()  # No IntegrityError.


def test_staging_record_defaults_pending(db_session) -> None:
    """`status` defaults to pending so scrapers can omit it."""
    sr = make_staging_record()
    db_session.add(sr)
    db_session.flush()
    db_session.refresh(sr)
    assert sr.status is StagingStatus.PENDING


def test_staging_record_canonical_id_nullable(db_session) -> None:
    """A pre-resolution row legally has no canonical_id."""
    sr = make_staging_record(canonical_id=None, status=StagingStatus.REVIEW)
    db_session.add(sr)
    db_session.flush()
    assert sr.canonical_id is None


def test_raw_record_content_hash_unique(db_session) -> None:
    """Re-fetching the same payload must not produce a duplicate raw_record."""
    a = make_raw_record(content_hash="abc123")
    b = make_raw_record(content_hash="abc123")  # same hash, different id
    db_session.add(a)
    db_session.flush()
    db_session.add(b)
    with pytest.raises(IntegrityError):
        db_session.flush()
    db_session.rollback()


def test_review_queue_item_defaults_pending(db_session) -> None:
    item = make_review_queue_item()
    db_session.add(item)
    db_session.flush()
    db_session.refresh(item)
    assert item.status is ReviewStatus.PENDING


def test_cascade_delete_drops_aliases_with_entity(db_session) -> None:
    """Deleting a canonical entity removes its aliases (FK ON DELETE CASCADE).

    Important for the resolver's ``handle_rebrand`` (BUF-12) error path: a
    botched merge that has to be undone shouldn't leave orphan alias rows
    pointing at a missing canonical_id.
    """
    e = make_entity()
    make_entity_alias(entity=e)
    db_session.add(e)
    db_session.flush()

    db_session.delete(e)
    db_session.flush()

    remaining = db_session.execute(select(EntityAlias)).scalars().all()
    assert remaining == []


def test_staging_canonical_id_nulls_on_entity_delete(db_session) -> None:
    """`staging_record.canonical_id` uses ON DELETE SET NULL, not CASCADE.

    Staging is an audit trail; the row should survive the canonical it was
    pointing at. Otherwise we'd lose the input that produced a bad merge.
    """
    e = make_entity()
    sr = make_staging_record(canonical_id=e.canonical_id, status=StagingStatus.PROCESSED)
    db_session.add_all([e, sr])
    db_session.flush()

    db_session.delete(e)
    db_session.flush()
    db_session.refresh(sr)
    assert sr.canonical_id is None


def test_uuid_default_for_entity(db_session) -> None:
    """Inserting without an explicit canonical_id mints one client-side."""
    e = Entity(entity_type=EntityType.TEAM)
    db_session.add(e)
    db_session.flush()
    assert isinstance(e.canonical_id, uuid.UUID)
