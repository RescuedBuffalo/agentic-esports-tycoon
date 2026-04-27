"""Tests for the canonical-id resolver (BUF-7).

The four unit-style cases (exact match, high-confidence fuzzy, low-confidence
fuzzy, brand-new entity) plus an integration scenario covering cross-platform
consistency on "TenZ" land here.

Marked ``integration`` because the resolver is intentionally written against a
real SQLAlchemy session — the BUF-6 schema uses Postgres-specific types
(JSONB, named ENUMs) that an in-memory SQLite cannot fake without enough
adapter scaffolding to defeat the point. CI provides Postgres; locally
``TEST_DATABASE_URL`` skips these out gracefully.
"""

from __future__ import annotations

import pytest
from esports_sim.db.enums import (
    EntityType,
    Platform,
    ReviewStatus,
    StagingStatus,
)
from esports_sim.db.models import (
    AliasReviewQueue,
    Entity,
    EntityAlias,
    StagingInvariantError,
    StagingRecord,
)
from esports_sim.resolver import (
    AUTO_MERGE_THRESHOLD,
    REVIEW_THRESHOLD,
    ResolutionStatus,
    resolve_entity,
)
from sqlalchemy import select

from tests.fixtures import (
    make_entity,
    make_entity_alias,
    make_staging_record,
)

pytestmark = pytest.mark.integration


# --- exact match -------------------------------------------------------------


def test_exact_match_returns_existing_canonical(db_session) -> None:
    """A second call with the same (platform, platform_id) is a free lookup."""
    entity = make_entity(entity_type=EntityType.PLAYER)
    alias = make_entity_alias(
        entity=entity,
        platform=Platform.VLR,
        platform_id="vlr-tenz",
        platform_name="TenZ",
    )
    db_session.add_all([entity, alias])
    db_session.flush()

    result = resolve_entity(
        db_session,
        platform=Platform.VLR,
        platform_id="vlr-tenz",
        platform_name="TenZ",
        entity_type=EntityType.PLAYER,
    )

    assert result.status is ResolutionStatus.MATCHED
    assert result.canonical_id == entity.canonical_id
    assert result.confidence == 1.0
    # No new alias rows should appear.
    aliases = db_session.execute(select(EntityAlias)).scalars().all()
    assert len(aliases) == 1


def test_exact_match_is_idempotent(db_session) -> None:
    """Re-resolving the exact same triple twice produces no extra rows."""
    first = resolve_entity(
        db_session,
        platform=Platform.VLR,
        platform_id="vlr-newcomer",
        platform_name="Newcomer",
        entity_type=EntityType.PLAYER,
    )
    db_session.flush()

    second = resolve_entity(
        db_session,
        platform=Platform.VLR,
        platform_id="vlr-newcomer",
        platform_name="Newcomer",
        entity_type=EntityType.PLAYER,
    )

    assert first.status is ResolutionStatus.CREATED
    assert second.status is ResolutionStatus.MATCHED
    assert first.canonical_id == second.canonical_id

    aliases = (
        db_session.execute(
            select(EntityAlias).where(EntityAlias.canonical_id == first.canonical_id)
        )
        .scalars()
        .all()
    )
    assert len(aliases) == 1


# --- high-confidence fuzzy match (auto-merge) --------------------------------


def test_high_confidence_fuzzy_auto_merges(db_session) -> None:
    """Score >= AUTO_MERGE_THRESHOLD adds an alias under the existing canonical."""
    entity = make_entity(entity_type=EntityType.PLAYER)
    db_session.add(entity)
    db_session.add(
        make_entity_alias(
            entity=entity,
            platform=Platform.VLR,
            platform_id="vlr-tenz",
            platform_name="TenZ",
        )
    )
    db_session.flush()

    # "tenz" vs "TenZ" is a near-perfect WRatio (case-insensitive, identical
    # tokens) — well above 0.90.
    result = resolve_entity(
        db_session,
        platform=Platform.LIQUIPEDIA,
        platform_id="liq-tenz",
        platform_name="tenz",
        entity_type=EntityType.PLAYER,
    )

    assert result.status is ResolutionStatus.AUTO_MERGED
    assert result.canonical_id == entity.canonical_id
    assert result.confidence >= AUTO_MERGE_THRESHOLD

    aliases = (
        db_session.execute(
            select(EntityAlias)
            .where(EntityAlias.canonical_id == entity.canonical_id)
            .order_by(EntityAlias.platform)
        )
        .scalars()
        .all()
    )
    assert {a.platform for a in aliases} == {Platform.VLR, Platform.LIQUIPEDIA}
    new_alias = next(a for a in aliases if a.platform is Platform.LIQUIPEDIA)
    assert new_alias.confidence == result.confidence


# --- low-confidence fuzzy match (review queue) -------------------------------


def test_low_confidence_fuzzy_enqueues_review(db_session) -> None:
    """Score in [REVIEW, AUTO_MERGE) lands on the human review queue."""
    # Two existing players that score in the review band against the query
    # name. "Sentinel" / "Sinatraa" vs "Sentinal" sit near the middle band on
    # WRatio without crossing 0.90.
    e1 = make_entity(entity_type=EntityType.PLAYER)
    e2 = make_entity(entity_type=EntityType.PLAYER)
    db_session.add_all([e1, e2])
    db_session.add_all(
        [
            make_entity_alias(
                entity=e1,
                platform=Platform.VLR,
                platform_id="vlr-sentinel",
                platform_name="Sentinel",
            ),
            make_entity_alias(
                entity=e2,
                platform=Platform.VLR,
                platform_id="vlr-sinatraa",
                platform_name="Sinatraa",
            ),
        ]
    )
    db_session.flush()

    result = resolve_entity(
        db_session,
        platform=Platform.LIQUIPEDIA,
        platform_id="liq-sentinal",
        platform_name="Sentinal",
        entity_type=EntityType.PLAYER,
    )

    assert result.status is ResolutionStatus.PENDING
    assert result.canonical_id is None
    assert REVIEW_THRESHOLD <= result.confidence < AUTO_MERGE_THRESHOLD
    assert result.candidates, "review path must surface candidates to the human"

    review_rows = db_session.execute(select(AliasReviewQueue)).scalars().all()
    assert len(review_rows) == 1
    queued = review_rows[0]
    assert queued.platform is Platform.LIQUIPEDIA
    assert queued.platform_id == "liq-sentinal"
    assert queued.status is ReviewStatus.PENDING
    assert len(queued.candidates) == len(result.candidates)
    # No alias was minted along the review path.
    assert (
        db_session.execute(
            select(EntityAlias).where(EntityAlias.platform == Platform.LIQUIPEDIA)
        ).scalar_one_or_none()
        is None
    )


def test_low_confidence_fuzzy_is_idempotent(db_session) -> None:
    """Calling resolve twice with the same review-band triple shouldn't dup."""
    e1 = make_entity(entity_type=EntityType.PLAYER)
    db_session.add(e1)
    db_session.add(
        make_entity_alias(
            entity=e1,
            platform=Platform.VLR,
            platform_id="vlr-sentinel",
            platform_name="Sentinel",
        )
    )
    db_session.flush()

    first = resolve_entity(
        db_session,
        platform=Platform.LIQUIPEDIA,
        platform_id="liq-sentinal",
        platform_name="Sentinal",
        entity_type=EntityType.PLAYER,
    )
    second = resolve_entity(
        db_session,
        platform=Platform.LIQUIPEDIA,
        platform_id="liq-sentinal",
        platform_name="Sentinal",
        entity_type=EntityType.PLAYER,
    )
    assert first.status is ResolutionStatus.PENDING
    assert second.status is ResolutionStatus.PENDING

    review_rows = db_session.execute(select(AliasReviewQueue)).scalars().all()
    assert len(review_rows) == 1


# --- brand-new entity --------------------------------------------------------


def test_brand_new_entity_creates_canonical_and_alias(db_session) -> None:
    """No exact, no fuzzy: mint a fresh canonical with confidence 1.0."""
    result = resolve_entity(
        db_session,
        platform=Platform.VLR,
        platform_id="vlr-rookie",
        platform_name="QuasarBlaze",
        entity_type=EntityType.PLAYER,
    )

    assert result.status is ResolutionStatus.CREATED
    assert result.canonical_id is not None
    assert result.confidence == 1.0

    entity = db_session.get(Entity, result.canonical_id)
    assert entity is not None
    assert entity.entity_type is EntityType.PLAYER

    alias = db_session.execute(
        select(EntityAlias).where(EntityAlias.canonical_id == result.canonical_id)
    ).scalar_one()
    assert alias.platform is Platform.VLR
    assert alias.platform_id == "vlr-rookie"
    assert alias.platform_name == "QuasarBlaze"
    assert alias.confidence == 1.0


def test_resolver_isolates_per_entity_type(db_session) -> None:
    """A team named the same as a player shouldn't cause an auto-merge."""
    player = make_entity(entity_type=EntityType.PLAYER)
    db_session.add(player)
    db_session.add(
        make_entity_alias(
            entity=player,
            platform=Platform.VLR,
            platform_id="vlr-cloud",
            platform_name="Cloud",
        )
    )
    db_session.flush()

    result = resolve_entity(
        db_session,
        platform=Platform.LIQUIPEDIA,
        platform_id="liq-cloud-team",
        platform_name="Cloud",
        entity_type=EntityType.TEAM,  # different type — must not match the player
    )
    assert result.status is ResolutionStatus.CREATED
    assert result.canonical_id != player.canonical_id


def test_resolver_rejects_empty_platform_id(db_session) -> None:
    with pytest.raises(ValueError):
        resolve_entity(
            db_session,
            platform=Platform.VLR,
            platform_id="",
            platform_name="TenZ",
            entity_type=EntityType.PLAYER,
        )


# --- TenZ cross-platform integration scenario -------------------------------


def test_tenz_resolves_to_same_canonical_via_vlr_then_liquipedia(db_session) -> None:
    """Acceptance scenario from BUF-7.

    The first scrape sees TenZ on VLR; the resolver mints a fresh canonical.
    The second scrape sees lowercase "tenz" on Liquipedia; the resolver finds
    the existing canonical and adds a second alias under it.
    """
    first = resolve_entity(
        db_session,
        platform=Platform.VLR,
        platform_id="vlr-1234",
        platform_name="TenZ",
        entity_type=EntityType.PLAYER,
    )
    assert first.status is ResolutionStatus.CREATED

    second = resolve_entity(
        db_session,
        platform=Platform.LIQUIPEDIA,
        platform_id="liq-tenz",
        platform_name="tenz",
        entity_type=EntityType.PLAYER,
    )
    assert second.status is ResolutionStatus.AUTO_MERGED
    assert second.canonical_id == first.canonical_id

    aliases = (
        db_session.execute(
            select(EntityAlias)
            .where(EntityAlias.canonical_id == first.canonical_id)
            .order_by(EntityAlias.platform)
        )
        .scalars()
        .all()
    )
    assert len(aliases) == 2
    assert {a.platform for a in aliases} == {Platform.VLR, Platform.LIQUIPEDIA}


# --- StagingRecord.save() guard ---------------------------------------------


def test_staging_save_allows_null_canonical_in_pending(db_session) -> None:
    """PENDING is the pre-resolver queue state; null canonical is fine."""
    sr = make_staging_record(canonical_id=None, status=StagingStatus.PENDING)
    sr.save(db_session)
    db_session.flush()
    assert sr.canonical_id is None
    assert sr.status is StagingStatus.PENDING


def test_staging_save_rejects_null_canonical_in_processed(db_session) -> None:
    """PROCESSED is the only status that asserts the row has a canonical id."""
    sr = make_staging_record(canonical_id=None, status=StagingStatus.PROCESSED)
    with pytest.raises(StagingInvariantError):
        sr.save(db_session)


def test_staging_save_allows_null_canonical_in_review(db_session) -> None:
    """A row deferred to the alias review queue legitimately has no canonical."""
    sr = make_staging_record(canonical_id=None, status=StagingStatus.REVIEW)
    sr.save(db_session)
    db_session.flush()
    assert sr.canonical_id is None
    assert sr.status is StagingStatus.REVIEW


def test_staging_save_allows_null_canonical_in_blocked(db_session) -> None:
    sr = make_staging_record(canonical_id=None, status=StagingStatus.BLOCKED)
    sr.save(db_session)
    db_session.flush()
    assert sr.status is StagingStatus.BLOCKED


def test_staging_event_listener_blocks_direct_session_add(db_session) -> None:
    """Even a scraper that calls session.add directly cannot bypass the rule."""
    sr = StagingRecord(
        source="vlr",
        entity_type=EntityType.PLAYER,
        canonical_id=None,
        payload={"name": "TenZ"},
        status=StagingStatus.PROCESSED,
    )
    db_session.add(sr)
    with pytest.raises(StagingInvariantError):
        db_session.flush()
    db_session.rollback()


def test_staging_save_with_valid_canonical_persists(db_session) -> None:
    """The happy path: resolver populated canonical_id, save() lets it through."""
    entity = make_entity(entity_type=EntityType.PLAYER)
    db_session.add(entity)
    db_session.flush()

    sr = make_staging_record(canonical_id=entity.canonical_id, status=StagingStatus.PROCESSED)
    sr.save(db_session)
    db_session.flush()
    assert sr.canonical_id == entity.canonical_id


def test_staging_unset_status_lands_as_pending(db_session) -> None:
    """Constructing a row without a status must rely on the column default.

    Regression for a code-review finding: the validator used to dereference
    ``record.status.value`` on the error path, which would raise
    AttributeError when ``status`` was None at ``before_insert`` time. The
    Python-level ``default=StagingStatus.PENDING`` populates the attribute
    in time and the defensive None branch keeps the validator robust if a
    caller somehow gets past it.
    """
    sr = StagingRecord(
        source="vlr",
        entity_type=EntityType.PLAYER,
        canonical_id=None,
        payload={"name": "TenZ"},
        # status deliberately omitted — server_default + Python default
        # should both kick in.
    )
    db_session.add(sr)
    db_session.flush()
    db_session.refresh(sr)
    assert sr.status is StagingStatus.PENDING


# --- alias uniqueness race recovery -----------------------------------------


def test_create_path_recovers_from_concurrent_alias_insert(db_session, monkeypatch) -> None:
    """Two workers race past the (platform, platform_id) lookup.

    Simulate the race by monkeypatching ``_lookup_exact_alias`` to miss on
    the first call (the resolver thinks no alias exists and falls into the
    CREATED path), then return the real row on the recovery lookup. The
    savepoint must catch the IntegrityError on the alias insert, roll back
    the orphan entity, and degrade into a MATCHED return for the loser.
    """
    from esports_sim.resolver import core

    first = resolve_entity(
        db_session,
        platform=Platform.VLR,
        platform_id="vlr-race",
        platform_name="Racer",
        entity_type=EntityType.PLAYER,
    )
    assert first.status is ResolutionStatus.CREATED

    real_lookup = core._lookup_exact_alias
    calls = {"n": 0}

    def fake_lookup(session, **kwargs):  # type: ignore[no-untyped-def]
        # Miss only on the very first call (the pre-flight lookup at the top
        # of resolve_entity); subsequent recovery lookups behave normally.
        if calls["n"] == 0:
            calls["n"] += 1
            return None
        return real_lookup(session, **kwargs)

    monkeypatch.setattr(core, "_lookup_exact_alias", fake_lookup)

    second = resolve_entity(
        db_session,
        platform=Platform.VLR,
        platform_id="vlr-race",
        platform_name="Racer",
        entity_type=EntityType.PLAYER,
    )
    assert second.status is ResolutionStatus.MATCHED
    assert second.canonical_id == first.canonical_id

    aliases = (
        db_session.execute(select(EntityAlias).where(EntityAlias.platform_id == "vlr-race"))
        .scalars()
        .all()
    )
    # No orphan entity, no second alias row.
    assert len(aliases) == 1
    entities = (
        db_session.execute(select(Entity).where(Entity.entity_type == EntityType.PLAYER))
        .scalars()
        .all()
    )
    assert len(entities) == 1


def test_auto_merge_path_recovers_from_concurrent_alias_insert(db_session, monkeypatch) -> None:
    """The AUTO_MERGED path must also degrade into MATCHED on a race."""
    from esports_sim.resolver import core

    # Seed an existing canonical so the auto-merge branch fires.
    entity = make_entity(entity_type=EntityType.PLAYER)
    db_session.add(entity)
    db_session.add(
        make_entity_alias(
            entity=entity,
            platform=Platform.VLR,
            platform_id="vlr-tenz",
            platform_name="TenZ",
        )
    )
    db_session.flush()

    # First resolve creates the Liquipedia alias under the existing canonical.
    first = resolve_entity(
        db_session,
        platform=Platform.LIQUIPEDIA,
        platform_id="liq-tenz",
        platform_name="tenz",
        entity_type=EntityType.PLAYER,
    )
    assert first.status is ResolutionStatus.AUTO_MERGED

    # Now make the second call miss the initial exact lookup so it tries to
    # auto-merge again — and crash into the unique constraint we just set.
    real_lookup = core._lookup_exact_alias
    calls = {"n": 0}

    def fake_lookup(session, **kwargs):  # type: ignore[no-untyped-def]
        if calls["n"] == 0:
            calls["n"] += 1
            return None
        return real_lookup(session, **kwargs)

    monkeypatch.setattr(core, "_lookup_exact_alias", fake_lookup)

    second = resolve_entity(
        db_session,
        platform=Platform.LIQUIPEDIA,
        platform_id="liq-tenz",
        platform_name="tenz",
        entity_type=EntityType.PLAYER,
    )
    assert second.status is ResolutionStatus.MATCHED
    assert second.canonical_id == entity.canonical_id


def test_unrelated_integrity_error_is_not_swallowed_by_create_path(db_session, monkeypatch) -> None:
    """A non-race IntegrityError must propagate, not become a fake MATCHED.

    Regression for a code-review finding: the CREATED-path savepoint used
    to catch any IntegrityError. If an unrelated pending object in the
    same session caused flush to fail (e.g. a duplicate
    ``raw_record.content_hash``), the resolver would misclassify it as the
    alias race and return a wrong MATCHED for a row that was never written.
    """
    from esports_sim.resolver import core
    from sqlalchemy.exc import IntegrityError

    def fake_insert_alias(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        # Simulate a different-constraint unique violation. The constraint
        # name is *not* uq_entity_alias_platform_platform_id, so the resolver
        # must re-raise rather than degrade into MATCHED.
        raise IntegrityError(
            "INSERT INTO raw_record",
            {},
            Exception("violates unique constraint uq_raw_record_content_hash"),
        )

    monkeypatch.setattr(core, "_insert_alias", fake_insert_alias)

    with pytest.raises(IntegrityError):
        resolve_entity(
            db_session,
            platform=Platform.VLR,
            platform_id="vlr-fresh",
            platform_name="Fresh",
            entity_type=EntityType.PLAYER,
        )


def test_unrelated_integrity_error_is_not_swallowed_by_auto_merge_path(
    db_session, monkeypatch
) -> None:
    """Same guarantee for the AUTO_MERGED-path recovery branch."""
    from esports_sim.resolver import core
    from sqlalchemy.exc import IntegrityError

    # Seed an existing canonical + alias so AUTO_MERGED is the path under test.
    entity = make_entity(entity_type=EntityType.PLAYER)
    db_session.add(entity)
    db_session.add(
        make_entity_alias(
            entity=entity,
            platform=Platform.VLR,
            platform_id="vlr-tenz",
            platform_name="TenZ",
        )
    )
    db_session.flush()

    def fake_insert_alias(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise IntegrityError(
            "INSERT INTO raw_record",
            {},
            Exception("violates unique constraint uq_raw_record_content_hash"),
        )

    monkeypatch.setattr(core, "_insert_alias", fake_insert_alias)

    with pytest.raises(IntegrityError):
        resolve_entity(
            db_session,
            platform=Platform.LIQUIPEDIA,
            platform_id="liq-tenz",
            platform_name="tenz",
            entity_type=EntityType.PLAYER,
        )
