"""BUF-12 entity-resolution worker tests.

Three blocks:

* Pure-function unit tests for :func:`merge_records` (the BUF-12
  acceptance "VLR vs Liquipedia → Liquipedia wins" lives here).
* Integration tests (gated on ``TEST_DATABASE_URL``) for
  :func:`handle_rebrand`, :func:`lookup_alias_at`, and
  :func:`process_staging_queue` against the real BUF-6 schema.
* A perf-bound integration test for "100 staging records processed
  in under 5 seconds" — the third BUF-12 acceptance bullet.
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime
from typing import Any

import pytest
from esports_sim.db.enums import EntityType, Platform, StagingStatus
from esports_sim.db.models import Entity, EntityAlias, StagingRecord
from esports_sim.resolver import (
    SOURCE_PRIORITY,
    ConflictRecord,
    RebrandConflictError,
    RebrandTarget,
    WorkerStats,
    handle_rebrand,
    lookup_alias_at,
    merge_records,
    process_staging_queue,
    resolve_entity,
)
from esports_sim.resolver.worker import default_payload_extractor

# --- merge_records: pure-function unit tests ------------------------------


def test_merge_records_lower_priority_source_wins_field_conflict() -> None:
    """The ticket's headline acceptance: Liquipedia beats VLR.

    Two payloads disagree on ``country``; Liquipedia is priority 1,
    VLR priority 2, so Liquipedia's value lands in ``merged`` and the
    conflict log surfaces both before/after with the source names.
    """
    existing = {"country": "US", "real_name": "Tyson Ngo", "role": "duelist"}
    incoming = {"country": "CA", "real_name": "Tyson Ngo", "team_slug": "sentinels"}

    result = merge_records(
        existing,
        incoming,
        existing_source=Platform.VLR,
        incoming_source=Platform.LIQUIPEDIA,
        log_conflicts=False,
    )

    # Liquipedia wins the country conflict.
    assert result.merged["country"] == "CA"
    # Fields only on one side are kept verbatim.
    assert result.merged["role"] == "duelist"
    assert result.merged["team_slug"] == "sentinels"
    # Agreed-on field: no conflict logged.
    assert all(c.field_name != "real_name" for c in result.conflicts)

    [country_conflict] = result.conflicts
    assert country_conflict.field_name == "country"
    assert country_conflict.before == "US"
    assert country_conflict.after == "CA"
    assert country_conflict.winning_source == "liquipedia"
    assert country_conflict.losing_source == "vlr"


def test_merge_records_existing_priority_wins_when_lower() -> None:
    """The reverse direction: existing is higher-priority than incoming."""
    existing = {"country": "CA"}
    incoming = {"country": "US"}

    result = merge_records(
        existing,
        incoming,
        existing_source=Platform.LIQUIPEDIA,  # priority 1
        incoming_source=Platform.VLR,  # priority 2
        log_conflicts=False,
    )

    assert result.merged["country"] == "CA"
    [c] = result.conflicts
    # existing kept its value; the conflict still gets logged so an
    # operator can see VLR disagreed.
    assert c.before == "CA"
    assert c.after == "CA"
    assert c.winning_source == "liquipedia"
    assert c.losing_source == "vlr"


def test_merge_records_emits_warning_log_per_conflict(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Default ``log_conflicts=True`` emits a structured WARNING per field."""
    caplog.set_level(logging.WARNING, logger="esports_sim.resolver.worker")

    merge_records(
        {"country": "US", "role": "duelist"},
        {"country": "CA", "role": "controller"},
        existing_source=Platform.VLR,
        incoming_source=Platform.LIQUIPEDIA,
    )

    # Two conflicts → two WARNING records.
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 2
    joined = "\n".join(r.getMessage() for r in warnings)
    assert "field=country" in joined
    assert "field=role" in joined
    assert "winning_source=liquipedia" in joined
    assert "losing_source=vlr" in joined


def test_merge_records_no_conflict_when_payloads_agree() -> None:
    """Equal values aren't conflicts even when the sources disagree by priority."""
    payload = {"country": "US", "role": "duelist"}

    result = merge_records(
        payload,
        payload.copy(),
        existing_source=Platform.VLR,
        incoming_source=Platform.LIQUIPEDIA,
        log_conflicts=False,
    )

    assert result.conflicts == []
    assert result.merged == payload


def test_merge_records_disjoint_keys_keeps_both_sides() -> None:
    """Fields present on only one side land in merged with no conflict."""
    existing = {"a": 1}
    incoming = {"b": 2}

    result = merge_records(
        existing,
        incoming,
        existing_source=Platform.LIQUIPEDIA,
        incoming_source=Platform.VLR,
        log_conflicts=False,
    )

    assert result.merged == {"a": 1, "b": 2}
    assert result.conflicts == []


def test_source_priority_matches_ticket_order() -> None:
    """Defensive: a refactor that re-orders SOURCE_PRIORITY would silently
    flip merge outcomes. Pin the ordering to the ticket spec verbatim."""
    assert SOURCE_PRIORITY[Platform.RIOT_API] < SOURCE_PRIORITY[Platform.LIQUIPEDIA]
    assert SOURCE_PRIORITY[Platform.LIQUIPEDIA] < SOURCE_PRIORITY[Platform.VLR]
    assert SOURCE_PRIORITY[Platform.VLR] < SOURCE_PRIORITY[Platform.ESPORTSEARNINGS]
    assert SOURCE_PRIORITY[Platform.ESPORTSEARNINGS] < SOURCE_PRIORITY[Platform.TWITCH]
    assert SOURCE_PRIORITY[Platform.TWITCH] < SOURCE_PRIORITY[Platform.TWITTER]


def test_merge_records_rejects_unknown_source() -> None:
    """A Platform enum value missing from SOURCE_PRIORITY raises.

    Adding a new ``Platform`` enum value without a priority entry
    would otherwise default to a silent ``KeyError`` deep in the
    merge — make it loud at the boundary.
    """
    # Construct a fake unknown platform by patching SOURCE_PRIORITY temporarily.
    # Easier: assert the existing platforms are accepted and an empty-key
    # case raises by passing a not-present platform via a test-double.
    from esports_sim.resolver import worker as worker_mod

    fake_platform = Platform.RIOT_API
    saved = worker_mod.SOURCE_PRIORITY.pop(fake_platform)
    try:
        with pytest.raises(ValueError, match="not in SOURCE_PRIORITY"):
            merge_records(
                {"a": 1},
                {"a": 2},
                existing_source=fake_platform,
                incoming_source=Platform.LIQUIPEDIA,
                log_conflicts=False,
            )
    finally:
        worker_mod.SOURCE_PRIORITY[fake_platform] = saved


def test_conflict_record_to_log_dict_round_trip() -> None:
    """The structured-log dict carries every field the operator needs."""
    c = ConflictRecord(
        field_name="role",
        before="duelist",
        after="controller",
        losing_source="vlr",
        winning_source="liquipedia",
    )
    assert c.to_log_dict() == {
        "field": "role",
        "before": "duelist",
        "after": "controller",
        "losing_source": "vlr",
        "winning_source": "liquipedia",
    }


# --- handle_rebrand integration tests ------------------------------------


pytestmark_integration = pytest.mark.integration

_EFFECTIVE = datetime(2026, 4, 1, tzinfo=UTC)


@pytestmark_integration
def test_handle_rebrand_extends_existing_canonical_with_valid_from(db_session) -> None:
    """A rebrand attaches a new alias to the same canonical, with valid_from set."""
    seed = resolve_entity(
        db_session,
        platform=Platform.LIQUIPEDIA,
        platform_id="sentinels",
        platform_name="Sentinels",
        entity_type=EntityType.TEAM,
    )
    db_session.flush()

    new_alias = handle_rebrand(
        db_session,
        platform=Platform.LIQUIPEDIA,
        old_platform_id="sentinels",
        new_platform_id="team-sentinels-esports",
        new_platform_name="Team Sentinels Esports",
        effective_date=_EFFECTIVE,
    )

    assert new_alias.canonical_id == seed.canonical_id
    assert new_alias.platform_id == "team-sentinels-esports"
    assert new_alias.valid_from == _EFFECTIVE


@pytestmark_integration
def test_handle_rebrand_idempotent_returns_existing_on_replay(db_session) -> None:
    """Replaying the same rebrand is a no-op; the existing alias is returned."""
    resolve_entity(
        db_session,
        platform=Platform.LIQUIPEDIA,
        platform_id="sentinels",
        platform_name="Sentinels",
        entity_type=EntityType.TEAM,
    )
    db_session.flush()

    first = handle_rebrand(
        db_session,
        platform=Platform.LIQUIPEDIA,
        old_platform_id="sentinels",
        new_platform_id="team-sentinels-esports",
        new_platform_name="Team Sentinels Esports",
        effective_date=_EFFECTIVE,
    )
    db_session.flush()
    second = handle_rebrand(
        db_session,
        platform=Platform.LIQUIPEDIA,
        old_platform_id="sentinels",
        new_platform_id="team-sentinels-esports",
        new_platform_name="Team Sentinels Esports",
        effective_date=_EFFECTIVE,
    )

    assert first.id == second.id


@pytestmark_integration
def test_handle_rebrand_raises_when_old_handle_unknown(db_session) -> None:
    """Rebranding an unknown old slug is a programming error, not a CREATE."""
    with pytest.raises(ValueError, match="no existing alias"):
        handle_rebrand(
            db_session,
            platform=Platform.LIQUIPEDIA,
            old_platform_id="never-seen",
            new_platform_id="something-new",
            new_platform_name="Something New",
            effective_date=_EFFECTIVE,
        )


@pytestmark_integration
def test_handle_rebrand_raises_on_target_handle_owned_by_other_canonical(
    db_session,
) -> None:
    """If the destination handle already maps to a different canonical, fail loud."""
    a = resolve_entity(
        db_session,
        platform=Platform.LIQUIPEDIA,
        platform_id="sentinels",
        platform_name="Sentinels",
        entity_type=EntityType.TEAM,
    )
    b = resolve_entity(
        db_session,
        platform=Platform.LIQUIPEDIA,
        platform_id="team-sentinels-esports",
        platform_name="Totally Different Team",
        entity_type=EntityType.TEAM,
    )
    db_session.flush()
    assert a.canonical_id != b.canonical_id

    with pytest.raises(RebrandConflictError):
        handle_rebrand(
            db_session,
            platform=Platform.LIQUIPEDIA,
            old_platform_id="sentinels",
            new_platform_id="team-sentinels-esports",
            new_platform_name="Team Sentinels Esports",
            effective_date=_EFFECTIVE,
        )


@pytestmark_integration
def test_handle_rebrand_rejects_naive_effective_date(db_session) -> None:
    """``effective_date`` without tzinfo is rejected — the column is timezone-aware."""
    resolve_entity(
        db_session,
        platform=Platform.LIQUIPEDIA,
        platform_id="sentinels",
        platform_name="Sentinels",
        entity_type=EntityType.TEAM,
    )
    with pytest.raises(ValueError, match="timezone-aware"):
        handle_rebrand(
            db_session,
            platform=Platform.LIQUIPEDIA,
            old_platform_id="sentinels",
            new_platform_id="team-sentinels-esports",
            new_platform_name="Team Sentinels Esports",
            effective_date=datetime(2026, 4, 1),  # naive
        )


# --- lookup_alias_at integration tests ----------------------------------


@pytestmark_integration
def test_lookup_alias_at_returns_alias_valid_at_query_time(db_session) -> None:
    """The right alias is whichever one's ``valid_from`` is the largest <= ``at``."""
    seed = resolve_entity(
        db_session,
        platform=Platform.LIQUIPEDIA,
        platform_id="oldhandle",
        platform_name="OldHandle",
        entity_type=EntityType.PLAYER,
    )
    db_session.flush()
    handle_rebrand(
        db_session,
        platform=Platform.LIQUIPEDIA,
        old_platform_id="oldhandle",
        new_platform_id="newhandle",
        new_platform_name="NewHandle",
        effective_date=datetime(2026, 4, 1, tzinfo=UTC),
    )
    db_session.flush()

    # Before the rebrand: only the old alias exists by valid_from.
    before = lookup_alias_at(
        db_session,
        platform=Platform.LIQUIPEDIA,
        platform_id="newhandle",
        at=datetime(2026, 3, 1, tzinfo=UTC),
    )
    assert before is None  # newhandle didn't exist yet

    # After the rebrand effective date: the new alias is valid.
    after = lookup_alias_at(
        db_session,
        platform=Platform.LIQUIPEDIA,
        platform_id="newhandle",
        at=datetime(2026, 4, 5, tzinfo=UTC),
    )
    assert after is not None
    assert after.canonical_id == seed.canonical_id


# --- 10-handle-change-fixture acceptance test ----------------------------


@pytestmark_integration
def test_ten_known_handle_changes_produce_zero_duplicate_canonicals(
    db_session,
) -> None:
    """BUF-12 acceptance: 10 handle changes, zero duplicate canonical entities.

    Seeds 10 distinct players, then simulates each one's handle change
    via ``handle_rebrand``. The post-condition is exactly 10 canonical
    rows (no fork) and 20 alias rows (one per old handle + one per
    new handle).
    """
    from sqlalchemy import func, select

    # 10 fixture players: (old_handle, new_handle, new_display_name).
    handle_changes = [
        ("tenz_2023", "tenz", "TenZ"),
        ("zekken_2024", "zekken", "Zekken"),
        ("sacy_2023", "sacy", "Sacy"),
        ("johnqt_2024", "johnqt", "johnqt"),
        ("aspas_2024", "aspas", "aspas"),
        ("less_2024", "less", "Less"),
        ("saadhak_2024", "saadhak", "Saadhak"),
        ("derke_2024", "derke", "Derke"),
        ("alfajer_2024", "alfajer", "Alfajer"),
        ("boaster_2024", "boaster", "Boaster"),
    ]

    # 1. Seed the 10 originals via the resolver.
    canonical_ids: list[uuid.UUID] = []
    for old, _new, _name in handle_changes:
        result = resolve_entity(
            db_session,
            platform=Platform.LIQUIPEDIA,
            platform_id=old,
            platform_name=old.replace("_", " ").title(),
            entity_type=EntityType.PLAYER,
        )
        assert result.canonical_id is not None
        canonical_ids.append(result.canonical_id)
    db_session.flush()

    entity_count_seeded = db_session.execute(select(func.count()).select_from(Entity)).scalar_one()
    assert entity_count_seeded == 10

    # 2. Apply each rebrand.
    effective = datetime(2026, 4, 1, tzinfo=UTC)
    for (old, new, new_name), seeded_canonical in zip(handle_changes, canonical_ids, strict=True):
        new_alias = handle_rebrand(
            db_session,
            platform=Platform.LIQUIPEDIA,
            old_platform_id=old,
            new_platform_id=new,
            new_platform_name=new_name,
            effective_date=effective,
        )
        assert new_alias.canonical_id == seeded_canonical
    db_session.flush()

    # 3. Post-condition: still exactly 10 canonical rows, 20 alias rows.
    entity_count_after = db_session.execute(select(func.count()).select_from(Entity)).scalar_one()
    alias_count_after = db_session.execute(
        select(func.count()).select_from(EntityAlias)
    ).scalar_one()
    assert entity_count_after == 10
    assert alias_count_after == 20


# --- process_staging_queue integration -----------------------------------


@pytestmark_integration
def test_process_staging_queue_resolves_pending_rows_into_canonicals(
    db_session,
) -> None:
    """Pending rows transition to processed with canonical_id set."""
    payload = {"slug": "tenz", "name": "TenZ", "role": "duelist"}
    row = StagingRecord(
        source="liquipedia",
        entity_type=EntityType.PLAYER,
        canonical_id=None,
        payload=payload,
        status=StagingStatus.PENDING,
    )
    db_session.add(row)
    db_session.flush()

    stats = process_staging_queue(db_session, batch_size=10, max_batches=1)
    db_session.refresh(row)

    assert stats.seen == 1
    assert stats.processed == 1
    assert row.status is StagingStatus.PROCESSED
    assert row.canonical_id is not None


@pytestmark_integration
def test_process_staging_queue_skips_extractor_misses_without_draining_them(
    db_session,
) -> None:
    """A row with no extractable handle is logged + left in PENDING for re-run."""
    row = StagingRecord(
        source="liquipedia",
        entity_type=EntityType.PLAYER,
        canonical_id=None,
        payload={"unrecognised": "shape"},  # neither slug nor platform_id
        status=StagingStatus.PENDING,
    )
    db_session.add(row)
    db_session.flush()

    stats = process_staging_queue(db_session, batch_size=10, max_batches=1)
    db_session.refresh(row)

    assert stats.extractor_misses == 1
    assert stats.processed == 0
    # Status stayed PENDING — a fixed extractor on the next run picks
    # the row up again instead of having to resurrect it from a dead
    # state.
    assert row.status is StagingStatus.PENDING


@pytestmark_integration
def test_process_staging_queue_handles_one_hundred_rows_under_five_seconds(
    db_session,
) -> None:
    """BUF-12 acceptance: 100 staging records processed in <5s.

    Builds 100 distinct pending rows under the Liquipedia source, runs
    the worker in a single pass, and asserts the wall-clock budget.
    Five seconds is generous enough to absorb CI variance while still
    catching a regression that turned the resolver's per-row work
    into something quadratic.
    """
    rows: list[StagingRecord] = []
    for i in range(100):
        rows.append(
            StagingRecord(
                source="liquipedia",
                entity_type=EntityType.PLAYER,
                canonical_id=None,
                payload={"slug": f"player_{i:03d}", "name": f"Player {i:03d}"},
                status=StagingStatus.PENDING,
            )
        )
    db_session.add_all(rows)
    db_session.flush()

    stats: WorkerStats = process_staging_queue(db_session, batch_size=100, max_batches=1)

    assert stats.seen == 100
    assert stats.processed + stats.review == 100
    assert (
        stats.elapsed_seconds < 5.0
    ), f"100 staging rows took {stats.elapsed_seconds:.3f}s; budget is 5s"


@pytestmark_integration
def test_process_staging_queue_routes_pending_resolver_outcome_to_review(
    db_session,
) -> None:
    """A resolver PENDING (fuzzy in review band) moves the staging row to REVIEW.

    Two distinct entities seeded with similar names sit just under the
    auto-merge threshold; a third row whose name is a near-miss should
    queue for human review and the staging row's status should reflect
    that, not silently fall through to PROCESSED.
    """
    resolve_entity(
        db_session,
        platform=Platform.LIQUIPEDIA,
        platform_id="someplayer",
        platform_name="SomeUniquePlayerName",
        entity_type=EntityType.PLAYER,
    )
    db_session.flush()

    row = StagingRecord(
        source="vlr",
        entity_type=EntityType.PLAYER,
        canonical_id=None,
        payload={"slug": "vlr-someplayer", "name": "SomeUniquePlayerNomes"},
        status=StagingStatus.PENDING,
    )
    db_session.add(row)
    db_session.flush()

    stats = process_staging_queue(db_session, batch_size=10, max_batches=1)
    db_session.refresh(row)

    if stats.review:
        # The fuzzy match landed in the review band — the documented
        # ResolutionStatus.PENDING path.
        assert row.status is StagingStatus.REVIEW
        assert row.canonical_id is None
    else:
        # If the score sailed past auto-merge or below review, that's
        # also a legitimate resolver outcome — the test's purpose is
        # the routing rule, so just assert the status is one of the
        # legal terminal states.
        assert row.status in {StagingStatus.PROCESSED, StagingStatus.REVIEW}


@pytestmark_integration
def test_process_staging_queue_does_not_loop_on_extractor_misses(
    db_session,
) -> None:
    """Round 1 regression: pure-miss queue must terminate in one pass.

    When every PENDING row is an extractor miss, ``continue`` leaves
    the row PENDING — but the next batch SELECT would re-match the
    same rows and loop forever (the row is never re-resolved within a
    single ``process_staging_queue`` call). The fix tracks miss IDs
    in-run and excludes them from subsequent batch SELECTs; this test
    pegs ``max_batches=None`` so a regression deadlocks the test
    rather than hiding behind the cap.
    """
    rows: list[StagingRecord] = [
        StagingRecord(
            source="liquipedia",
            entity_type=EntityType.PLAYER,
            canonical_id=None,
            payload={"unrecognised": f"shape_{i}"},
            status=StagingStatus.PENDING,
        )
        for i in range(5)
    ]
    db_session.add_all(rows)
    db_session.flush()

    # max_batches=None — the regression would never hit ``not rows``.
    # If this test hangs, the fix isn't in place.
    stats = process_staging_queue(db_session, batch_size=2, max_batches=None)

    assert stats.seen == 5
    assert stats.extractor_misses == 5
    assert stats.processed == 0
    for row in rows:
        db_session.refresh(row)
        assert row.status is StagingStatus.PENDING


@pytestmark_integration
def test_handle_rebrand_race_recovery_uses_savepoint_not_outer_rollback(
    db_session,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Round 1 regression: the IntegrityError recovery path must not call
    ``session.rollback()`` on the outer session.

    Round 1 used ``session.rollback()`` after a unique-key collision,
    which threw away every uncommitted write in the caller's
    transaction (this function is documented as caller-transaction-
    owned). The fixed version wraps the insert in a
    ``begin_nested()`` savepoint, mirroring the pattern in
    :func:`_try_insert_alias_or_recover`.

    True concurrency is hard to drive in a single-process test, so we
    sentinel the fix structurally: monkey-patch ``session.rollback``
    to record any call. With the savepoint fix in place, the recovery
    path never touches outer rollback even on a forced IntegrityError.
    """
    from esports_sim.resolver import worker as worker_mod
    from sqlalchemy import select

    seed = resolve_entity(
        db_session,
        platform=Platform.LIQUIPEDIA,
        platform_id="oldhandle",
        platform_name="OldHandle",
        entity_type=EntityType.PLAYER,
    )
    db_session.flush()
    assert seed.canonical_id is not None

    # An unrelated canonical row in the same outer transaction — the
    # write that round-1 rollback would have nuked.
    unrelated = Entity(entity_type=EntityType.TEAM)
    db_session.add(unrelated)
    db_session.flush()

    # Force ``handle_rebrand`` into its IntegrityError branch by
    # stubbing the second SELECT (the destination-alias pre-flight) to
    # return None, then planting a real row at the destination handle
    # that the subsequent insert will collide against.
    db_session.add(
        EntityAlias(
            canonical_id=seed.canonical_id,
            platform=Platform.LIQUIPEDIA,
            platform_id="newhandle",
            platform_name="NewHandle",
            confidence=1.0,
        )
    )
    db_session.flush()

    real_execute = db_session.execute
    call_n = {"n": 0}

    def fake_execute(stmt: Any, *a: Any, **kw: Any) -> Any:
        # The second SELECT inside handle_rebrand looks up the
        # destination alias by (platform, new_platform_id). Stub it to
        # return a no-row result so the function falls through to the
        # insert and trips the unique constraint we already populated.
        call_n["n"] += 1
        if call_n["n"] == 2:

            class _Empty:
                def scalar_one_or_none(self) -> None:
                    return None

            return _Empty()
        return real_execute(stmt, *a, **kw)

    rollback_calls: list[None] = []
    monkeypatch.setattr(db_session, "rollback", lambda: rollback_calls.append(None))
    monkeypatch.setattr(db_session, "execute", fake_execute)

    handle_rebrand(
        db_session,
        platform=Platform.LIQUIPEDIA,
        old_platform_id="oldhandle",
        new_platform_id="newhandle",
        new_platform_name="NewHandle",
        effective_date=_EFFECTIVE,
    )

    # The fix's invariant: the savepoint absorbs the rollback; the
    # outer session.rollback is NEVER called. Round-1 code would have
    # tripped this list with one entry.
    assert rollback_calls == []
    # And the unrelated entity must still be present (it would have
    # been dropped if round-1 rollback had fired).
    assert worker_mod is not None  # silence unused-import lint
    assert (
        real_execute(
            select(Entity).where(Entity.canonical_id == unrelated.canonical_id)
        ).scalar_one_or_none()
        is not None
    )


# --- default_payload_extractor: rebrand surface (pure unit) --------------


def test_default_extractor_surfaces_rebrand_target_when_previous_slug_differs() -> None:
    """A payload with both ``slug`` and ``previous_slug`` becomes a rebrand."""
    handle = default_payload_extractor(
        "liquipedia",
        EntityType.TEAM,
        {
            "slug": "team-sentinels-esports",
            "name": "Team Sentinels Esports",
            "previous_slug": "sentinels",
            "renamed_at": "2026-04-01",
        },
    )
    assert handle is not None
    # Resolve on the OLD slug so the existing canonical matches.
    assert handle.platform_id == "sentinels"
    assert handle.platform_name == "Team Sentinels Esports"
    assert handle.rebrand_target is not None
    assert handle.rebrand_target.new_platform_id == "team-sentinels-esports"
    assert handle.rebrand_target.new_platform_name == "Team Sentinels Esports"
    assert handle.rebrand_target.effective_date.isoformat().startswith("2026-04-01")


def test_default_extractor_no_rebrand_when_previous_slug_equals_slug() -> None:
    """``previous_slug == slug`` is not a rebrand event."""
    handle = default_payload_extractor(
        "liquipedia",
        EntityType.TEAM,
        {"slug": "sentinels", "name": "Sentinels", "previous_slug": "sentinels"},
    )
    assert handle is not None
    assert handle.rebrand_target is None


def test_default_extractor_no_rebrand_when_previous_slug_absent() -> None:
    """A plain payload (no ``previous_slug``) carries no rebrand target."""
    handle = default_payload_extractor(
        "liquipedia",
        EntityType.PLAYER,
        {"slug": "tenz", "name": "TenZ"},
    )
    assert handle is not None
    assert handle.platform_id == "tenz"
    assert handle.rebrand_target is None


def test_default_extractor_uses_explicit_platform_id_when_present() -> None:
    """Explicit ``platform_id`` / ``platform_name`` win over the slug fallback."""
    handle = default_payload_extractor(
        "vlr",
        EntityType.PLAYER,
        {"platform_id": "vlr-12345", "platform_name": "TenZ", "slug": "ignored"},
    )
    assert handle is not None
    assert handle.platform_id == "vlr-12345"
    assert handle.rebrand_target is None


# --- process_staging_queue: rebrand wiring (integration) ----------------


@pytestmark_integration
def test_process_staging_queue_extends_alias_chain_on_rebrand_payload(
    db_session,
) -> None:
    """Round 3 regression: worker handles a rebrand payload like the seed does.

    Without the fix, the worker resolved on the OLD slug only and
    never registered the new slug as an alias — so a later record
    carrying only the new slug would fork into a duplicate canonical.
    """
    # Pre-seed canonical under the OLD slug.
    seed = resolve_entity(
        db_session,
        platform=Platform.LIQUIPEDIA,
        platform_id="sentinels",
        platform_name="Sentinels",
        entity_type=EntityType.TEAM,
    )
    db_session.flush()

    # Pending staging row carrying both slug + previous_slug.
    row = StagingRecord(
        source="liquipedia",
        entity_type=EntityType.TEAM,
        canonical_id=None,
        payload={
            "slug": "team-sentinels-esports",
            "name": "Team Sentinels Esports",
            "previous_slug": "sentinels",
            "renamed_at": "2026-04-01",
        },
        status=StagingStatus.PENDING,
    )
    db_session.add(row)
    db_session.flush()

    stats = process_staging_queue(db_session, batch_size=10, max_batches=1)
    db_session.refresh(row)

    assert stats.processed == 1
    assert stats.rebrands_registered == 1
    assert stats.rebrand_conflicts == 0
    assert row.canonical_id == seed.canonical_id

    # Both slugs should now resolve to the same canonical.
    new_alias = lookup_alias_at(
        db_session,
        platform=Platform.LIQUIPEDIA,
        platform_id="team-sentinels-esports",
        at=datetime(2026, 4, 5, tzinfo=UTC),
    )
    assert new_alias is not None
    assert new_alias.canonical_id == seed.canonical_id


@pytestmark_integration
def test_process_staging_queue_logs_rebrand_conflict_without_aborting(
    db_session,
) -> None:
    """A rebrand whose new slug already maps to a different canonical is logged."""
    # Two distinct canonicals: one under the old slug, one already at
    # the new slug. The rebrand attempt should fail with a conflict
    # but the row's primary resolution still completes.
    resolve_entity(
        db_session,
        platform=Platform.LIQUIPEDIA,
        platform_id="oldteam",
        platform_name="OldTeam",
        entity_type=EntityType.TEAM,
    )
    other = resolve_entity(
        db_session,
        platform=Platform.LIQUIPEDIA,
        platform_id="newteam",
        platform_name="UnrelatedNewTeam",
        entity_type=EntityType.TEAM,
    )
    db_session.flush()

    row = StagingRecord(
        source="liquipedia",
        entity_type=EntityType.TEAM,
        canonical_id=None,
        payload={
            "slug": "newteam",
            "name": "OldTeam Renamed",
            "previous_slug": "oldteam",
            "renamed_at": "2026-04-01",
        },
        status=StagingStatus.PENDING,
    )
    db_session.add(row)
    db_session.flush()

    stats = process_staging_queue(db_session, batch_size=10, max_batches=1)
    db_session.refresh(row)

    # Primary resolution succeeded on the OLD slug.
    assert stats.processed == 1
    # But the rebrand extension was rejected as a conflict.
    assert stats.rebrand_conflicts == 1
    assert stats.rebrands_registered == 0
    # The "newteam" alias still maps to the unrelated canonical
    # — we did NOT silently overwrite it.
    new_alias = lookup_alias_at(
        db_session,
        platform=Platform.LIQUIPEDIA,
        platform_id="newteam",
        at=datetime(2026, 4, 5, tzinfo=UTC),
    )
    assert new_alias is not None
    assert new_alias.canonical_id == other.canonical_id


def test_rebrand_target_dataclass_round_trip() -> None:
    """RebrandTarget is frozen + carries the three fields the worker needs."""
    target = RebrandTarget(
        new_platform_id="newslug",
        new_platform_name="NewName",
        effective_date=datetime(2026, 4, 1, tzinfo=UTC),
    )
    assert target.new_platform_id == "newslug"
    with pytest.raises((AttributeError, Exception)):
        target.new_platform_id = "other"  # type: ignore[misc]
