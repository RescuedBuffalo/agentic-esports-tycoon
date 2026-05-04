"""Integration tests for the BUF-26 teammate-edge bootstrap.

Skipped automatically when ``TEST_DATABASE_URL`` is not set — see
``conftest.py``. The bootstrap reads ``MapResult`` /
``PlayerMatchStat`` / ``Match`` rows seeded inline by these tests
(no fixture corpus dependency) and asserts the BUF-26 acceptance:
*every pair of former teammates in the DB has a teammate or
ex-teammate edge*.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from data_pipeline.seeds.relationships import (
    RECENT_TEAMMATE_DAYS,
    bootstrap_teammate_edges,
)
from esports_sim.db.enums import EntityType, RelationshipEdgeType
from esports_sim.db.models import (
    Entity,
    MapResult,
    Match,
    PlayerMatchStat,
    RelationshipEdge,
    RelationshipEvent,
)
from sqlalchemy import select

pytestmark = pytest.mark.integration


def _make_player(db_session) -> Entity:
    e = Entity(canonical_id=uuid.uuid4(), entity_type=EntityType.PLAYER)
    db_session.add(e)
    return e


def _seed_map(
    db_session,
    *,
    match_date: datetime,
    team1_players: list[Entity],
    team2_players: list[Entity],
    vlr_match_id: str | None = None,
) -> uuid.UUID:
    """Seed one match + one map with the supplied rosters per side."""
    match = Match(
        match_id=uuid.uuid4(),
        vlr_match_id=vlr_match_id or f"m-{uuid.uuid4().hex[:10]}",
        match_date=match_date,
    )
    db_session.add(match)
    db_session.flush()

    mr = MapResult(
        map_result_id=uuid.uuid4(),
        match_id=match.match_id,
        vlr_game_id=f"g-{uuid.uuid4().hex[:10]}",
        vlr_map_id=1,
        team1_rounds=13,
        team2_rounds=10,
        team1_atk_rounds=7,
        team1_def_rounds=6,
        team2_atk_rounds=4,
        team2_def_rounds=6,
        team1_stats={},
        team2_stats={},
    )
    db_session.add(mr)
    db_session.flush()

    for side, roster in (("team1", team1_players), ("team2", team2_players)):
        for player in roster:
            db_session.add(
                PlayerMatchStat(
                    player_match_stat_id=uuid.uuid4(),
                    map_result_id=mr.map_result_id,
                    entity_id=player.canonical_id,
                    source="vlr",
                    source_player_id=str(player.canonical_id),
                    team_side=side,
                )
            )
    db_session.flush()
    return mr.map_result_id


def _canonical_pair(a: Entity, b: Entity) -> tuple[uuid.UUID, uuid.UUID]:
    if a.canonical_id < b.canonical_id:
        return (a.canonical_id, b.canonical_id)
    return (b.canonical_id, a.canonical_id)


def test_bootstrap_emits_one_edge_per_former_teammate_pair(db_session) -> None:
    """Acceptance pin: every pair of former teammates has an edge."""
    p1 = _make_player(db_session)
    p2 = _make_player(db_session)
    p3 = _make_player(db_session)
    p4 = _make_player(db_session)
    p5 = _make_player(db_session)
    opp_a = _make_player(db_session)
    opp_b = _make_player(db_session)
    opp_c = _make_player(db_session)
    opp_d = _make_player(db_session)
    opp_e = _make_player(db_session)
    db_session.flush()

    now = datetime(2026, 5, 1, tzinfo=UTC)

    _seed_map(
        db_session,
        match_date=now - timedelta(days=10),
        team1_players=[p1, p2, p3, p4, p5],
        team2_players=[opp_a, opp_b, opp_c, opp_d, opp_e],
    )

    manifest = bootstrap_teammate_edges(db_session, reference_timestamp=now)

    # 5 teammate roster -> C(5, 2) = 10 internal pairs per side -> 20 total.
    assert manifest.pairs_seen == 20
    assert manifest.total_edges == 20

    edges = db_session.execute(select(RelationshipEdge)).scalars().all()
    assert len(edges) == 20

    # Every directed pair we expect appears exactly once, canonicalised
    # so the unordered pair is keyed unambiguously.
    expected_pairs = set()
    for roster in ([p1, p2, p3, p4, p5], [opp_a, opp_b, opp_c, opp_d, opp_e]):
        for i in range(len(roster)):
            for j in range(i + 1, len(roster)):
                expected_pairs.add(_canonical_pair(roster[i], roster[j]))

    actual_pairs = {(e.src_id, e.dst_id) for e in edges}
    assert actual_pairs == expected_pairs


def test_bootstrap_distinguishes_recent_teammate_from_ex_teammate(db_session) -> None:
    """Recent shared maps -> TEAMMATE; older shared maps -> EX_TEAMMATE."""
    current_a = _make_player(db_session)
    current_b = _make_player(db_session)
    historic_a = _make_player(db_session)
    historic_b = _make_player(db_session)
    db_session.flush()

    now = datetime(2026, 5, 1, tzinfo=UTC)

    # current pair: well within the recency threshold.
    _seed_map(
        db_session,
        match_date=now - timedelta(days=5),
        team1_players=[current_a, current_b],
        team2_players=[],
    )
    # historic pair: pushed past the cutoff so they fall into EX_TEAMMATE.
    _seed_map(
        db_session,
        match_date=now - timedelta(days=RECENT_TEAMMATE_DAYS + 30),
        team1_players=[historic_a, historic_b],
        team2_players=[],
    )

    manifest = bootstrap_teammate_edges(db_session, reference_timestamp=now)

    assert manifest.teammate_edges_inserted == 1
    assert manifest.ex_teammate_edges_inserted == 1

    by_kind: dict[RelationshipEdgeType, list[RelationshipEdge]] = {}
    for edge in db_session.execute(select(RelationshipEdge)).scalars().all():
        by_kind.setdefault(edge.edge_type, []).append(edge)

    assert RelationshipEdgeType.TEAMMATE in by_kind
    assert RelationshipEdgeType.EX_TEAMMATE in by_kind

    teammate_pair = {
        by_kind[RelationshipEdgeType.TEAMMATE][0].src_id,
        by_kind[RelationshipEdgeType.TEAMMATE][0].dst_id,
    }
    assert teammate_pair == {current_a.canonical_id, current_b.canonical_id}


def test_bootstrap_canonicalises_symmetric_pair_endpoints(db_session) -> None:
    """For symmetric kinds the migration's CHECK enforces ``src < dst``;
    the bootstrap canonicalises before insert so that constraint never
    trips."""
    a = _make_player(db_session)
    b = _make_player(db_session)
    db_session.flush()

    _seed_map(
        db_session,
        match_date=datetime(2026, 4, 1, tzinfo=UTC),
        team1_players=[a, b],
        team2_players=[],
    )

    bootstrap_teammate_edges(db_session, reference_timestamp=datetime(2026, 5, 1, tzinfo=UTC))

    edge = db_session.execute(select(RelationshipEdge)).scalar_one()
    assert edge.src_id < edge.dst_id


def test_bootstrap_strength_saturates_with_more_shared_maps(db_session) -> None:
    """A pair that played a single map together has a smaller strength
    than one with a season's worth of co-play. The saturating linear
    formula caps at 1.0."""
    one_off_a = _make_player(db_session)
    one_off_b = _make_player(db_session)
    veteran_a = _make_player(db_session)
    veteran_b = _make_player(db_session)
    db_session.flush()

    now = datetime(2026, 5, 1, tzinfo=UTC)

    # one shared map for the one-off pair.
    _seed_map(
        db_session,
        match_date=now - timedelta(days=10),
        team1_players=[one_off_a, one_off_b],
        team2_players=[],
    )
    # 30 shared maps for the veteran pair (well past saturation).
    for i in range(30):
        _seed_map(
            db_session,
            match_date=now - timedelta(days=10 + i),
            team1_players=[veteran_a, veteran_b],
            team2_players=[],
        )

    bootstrap_teammate_edges(db_session, reference_timestamp=now)

    edges_by_pair = {
        frozenset({e.src_id, e.dst_id}): e
        for e in db_session.execute(select(RelationshipEdge)).scalars().all()
    }
    one_off_edge = edges_by_pair[frozenset({one_off_a.canonical_id, one_off_b.canonical_id})]
    veteran_edge = edges_by_pair[frozenset({veteran_a.canonical_id, veteran_b.canonical_id})]

    assert one_off_edge.strength < veteran_edge.strength
    assert veteran_edge.strength == pytest.approx(1.0)


def test_bootstrap_is_idempotent(db_session) -> None:
    """Re-running the seed on an already-populated DB updates in place
    rather than colliding on the unique constraint."""
    a = _make_player(db_session)
    b = _make_player(db_session)
    db_session.flush()

    _seed_map(
        db_session,
        match_date=datetime(2026, 4, 1, tzinfo=UTC),
        team1_players=[a, b],
        team2_players=[],
    )

    first = bootstrap_teammate_edges(
        db_session, reference_timestamp=datetime(2026, 5, 1, tzinfo=UTC)
    )
    second = bootstrap_teammate_edges(
        db_session, reference_timestamp=datetime(2026, 5, 1, tzinfo=UTC)
    )

    assert first.total_edges == 1
    assert second.total_edges == 0  # nothing new inserted
    assert second.edges_updated == 1

    edges = db_session.execute(select(RelationshipEdge)).scalars().all()
    assert len(edges) == 1


def test_bootstrap_handles_kind_flip_on_rerun(db_session) -> None:
    """If a fresh ingest moves a pair's most-recent shared map into the
    recent window, a re-run swaps the edge from ex-teammate to
    teammate without colliding on the unique constraint."""
    a = _make_player(db_session)
    b = _make_player(db_session)
    db_session.flush()

    older = datetime(2026, 1, 1, tzinfo=UTC)
    refresh_ref = datetime(2026, 5, 1, tzinfo=UTC)

    # First seed: only an old map, so the pair lands as ex-teammate.
    _seed_map(
        db_session,
        match_date=older,
        team1_players=[a, b],
        team2_players=[],
    )
    bootstrap_teammate_edges(db_session, reference_timestamp=refresh_ref)
    initial = db_session.execute(select(RelationshipEdge)).scalar_one()
    assert initial.edge_type is RelationshipEdgeType.EX_TEAMMATE

    # Fresh map within the recency window arrives.
    _seed_map(
        db_session,
        match_date=refresh_ref - timedelta(days=5),
        team1_players=[a, b],
        team2_players=[],
    )
    bootstrap_teammate_edges(db_session, reference_timestamp=refresh_ref)

    after = db_session.execute(select(RelationshipEdge)).scalar_one()
    assert after.edge_type is RelationshipEdgeType.TEAMMATE


def test_bootstrap_rerun_preserves_post_bootstrap_sentiment(db_session) -> None:
    """Codex P1 (PR #28): an event-driven writer may have evolved the
    valence past the +1.0 bootstrap default. A re-run must refresh
    strength + audit metadata without clobbering that learned
    sentiment."""
    a = _make_player(db_session)
    b = _make_player(db_session)
    db_session.flush()

    _seed_map(
        db_session,
        match_date=datetime(2026, 4, 1, tzinfo=UTC),
        team1_players=[a, b],
        team2_players=[],
    )
    bootstrap_teammate_edges(db_session, reference_timestamp=datetime(2026, 5, 1, tzinfo=UTC))

    edge = db_session.execute(select(RelationshipEdge)).scalar_one()
    # Simulate a downstream event-driven writer evolving the valence
    # below the bootstrap default (a feud broke out post-bootstrap).
    edge.sentiment = -0.4
    db_session.flush()

    bootstrap_teammate_edges(db_session, reference_timestamp=datetime(2026, 5, 1, tzinfo=UTC))
    refreshed = db_session.execute(select(RelationshipEdge)).scalar_one()
    assert refreshed.sentiment == pytest.approx(-0.4)


def test_bootstrap_kind_flip_preserves_event_audit_trail(db_session) -> None:
    """Codex P1 (PR #28): ``relationship_event`` cascades on the parent
    edge's delete. A naive delete-and-reinsert on a kind flip would
    wipe the append-only audit log; the in-place mutation keeps the
    history."""
    a = _make_player(db_session)
    b = _make_player(db_session)
    db_session.flush()

    older = datetime(2026, 1, 1, tzinfo=UTC)
    refresh_ref = datetime(2026, 5, 1, tzinfo=UTC)

    _seed_map(
        db_session,
        match_date=older,
        team1_players=[a, b],
        team2_players=[],
    )
    bootstrap_teammate_edges(db_session, reference_timestamp=refresh_ref)
    initial = db_session.execute(select(RelationshipEdge)).scalar_one()
    assert initial.edge_type is RelationshipEdgeType.EX_TEAMMATE

    # An event-driven writer logs a post-bootstrap signal against this
    # edge — exactly the audit-trail row Codex's review flagged.
    db_session.add(
        RelationshipEvent(
            event_id=uuid.uuid4(),
            edge_id=initial.edge_id,
            event_kind="public_feud",
            delta_strength=0.0,
            delta_sentiment=-0.5,
            occurred_at=older + timedelta(days=30),
            payload={"source": "test"},
        )
    )
    db_session.flush()

    # Fresh recent map flips the kind; the event log must survive.
    _seed_map(
        db_session,
        match_date=refresh_ref - timedelta(days=5),
        team1_players=[a, b],
        team2_players=[],
    )
    bootstrap_teammate_edges(db_session, reference_timestamp=refresh_ref)

    after = db_session.execute(select(RelationshipEdge)).scalar_one()
    assert after.edge_type is RelationshipEdgeType.TEAMMATE
    # The event_id-stable check: the row that was attached to the
    # ex-teammate edge is now attached to the same (mutated) edge —
    # we keep the same edge_id so the audit trail is contiguous.
    events = db_session.execute(select(RelationshipEvent)).scalars().all()
    assert len(events) == 1
    assert events[0].event_kind == "public_feud"
    assert events[0].edge_id == after.edge_id


def test_bootstrap_consolidation_path_preserves_event_audit_trail(db_session) -> None:
    """Codex P1 (2nd round, PR #28): defensive case where both teammate
    kinds happen to coexist for the same pair. Re-parenting must move
    the stale row's events onto the target *at the ORM level*; a bulk
    SQL UPDATE alone leaves the in-memory ``stale.events`` collection
    populated and the cascade-delete on ``stale`` would still wipe the
    audit trail."""
    a = _make_player(db_session)
    b = _make_player(db_session)
    db_session.flush()
    src_id, dst_id = (
        (a.canonical_id, b.canonical_id)
        if a.canonical_id < b.canonical_id
        else (b.canonical_id, a.canonical_id)
    )

    # Hand-construct the inconsistent state: both a TEAMMATE and an
    # EX_TEAMMATE row for the same pair, each with its own event log.
    teammate_edge = RelationshipEdge(
        edge_id=uuid.uuid4(),
        src_id=src_id,
        dst_id=dst_id,
        edge_type=RelationshipEdgeType.TEAMMATE,
        strength=0.5,
        sentiment=0.2,
        last_updated_at=datetime(2026, 4, 1, tzinfo=UTC),
        extra={},
    )
    ex_teammate_edge = RelationshipEdge(
        edge_id=uuid.uuid4(),
        src_id=src_id,
        dst_id=dst_id,
        edge_type=RelationshipEdgeType.EX_TEAMMATE,
        strength=0.3,
        sentiment=-0.4,
        last_updated_at=datetime(2026, 1, 1, tzinfo=UTC),
        extra={},
    )
    db_session.add_all([teammate_edge, ex_teammate_edge])
    db_session.flush()

    db_session.add_all(
        [
            RelationshipEvent(
                event_id=uuid.uuid4(),
                edge_id=ex_teammate_edge.edge_id,
                event_kind="legacy_feud",
                delta_strength=0.0,
                delta_sentiment=-0.4,
                occurred_at=datetime(2026, 2, 1, tzinfo=UTC),
                payload={},
            ),
            RelationshipEvent(
                event_id=uuid.uuid4(),
                edge_id=teammate_edge.edge_id,
                event_kind="recent_clutch",
                delta_strength=0.1,
                delta_sentiment=0.0,
                occurred_at=datetime(2026, 4, 15, tzinfo=UTC),
                payload={},
            ),
        ]
    )
    db_session.flush()

    # Drive the bootstrap with a recent shared map so it converges on
    # the TEAMMATE kind and has to consolidate the stale EX_TEAMMATE row.
    refresh_ref = datetime(2026, 5, 1, tzinfo=UTC)
    _seed_map(
        db_session,
        match_date=refresh_ref - timedelta(days=5),
        team1_players=[a, b],
        team2_players=[],
    )
    bootstrap_teammate_edges(db_session, reference_timestamp=refresh_ref)
    db_session.flush()

    # Only the target row survives, and both events are attached to it.
    edges = db_session.execute(select(RelationshipEdge)).scalars().all()
    assert len(edges) == 1
    assert edges[0].edge_type is RelationshipEdgeType.TEAMMATE
    assert edges[0].edge_id == teammate_edge.edge_id

    events = db_session.execute(select(RelationshipEvent)).scalars().all()
    assert {e.event_kind for e in events} == {"legacy_feud", "recent_clutch"}
    assert all(e.edge_id == teammate_edge.edge_id for e in events)


def test_bootstrap_rerun_preserves_event_evolved_strength_and_anchor(db_session) -> None:
    """Codex P1 (PR #28, 3rd round): once a post-bootstrap writer has
    advanced ``last_updated_at`` past the match-history baseline (an
    event was folded in, or the monthly decay job ran), the rerun
    must NOT re-derive ``strength`` from shared-map count or rewind
    the anchor — both moves would wipe the writer's adjustments and
    cause the next decay pass to over-apply elapsed weeks."""
    a = _make_player(db_session)
    b = _make_player(db_session)
    db_session.flush()

    last_match_at = datetime(2026, 4, 1, tzinfo=UTC)
    _seed_map(
        db_session,
        match_date=last_match_at,
        team1_players=[a, b],
        team2_players=[],
    )
    bootstrap_teammate_edges(db_session, reference_timestamp=datetime(2026, 5, 1, tzinfo=UTC))
    edge = db_session.execute(select(RelationshipEdge)).scalar_one()

    # Simulate a post-bootstrap event-driven writer: a delta_strength
    # was folded in and the anchor was advanced past the match-history
    # baseline (the contract on ``last_updated_at`` per the model
    # docstring).
    bumped_strength = 0.85
    bumped_anchor = last_match_at + timedelta(days=7)
    edge.strength = bumped_strength
    edge.last_updated_at = bumped_anchor
    db_session.flush()

    bootstrap_teammate_edges(db_session, reference_timestamp=datetime(2026, 5, 1, tzinfo=UTC))
    refreshed = db_session.execute(select(RelationshipEdge)).scalar_one()
    assert refreshed.strength == pytest.approx(bumped_strength)
    assert refreshed.last_updated_at == bumped_anchor


def test_bootstrap_rerun_refreshes_strength_when_no_event_writer_has_taken_over(
    db_session,
) -> None:
    """Counterpart to the preservation test: if the existing edge is
    still anchored at the bootstrap baseline (``last_updated_at <=
    acc.last_match_at``), a fresh ingest with more shared maps should
    legitimately refresh ``strength``."""
    a = _make_player(db_session)
    b = _make_player(db_session)
    db_session.flush()

    early_match = datetime(2026, 4, 1, tzinfo=UTC)
    _seed_map(
        db_session,
        match_date=early_match,
        team1_players=[a, b],
        team2_players=[],
    )
    bootstrap_teammate_edges(db_session, reference_timestamp=datetime(2026, 5, 1, tzinfo=UTC))
    initial = db_session.execute(select(RelationshipEdge)).scalar_one()
    initial_strength = initial.strength

    # Fresh ingest brings more shared maps for the same pair; the
    # event-driven path has not touched the edge.
    for i in range(5):
        _seed_map(
            db_session,
            match_date=early_match + timedelta(days=i + 1),
            team1_players=[a, b],
            team2_players=[],
        )

    bootstrap_teammate_edges(db_session, reference_timestamp=datetime(2026, 5, 1, tzinfo=UTC))
    refreshed = db_session.execute(select(RelationshipEdge)).scalar_one()
    assert refreshed.strength > initial_strength
    # Anchor advances to the latest shared map.
    assert refreshed.last_updated_at == early_match + timedelta(days=5)
