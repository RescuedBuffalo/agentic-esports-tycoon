"""``bootstrap_teammate_edges()`` — one-shot teammate-graph seed (BUF-26).

Walks the canonical match history (BUF-8 v2 ``MapResult`` rows joined
to per-map participation in ``PlayerMatchStat``) and emits one
:class:`RelationshipEdge` per pair of canonical players that ever
shared a team_side on the same map.

Per the BUF-26 acceptance: *every pair of former teammates in the DB
has a teammate or ex-teammate edge*. The cut between the two kinds is
recency:

* If the pair's most-recent shared map is within
  :data:`RECENT_TEAMMATE_DAYS` of the bootstrap reference timestamp,
  the edge is :attr:`RelationshipEdgeType.TEAMMATE`.
* Otherwise the pair carries the historical :attr:`RelationshipEdgeType.EX_TEAMMATE`.

The bootstrap reference timestamp is the maximum ``match_date`` in the
DB — the closest analogue to "now" against the seed corpus. This
matches what the steady-state ecosystem layer would do at world-tick
0: replay the historical relationships into the world before the sim
starts ticking.

Strength initialisation is a saturating linear function of the number
of shared maps:

.. math::

    \\text{strength} = \\min\\!\\left( 1.0, \\frac{\\text{shared\\_maps}}{N} \\right)

with ``N = 20`` so a full split-season's worth of maps maxes the bond.
Sentiment defaults to ``+1.0``: the bootstrap has no upstream signal
that would warrant a negative valence — feuds, when they happen, land
as ``relationship_event`` rows after the bootstrap.

Idempotency: re-running the seed against an already-populated table is
an UPSERT-with-replace. Existing edges have their strength /
last_updated_at re-derived from the underlying shared-maps tally so
re-running after a fresh ``MapResult`` ingest catches the new pairs
without leaving the old edges stale.
"""

from __future__ import annotations

import logging
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

from esports_sim.db.enums import RelationshipEdgeType
from esports_sim.db.models import (
    MapResult,
    Match,
    PlayerMatchStat,
    RelationshipEdge,
)
from sqlalchemy import select
from sqlalchemy.orm import Session

_logger = logging.getLogger("data_pipeline.seeds.relationships")


# Number of days that the most-recent shared map must lie within for a
# pair to count as "current" teammates rather than ex-teammates. Tuned
# to ~13 weeks — roughly one VCT split — so a roster that played
# together this season is current; a duo last seen a year ago is ex.
RECENT_TEAMMATE_DAYS: int = 90


# Number of shared maps that saturates the initial strength to 1.0.
# Below this, strength scales linearly. The threshold is opinionated
# but anchored: a single VCT regular season is roughly 18-22 best-of-
# three matches per team, so a full split's worth of co-play tops out
# the bond — anything beyond that is signal redundancy.
STRENGTH_SATURATION_SHARED_MAPS: int = 20


@dataclass(frozen=True, slots=True)
class _PairAccumulator:
    """Per-pair aggregate the bootstrap builds before persisting.

    ``shared_maps`` counts distinct ``map_result_id`` values where both
    players appeared with the same ``team_side``. ``last_match_at`` is
    the most recent of those.
    """

    shared_maps: int
    last_match_at: datetime


@dataclass
class TeammateBootstrapManifest:
    """Counts emitted by :func:`bootstrap_teammate_edges`.

    Persisted alongside the seed run so an operator can confirm "the
    bootstrap saw N pairs and produced N edges" without re-running a
    diagnostic query against the DB.
    """

    pairs_seen: int = 0
    teammate_edges_inserted: int = 0
    ex_teammate_edges_inserted: int = 0
    edges_updated: int = 0
    reference_timestamp: datetime | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def total_edges(self) -> int:
        return self.teammate_edges_inserted + self.ex_teammate_edges_inserted


def _canonical_pair(a: uuid.UUID, b: uuid.UUID) -> tuple[uuid.UUID, uuid.UUID]:
    """Return ``(min, max)`` so a symmetric pair is keyed unambiguously.

    The DB also enforces this for symmetric edge kinds via the
    ``ck_relationship_edge_symmetric_canonical`` CHECK; canonicalising
    in Python keeps the bootstrap's accumulator dict on the same page
    so we don't end up with two accumulator entries for the same
    unordered pair.
    """
    return (a, b) if a < b else (b, a)


def _collect_pairs(
    session: Session,
) -> tuple[dict[tuple[uuid.UUID, uuid.UUID], _PairAccumulator], datetime | None]:
    """Walk PlayerMatchStat + MapResult and build the per-pair accumulator.

    The query joins ``player_match_stat`` to ``map_result`` to recover
    the per-map timestamp from the parent ``match`` row. Rows with a
    null ``team_side`` are skipped — without a side assignment we
    can't tell a teammate from an opponent.
    """
    rows = session.execute(
        select(
            PlayerMatchStat.entity_id,
            PlayerMatchStat.team_side,
            PlayerMatchStat.map_result_id,
            MapResult.match_id,
        )
        .join(MapResult, MapResult.map_result_id == PlayerMatchStat.map_result_id)
        .where(PlayerMatchStat.team_side.is_not(None))
    ).all()

    # Group player ids by (map_result_id, team_side); each group is a
    # team's roster for one map. Pairs within a group are teammates on
    # that map.
    by_side: dict[tuple[uuid.UUID, str], list[uuid.UUID]] = defaultdict(list)
    map_match_ids: dict[uuid.UUID, uuid.UUID] = {}
    for entity_id, team_side, map_result_id, match_id in rows:
        by_side[(map_result_id, team_side)].append(entity_id)
        map_match_ids[map_result_id] = match_id

    if not map_match_ids:
        return {}, None

    # Resolve match timestamps for the maps we care about. A second
    # round-trip rather than joining ``match`` directly: the join would
    # multiply rows we already discarded (null team_side) and the
    # explicit lookup is easier to read.
    match_date_rows = session.execute(
        select(Match.match_id, Match.match_date).where(
            Match.match_id.in_(set(map_match_ids.values()))
        )
    ).all()
    match_dates: dict[uuid.UUID, datetime] = {
        row.match_id: row.match_date for row in match_date_rows
    }

    pairs: dict[tuple[uuid.UUID, uuid.UUID], _PairAccumulator] = {}
    reference_ts: datetime | None = None
    for (map_result_id, _side), roster in by_side.items():
        match_id = map_match_ids[map_result_id]
        played_at = match_dates.get(match_id)
        if played_at is None:
            continue
        if reference_ts is None or played_at > reference_ts:
            reference_ts = played_at
        # All pairs within the same (map_result_id, team_side) group.
        # Deduplicate just in case the upstream emitted the same
        # entity twice in a roster (a partial-scrape edge case).
        unique_roster = sorted(set(roster))
        for i in range(len(unique_roster)):
            for j in range(i + 1, len(unique_roster)):
                key = _canonical_pair(unique_roster[i], unique_roster[j])
                acc = pairs.get(key)
                if acc is None:
                    pairs[key] = _PairAccumulator(
                        shared_maps=1,
                        last_match_at=played_at,
                    )
                else:
                    pairs[key] = _PairAccumulator(
                        shared_maps=acc.shared_maps + 1,
                        last_match_at=max(acc.last_match_at, played_at),
                    )

    return pairs, reference_ts


def _existing_edges(
    session: Session,
    pair_keys: list[tuple[uuid.UUID, uuid.UUID]],
) -> dict[tuple[uuid.UUID, uuid.UUID, RelationshipEdgeType], RelationshipEdge]:
    """Pre-load any teammate / ex-teammate edges that already exist.

    Keyed on ``(src, dst, kind)`` so the upsert path can branch on
    "row exists" vs "fresh insert" without paying a per-pair
    SELECT. Only pulls the two kinds the bootstrap manages — leaving
    the steady-state event-driven kinds (rival, mentor, …)
    untouched even if a pair shows up here.
    """
    if not pair_keys:
        return {}
    src_ids = [a for a, _ in pair_keys]
    dst_ids = [b for _, b in pair_keys]
    rows = (
        session.execute(
            select(RelationshipEdge).where(
                RelationshipEdge.src_id.in_(src_ids),
                RelationshipEdge.dst_id.in_(dst_ids),
                RelationshipEdge.edge_type.in_(
                    (RelationshipEdgeType.TEAMMATE, RelationshipEdgeType.EX_TEAMMATE),
                ),
            )
        )
        .scalars()
        .all()
    )
    return {(e.src_id, e.dst_id, e.edge_type): e for e in rows}


def bootstrap_teammate_edges(
    session: Session,
    *,
    reference_timestamp: datetime | None = None,
    recent_teammate_days: int = RECENT_TEAMMATE_DAYS,
    saturation_shared_maps: int = STRENGTH_SATURATION_SHARED_MAPS,
) -> TeammateBootstrapManifest:
    """Build a teammate / ex-teammate edge for every former-roster pair.

    Returns a :class:`TeammateBootstrapManifest` with the row counts
    so the operator-facing CLI can confirm the seed shape.

    ``reference_timestamp`` defaults to the latest match in the DB —
    the closest analogue to "now" against an offline seed corpus.
    Pass an explicit value to backdate the bootstrap (e.g. for a
    deterministic test fixture).

    The function does **not** commit; the caller owns the transaction
    so the seed composes with sibling work in the same session.
    """
    pairs, latest_match = _collect_pairs(session)
    manifest = TeammateBootstrapManifest(pairs_seen=len(pairs))
    if not pairs:
        manifest.reference_timestamp = reference_timestamp
        return manifest

    ref_ts = reference_timestamp or latest_match
    assert ref_ts is not None  # _collect_pairs returned pairs => latest_match is set
    manifest.reference_timestamp = ref_ts
    recent_threshold = ref_ts - timedelta(days=recent_teammate_days)

    # Symmetric kind: both endpoints are stored canonically (src < dst).
    # We pre-load existing rows for the pair-keys so re-runs UPSERT
    # rather than collide on the unique constraint.
    existing = _existing_edges(session, list(pairs.keys()))

    for (src_id, dst_id), acc in pairs.items():
        is_recent = acc.last_match_at >= recent_threshold
        edge_type = RelationshipEdgeType.TEAMMATE if is_recent else RelationshipEdgeType.EX_TEAMMATE
        strength = min(1.0, acc.shared_maps / float(saturation_shared_maps))
        extras = {
            "shared_maps": acc.shared_maps,
            "last_shared_match_at": acc.last_match_at.isoformat(),
            "bootstrap_reference_at": ref_ts.isoformat(),
        }

        # On a kind-flip (e.g. a fresh ingest moved a pair from
        # ex-teammate to teammate) mutate the existing row's
        # ``edge_type`` in place rather than delete-and-reinsert.
        # ``relationship_event.edge_id`` is ``ON DELETE CASCADE``,
        # so dropping the row would wipe the append-only audit
        # trail attached to it (Codex P1 on PR #28). The unique
        # constraint is ``(src, dst, edge_type)`` and we're
        # swapping the third column on a single row, which stays
        # unique by construction.
        for stale_kind in (RelationshipEdgeType.TEAMMATE, RelationshipEdgeType.EX_TEAMMATE):
            if stale_kind == edge_type:
                continue
            stale = existing.pop((src_id, dst_id, stale_kind), None)
            if stale is None:
                continue
            target = existing.get((src_id, dst_id, edge_type))
            if target is None:
                # Single-row flip: mutate kind in place and
                # re-key the lookup so the update branch below
                # finds it.
                stale.edge_type = edge_type
                existing[(src_id, dst_id, edge_type)] = stale
            else:
                # Defensive case: both kinds exist for the same
                # pair (shouldn't happen via the bootstrap, but
                # could arise from a hand-written event-driven
                # writer). Re-parent the stale row's events onto
                # the target at the ORM level so the cascade-delete
                # on ``stale`` doesn't propagate into the audit
                # trail. A bulk SQL UPDATE is *not* enough here:
                # ``RelationshipEdge.events`` is loaded eagerly
                # (``selectin``) with ``cascade="all, delete-orphan"``,
                # so SA's in-memory collection still anchors the
                # events to ``stale`` and the next flush would
                # delete them despite the SQL-level reparenting
                # (Codex P1 on PR #28). Re-assigning ``event.edge``
                # walks the bidirectional relationship and atomically
                # moves each child onto the target, so neither orphan
                # cascade nor delete cascade fires.
                for event in list(stale.events):
                    event.edge = target
                session.delete(stale)

        existing_edge = existing.get((src_id, dst_id, edge_type))
        if existing_edge is None:
            session.add(
                RelationshipEdge(
                    edge_id=uuid.uuid4(),
                    src_id=src_id,
                    dst_id=dst_id,
                    edge_type=edge_type,
                    strength=strength,
                    sentiment=1.0,
                    last_updated_at=acc.last_match_at,
                    extra=extras,
                )
            )
            if edge_type is RelationshipEdgeType.TEAMMATE:
                manifest.teammate_edges_inserted += 1
            else:
                manifest.ex_teammate_edges_inserted += 1
        else:
            # Rerun against an already-seeded edge. The bootstrap is a
            # baseline-seeding pass — once a post-bootstrap writer
            # (the event-driven path: ``relationship_event`` rows
            # folded into the edge, or the monthly decay job) has
            # advanced ``last_updated_at`` past ``acc.last_match_at``,
            # they own the live magnitude. Re-deriving ``strength``
            # from shared-map count would wipe their adjustments AND
            # rewind the decay anchor, so the next decay pass would
            # over-apply the elapsed weeks (Codex P1 on PR #28).
            # Sentiment is never touched on rerun, same rationale.
            #
            # The provenance ``extra`` is always refreshed so an
            # operator can see what the latest match history looks
            # like for this pair, regardless of whether the event-
            # driven writer has taken over.
            event_writer_has_taken_over = existing_edge.last_updated_at > acc.last_match_at
            if not event_writer_has_taken_over:
                existing_edge.strength = strength
                existing_edge.last_updated_at = acc.last_match_at
            existing_edge.extra = extras
            manifest.edges_updated += 1

    session.flush()
    _logger.info(
        "teammate bootstrap complete",
        extra={
            "pairs_seen": manifest.pairs_seen,
            "teammate_inserted": manifest.teammate_edges_inserted,
            "ex_teammate_inserted": manifest.ex_teammate_edges_inserted,
            "edges_updated": manifest.edges_updated,
        },
    )
    return manifest


__all__ = [
    "RECENT_TEAMMATE_DAYS",
    "STRENGTH_SATURATION_SHARED_MAPS",
    "TeammateBootstrapManifest",
    "bootstrap_teammate_edges",
]
