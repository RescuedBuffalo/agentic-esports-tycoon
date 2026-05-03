"""Canonical enums shared by SQLAlchemy models, Pydantic DTOs, and migrations.

Important: when a value is added here, the corresponding Postgres ENUM type
must be migrated explicitly via ``ALTER TYPE ... ADD VALUE``. Alembic
autogenerate will not catch enum-value drift, so always write the migration by
hand.
"""

from __future__ import annotations

import enum


class EntityType(enum.StrEnum):
    """The Systems-spec covers four canonical entity classes.

    Future entity classes (analyst, caster, organisation) will require an
    explicit migration plus a forward-compatibility plan; we don't pre-empt
    them here.
    """

    PLAYER = "player"
    TEAM = "team"
    COACH = "coach"
    TOURNAMENT = "tournament"


class Platform(enum.StrEnum):
    """External platforms the resolver may see an alias from.

    The order in BUF-12's source-priority list is *not* encoded here; that's
    a resolver concern and lives next to the merge logic. Adding a new
    platform requires both an ALTER TYPE migration and a priority decision in
    BUF-12.
    """

    RIOT_API = "riot_api"
    LIQUIPEDIA = "liquipedia"
    VLR = "vlr"
    ESPORTSEARNINGS = "esportsearnings"
    TWITCH = "twitch"
    TWITTER = "twitter"


class StagingStatus(enum.StrEnum):
    """Lifecycle of a row in the staging queue.

    The resolver (BUF-7) only consumes ``pending``. ``review`` items land in
    ``alias_review_queue``; ``blocked`` rows are kept for audit, never
    re-processed automatically.
    """

    PENDING = "pending"
    PROCESSED = "processed"
    BLOCKED = "blocked"
    REVIEW = "review"


class ReviewStatus(enum.StrEnum):
    """Lifecycle of a row in the alias review queue (BUF-16).

    ``resolved`` covers both human decisions that produce a write (merge into
    existing entity, or mint a new canonical id); the actual decision lives
    in the row's audit trail, not on this status field.
    """

    PENDING = "pending"
    RESOLVED = "resolved"
    SKIPPED = "skipped"
    BLOCKED = "blocked"


class RelationshipEdgeType(enum.StrEnum):
    """Kinds of pairwise relationships the ecosystem layer tracks (BUF-26, System 07).

    Each kind has its own decay rate in
    :data:`ecosystem.relationships.DECAY_RATES`; bumping the dict and the
    enum together is a versioned event that Phase-1 retro-decay analyses
    have to re-run against. Members are deliberately coarse — finer
    distinctions live in ``relationship_event.event_kind`` so a downstream
    feature extractor can split a TEAMMATE edge into "shared scrim" vs
    "co-clutch" without forcing a new edge type.

    Symmetric kinds (teammate, ex-teammate, rival, friend) MUST be
    persisted with ``src_id < dst_id`` so a single pair owns at most one
    row of that type — see :data:`SYMMETRIC_RELATIONSHIP_EDGE_TYPES`
    and the partial check in the migration.
    """

    TEAMMATE = "teammate"
    EX_TEAMMATE = "ex_teammate"
    RIVAL = "rival"
    FRIEND = "friend"
    MENTOR = "mentor"
    MANAGER_OF = "manager_of"


# Edge kinds whose semantics are symmetric: the relationship "A is a
# teammate of B" is the same fact as "B is a teammate of A". The
# bootstrap and the application API enforce ``src_id < dst_id`` for
# these, so a pair of players never owns two rows of the same symmetric
# kind. Asymmetric kinds (mentor, manager_of) keep direction.
SYMMETRIC_RELATIONSHIP_EDGE_TYPES: frozenset[RelationshipEdgeType] = frozenset(
    {
        RelationshipEdgeType.TEAMMATE,
        RelationshipEdgeType.EX_TEAMMATE,
        RelationshipEdgeType.RIVAL,
        RelationshipEdgeType.FRIEND,
    }
)
