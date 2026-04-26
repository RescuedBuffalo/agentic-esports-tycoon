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
