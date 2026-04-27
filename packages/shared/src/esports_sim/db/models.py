"""SQLAlchemy 2.0 models for the BUF-6 schema.

Each table maps 1:1 to the Systems-spec System 01 + 02 design. Downstream code
must import these models (or the Pydantic DTOs in
:mod:`esports_sim.schemas.dtos`) — never reach into raw SQL.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    String,
    UniqueConstraint,
    event,
    func,
)
from sqlalchemy.dialects.postgresql import ENUM as PgEnum
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.engine import Connection
from sqlalchemy.orm import Mapped, Mapper, Session, mapped_column, relationship

from esports_sim.db.base import Base
from esports_sim.db.enums import EntityType, Platform, ReviewStatus, StagingStatus

# Reusable Postgres ENUM types. ``create_type=False`` keeps every model
# definition from trying to recreate the type — the migration owns the
# CREATE TYPE statement, full stop.
_entity_type = PgEnum(
    EntityType,
    name="entity_type",
    create_type=False,
    values_callable=lambda e: [v.value for v in e],
)
_platform = PgEnum(
    Platform,
    name="platform",
    create_type=False,
    values_callable=lambda e: [v.value for v in e],
)
_staging_status = PgEnum(
    StagingStatus,
    name="staging_status",
    create_type=False,
    values_callable=lambda e: [v.value for v in e],
)
_review_status = PgEnum(
    ReviewStatus,
    name="review_status",
    create_type=False,
    values_callable=lambda e: [v.value for v in e],
)


class Entity(Base):
    """Canonical record. Every other table joins on ``canonical_id``."""

    __tablename__ = "entity"

    canonical_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    # ``index=True`` mirrors the ix_entity_entity_type index in the migration.
    # Without this, alembic autogenerate would treat the DB index as drift and
    # drop it in a future revision — costly because lots of downstream queries
    # filter by entity_type.
    entity_type: Mapped[EntityType] = mapped_column(_entity_type, nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    is_active: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        server_default="true",
    )

    aliases: Mapped[list[EntityAlias]] = relationship(
        back_populates="entity",
        cascade="all, delete-orphan",
        lazy="selectin",
    )


class EntityAlias(Base):
    """Many-to-one platform handle.

    The ``(platform, platform_id)`` uniqueness is the schema-level guarantee
    that two scrapers can't accidentally split a player into two canonical
    rows. The resolver (BUF-7) is the runtime guarantee that they don't try.
    """

    __tablename__ = "entity_alias"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    canonical_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("entity.canonical_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    platform: Mapped[Platform] = mapped_column(_platform, nullable=False)
    platform_id: Mapped[str] = mapped_column(String(255), nullable=False)
    platform_name: Mapped[str] = mapped_column(String(255), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    verified_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
    )
    valid_from: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    entity: Mapped[Entity] = relationship(back_populates="aliases")

    __table_args__ = (
        UniqueConstraint(
            "platform",
            "platform_id",
            name="uq_entity_alias_platform_platform_id",
        ),
    )


class StagingRecord(Base):
    """Scraped payload waiting for the resolver.

    ``canonical_id`` is nullable to support the documented staging
    lifecycle: ``pending`` rows queued for the resolver, ``review`` rows
    deferred to a human reviewer, and ``blocked`` rows kept only for
    audit may all legitimately carry a null canonical id. The bypass that
    BUF-7 closes is the ``processed`` state — once a row is marked as
    fully processed, the canonical id MUST be set. ``StagingRecord.save()``
    plus the event listener below refuse to flush a ``processed`` row with
    a null canonical id, which is the only way a scraper could "finish"
    a staging row without actually consulting the resolver.
    """

    __tablename__ = "staging_record"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    source: Mapped[str] = mapped_column(String(64), nullable=False)
    entity_type: Mapped[EntityType] = mapped_column(_entity_type, nullable=False)
    canonical_id: Mapped[uuid.UUID | None] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("entity.canonical_id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    status: Mapped[StagingStatus] = mapped_column(
        _staging_status,
        nullable=False,
        server_default=StagingStatus.PENDING.value,
        index=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    def save(self, session: Session) -> None:
        """Persist this record, enforcing the canonical-id invariant.

        Refuses to add a ``processed`` row whose ``canonical_id`` is still
        null — the only way that combination can arise is a scraper marking
        a staging row as fully processed without actually consulting the
        resolver. ``pending``/``review``/``blocked`` rows may legitimately
        have a null canonical_id and pass through unchanged.

        Direct ``session.add(record)`` is also covered by the event listener
        below; ``save()`` exists so call sites read intentionally and so
        unit tests can assert against a single named code path.
        """
        _check_canonical_invariant(self)
        session.add(self)


class StagingInvariantError(ValueError):
    """Raised when a :class:`StagingRecord` violates BUF-7's resolver rule.

    A ``processed`` staging row whose ``canonical_id`` is null is the
    bypass case: a scraper has declared the row "done" without going
    through :func:`esports_sim.resolver.resolve_entity`. Refuse it loudly
    rather than corrupting the canonical join key.
    """


# Statuses where a null canonical_id is meaningful state, not a bug. ``pending``
# is the pre-resolver queue state; ``review`` and ``blocked`` are terminal
# states the resolver itself produces when it can't (or won't) auto-decide.
# Only ``processed`` requires a non-null canonical_id, because that's the
# state that asserts "this row has a canonical mapping". Kept as a
# module-level frozenset so the listener and the wrapper can't drift apart.
_NULL_CANONICAL_OK: frozenset[StagingStatus] = frozenset(
    {StagingStatus.PENDING, StagingStatus.REVIEW, StagingStatus.BLOCKED}
)


def _check_canonical_invariant(record: StagingRecord) -> None:
    if record.canonical_id is not None:
        return
    if record.status in _NULL_CANONICAL_OK:
        return
    raise StagingInvariantError(
        f"StagingRecord with status={record.status.value!r} requires a "
        "non-null canonical_id. The resolver (BUF-7) is the only sanctioned "
        "writer of that column; marking a row processed without one means a "
        "scraper bypassed resolve_entity()."
    )


@event.listens_for(StagingRecord, "before_insert")
def _staging_before_insert(
    _mapper: Mapper[StagingRecord],
    _connection: Connection,
    target: StagingRecord,
) -> None:
    # SQLAlchemy event hook. Backstop for ``session.add(StagingRecord(...))``
    # paths that don't go through ``StagingRecord.save``. Note: this does not
    # fire for FK-cascade SET NULL, which happens inside Postgres — that's
    # the audit-trail outcome the schema deliberately allows.
    _check_canonical_invariant(target)


@event.listens_for(StagingRecord, "before_update")
def _staging_before_update(
    _mapper: Mapper[StagingRecord],
    _connection: Connection,
    target: StagingRecord,
) -> None:
    _check_canonical_invariant(target)


class RawRecord(Base):
    """Content-addressed blob store.

    Scrapers hash their payload before insert; ``content_hash`` is unique so
    the same fetch can be replayed without producing duplicates. Useful for
    debugging "why did the resolver decide that?" without re-hitting the
    upstream API.
    """

    __tablename__ = "raw_record"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    source: Mapped[str] = mapped_column(String(64), nullable=False)
    fetched_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)


class AliasReviewQueue(Base):
    """Items the fuzzy matcher couldn't auto-decide.

    ``candidates`` is a JSON array of plausible canonical_ids with their
    similarity scores; the human reviewer picks one or marks the row blocked.
    See BUF-16 for the CLI/UI that drains this.
    """

    __tablename__ = "alias_review_queue"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    platform: Mapped[Platform] = mapped_column(_platform, nullable=False)
    platform_id: Mapped[str] = mapped_column(String(255), nullable=False)
    platform_name: Mapped[str] = mapped_column(String(255), nullable=False)
    candidates: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, nullable=False)
    reason: Mapped[str] = mapped_column(String(255), nullable=False)
    status: Mapped[ReviewStatus] = mapped_column(
        _review_status,
        nullable=False,
        server_default=ReviewStatus.PENDING.value,
        index=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )


__all__ = [
    "Entity",
    "EntityAlias",
    "StagingRecord",
    "StagingInvariantError",
    "RawRecord",
    "AliasReviewQueue",
]
