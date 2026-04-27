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

    ``canonical_id`` is nullable so a row can be parked before the resolver
    decides what entity it belongs to. BUF-7's ``StagingRecord.save()``
    enforces the rule that a null ``canonical_id`` is only legal when the
    status is ``blocked`` or ``review``. The :func:`_validate_canonical_id`
    listener wired further down is the runtime backstop: even a scraper that
    bypasses :meth:`save` and calls ``session.add`` directly cannot push a
    null-canonical row in any status besides those two.
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

        BUF-7 requires the resolver to be the only thing that fills in a
        ``canonical_id``. A row that is neither under review nor blocked must
        therefore already carry a non-null canonical_id by the time it lands
        in the session — anything else means a scraper tried to skip the
        resolver. This wrapper raises :class:`StagingInvariantError` rather
        than letting the row reach the database.

        Direct ``session.add(record)`` is also covered by the event listener
        below; ``save()`` exists so call sites read intentionally and so
        unit tests can assert against a single named code path.
        """
        _check_canonical_invariant(self)
        session.add(self)


class StagingInvariantError(ValueError):
    """Raised when a :class:`StagingRecord` violates BUF-7's resolver rule.

    A staging row may only carry a null ``canonical_id`` when its
    ``status`` is ``blocked`` (kept for audit, never reprocessed) or
    ``review`` (deferred to the alias review queue). Any other status with a
    null canonical id means a scraper tried to write through without going
    through :func:`esports_sim.resolver.resolve_entity` — refuse it loudly
    rather than corrupting the canonical join key.
    """


# Statuses where a null canonical_id is meaningful state, not a bug. Kept as a
# module-level frozenset so the listener and the wrapper can't drift apart.
_NULL_CANONICAL_OK: frozenset[StagingStatus] = frozenset(
    {StagingStatus.BLOCKED, StagingStatus.REVIEW}
)


def _check_canonical_invariant(record: StagingRecord) -> None:
    if record.canonical_id is not None:
        return
    if record.status in _NULL_CANONICAL_OK:
        return
    raise StagingInvariantError(
        "StagingRecord with null canonical_id requires status=blocked or "
        f"status=review (got {record.status.value!r}). The resolver "
        "(BUF-7) is the only sanctioned writer of canonical_id."
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
