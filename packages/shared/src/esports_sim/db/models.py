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
    func,
)
from sqlalchemy.dialects.postgresql import ENUM as PgEnum
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

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
    status is ``blocked`` or ``review``.
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
    "RawRecord",
    "AliasReviewQueue",
]
