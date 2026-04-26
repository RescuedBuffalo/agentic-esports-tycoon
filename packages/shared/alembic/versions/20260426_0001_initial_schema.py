"""Initial schema: entity, entity_alias, staging_record, raw_record, alias_review_queue.

Lands the BUF-6 substrate plus the ``vector`` extension so BUF-28 can add
embedding tables without a second migration cycle (per the BUF-6 review
comment + ADR-006).

Revision ID: 0001
Revises:
Create Date: 2026-04-26
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


# Enum value lists. Kept as literals on purpose: a migration must keep working
# even after the application enum gets new values, so we don't import
# esports_sim.db.enums here. Adding a value goes through ALTER TYPE ... ADD
# VALUE in a follow-up migration.
_ENTITY_TYPES = ("player", "team", "coach", "tournament")
_PLATFORMS = (
    "riot_api",
    "liquipedia",
    "vlr",
    "esportsearnings",
    "twitch",
    "twitter",
)
_STAGING_STATUSES = ("pending", "processed", "blocked", "review")
_REVIEW_STATUSES = ("pending", "resolved", "skipped", "blocked")


def _enum(name: str, values: tuple[str, ...]) -> postgresql.ENUM:
    """Reference an already-created Postgres ENUM type without recreating it."""
    return postgresql.ENUM(*values, name=name, create_type=False)


def upgrade() -> None:
    # pgvector — BUF-28 will write into vector columns; lighting it up here
    # avoids a second migration that would otherwise need a maintenance
    # window. `IF NOT EXISTS` keeps this safe if the operator already ran it.
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # Enum types must exist before the tables that reference them.
    postgresql.ENUM(*_ENTITY_TYPES, name="entity_type").create(op.get_bind(), checkfirst=True)
    postgresql.ENUM(*_PLATFORMS, name="platform").create(op.get_bind(), checkfirst=True)
    postgresql.ENUM(*_STAGING_STATUSES, name="staging_status").create(
        op.get_bind(), checkfirst=True
    )
    postgresql.ENUM(*_REVIEW_STATUSES, name="review_status").create(op.get_bind(), checkfirst=True)

    op.create_table(
        "entity",
        sa.Column(
            "canonical_id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
        ),
        sa.Column(
            "entity_type",
            _enum("entity_type", _ENTITY_TYPES),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "is_active",
            sa.Boolean(),
            server_default=sa.text("true"),
            nullable=False,
        ),
    )
    op.create_index("ix_entity_entity_type", "entity", ["entity_type"])

    op.create_table(
        "entity_alias",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "canonical_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey(
                "entity.canonical_id",
                ondelete="CASCADE",
                name="fk_entity_alias_canonical_id_entity",
            ),
            nullable=False,
        ),
        sa.Column(
            "platform",
            _enum("platform", _PLATFORMS),
            nullable=False,
        ),
        sa.Column("platform_id", sa.String(255), nullable=False),
        sa.Column("platform_name", sa.String(255), nullable=False),
        sa.Column("confidence", sa.Float(), nullable=False),
        sa.Column("verified_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "valid_from",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.UniqueConstraint(
            "platform",
            "platform_id",
            name="uq_entity_alias_platform_platform_id",
        ),
    )
    op.create_index("ix_entity_alias_canonical_id", "entity_alias", ["canonical_id"])

    op.create_table(
        "staging_record",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("source", sa.String(64), nullable=False),
        sa.Column(
            "entity_type",
            _enum("entity_type", _ENTITY_TYPES),
            nullable=False,
        ),
        sa.Column(
            "canonical_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey(
                "entity.canonical_id",
                ondelete="SET NULL",
                name="fk_staging_record_canonical_id_entity",
            ),
            nullable=True,
        ),
        sa.Column("payload", postgresql.JSONB, nullable=False),
        sa.Column(
            "status",
            _enum("staging_status", _STAGING_STATUSES),
            server_default="pending",
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("ix_staging_record_status", "staging_record", ["status"])
    op.create_index("ix_staging_record_canonical_id", "staging_record", ["canonical_id"])

    op.create_table(
        "raw_record",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("source", sa.String(64), nullable=False),
        sa.Column(
            "fetched_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("payload", postgresql.JSONB, nullable=False),
        sa.Column("content_hash", sa.String(64), nullable=False, unique=True),
    )

    op.create_table(
        "alias_review_queue",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "platform",
            _enum("platform", _PLATFORMS),
            nullable=False,
        ),
        sa.Column("platform_id", sa.String(255), nullable=False),
        sa.Column("platform_name", sa.String(255), nullable=False),
        sa.Column("candidates", postgresql.JSONB, nullable=False),
        sa.Column("reason", sa.String(255), nullable=False),
        sa.Column(
            "status",
            _enum("review_status", _REVIEW_STATUSES),
            server_default="pending",
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index("ix_alias_review_queue_status", "alias_review_queue", ["status"])


def downgrade() -> None:
    # Tables first (respecting FK order), then enum types, then the extension.
    op.drop_index("ix_alias_review_queue_status", table_name="alias_review_queue")
    op.drop_table("alias_review_queue")

    op.drop_table("raw_record")

    op.drop_index("ix_staging_record_canonical_id", table_name="staging_record")
    op.drop_index("ix_staging_record_status", table_name="staging_record")
    op.drop_table("staging_record")

    op.drop_index("ix_entity_alias_canonical_id", table_name="entity_alias")
    op.drop_table("entity_alias")

    op.drop_index("ix_entity_entity_type", table_name="entity")
    op.drop_table("entity")

    for typename in ("review_status", "staging_status", "platform", "entity_type"):
        op.execute(f"DROP TYPE IF EXISTS {typename}")

    # Leaving `CREATE EXTENSION vector` undone is friendly to other databases
    # on the same instance — but the migration still owns it for symmetry.
    op.execute("DROP EXTENSION IF EXISTS vector")
