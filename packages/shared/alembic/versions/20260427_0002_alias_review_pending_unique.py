"""Partial unique index on alias_review_queue's pending rows.

The resolver's pending-review enqueue used to be a check-then-insert
sequence. Two concurrent ``resolve_entity()`` calls for the same
``(platform, platform_id)`` could both observe "no pending row exists"
and each insert one — duplicating human-review work. This migration
adds the partial unique index that closes the race at the schema level
so the resolver can rely on a unique-violation savepoint to enforce
idempotency.

Revision ID: 0002
Revises: 0001
Create Date: 2026-04-27
"""

from __future__ import annotations

from collections.abc import Sequence

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_index(
        "ix_alias_review_queue_pending_unique",
        "alias_review_queue",
        ["platform", "platform_id"],
        unique=True,
        # Postgres treats partial-index predicates as text; the literal
        # ``status = 'pending'`` matches the stored enum value. ``checkfirst``
        # is implicit in alembic's ``create_index``; running upgrade against a
        # DB that already has the index would error and that's the correct
        # signal to investigate.
        postgresql_where="status = 'pending'",
    )


def downgrade() -> None:
    op.drop_index("ix_alias_review_queue_pending_unique", table_name="alias_review_queue")
