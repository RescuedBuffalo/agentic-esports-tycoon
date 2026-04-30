"""Add patch_note table for the playvalorant.com scraper (BUF-83).

Patch notes are documents, not resolver-eligible entities, so they get
their own table rather than abusing ``raw_record`` (which would lose
``patch_version`` / ``published_at`` as first-class columns) or being
forced through the alias graph (no fuzzy match makes sense for a
versioned release-notes blob).

``patch_version`` is unique so the patch-notes runner can UPSERT on it
— a re-scrape of the same article updates the body in place rather than
inserting a duplicate row, satisfying the BUF-83 idempotency acceptance.

Revision ID: 0003
Revises: 0002
Create Date: 2026-04-29
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0003"
down_revision: str | None = "0002"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "patch_note",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        # 32 chars is plenty for "Major.Minor[.Hotfix]" version strings;
        # narrowing the column keeps the unique-index footprint small.
        sa.Column("patch_version", sa.String(32), nullable=False),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=False),
        # raw_html / body_text are unbounded — patch notes can be tens of KB.
        # Postgres TEXT has no length cap and no on-disk overhead vs VARCHAR.
        sa.Column("raw_html", sa.Text(), nullable=False),
        sa.Column("body_text", sa.Text(), nullable=False),
        sa.Column("url", sa.String(512), nullable=False),
        sa.Column(
            "fetched_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.UniqueConstraint("patch_version", name="uq_patch_note_patch_version"),
    )
    # Freshness check (``nexus validate``) reads the most recent
    # ``published_at`` per source; the index keeps that O(log n).
    op.create_index("ix_patch_note_published_at", "patch_note", ["published_at"])


def downgrade() -> None:
    op.drop_index("ix_patch_note_published_at", table_name="patch_note")
    op.drop_table("patch_note")
