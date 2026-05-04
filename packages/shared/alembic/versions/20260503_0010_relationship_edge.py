"""Add relationship_edge + relationship_event tables (BUF-26, System 07).

System 07 in the Systems-spec: pairwise relationships between canonical
entities, carrying both a magnitude (``strength``, decays exponentially
with no signal) and a valence (``sentiment``, persistent). The two
tables this migration installs are:

* ``relationship_edge`` — one row per (src, dst, kind) triple. Symmetric
  kinds (teammate, ex-teammate, rival, friend) must be persisted with
  ``src_id < dst_id`` so a single canonical pair owns at most one row
  of that type. Asymmetric kinds (mentor, manager_of) keep direction.
* ``relationship_event`` — append-only signal log. Every event that
  should affect an edge lands here first; the writer that produces it
  also folds the deltas into the parent edge and bumps
  ``last_updated_at`` so the next decay pass measures from the
  refreshed reference point.

Schema rationale:

* ``edge_type`` is a Postgres ENUM rather than a free-form string. The
  finite kind set drives the ``DECAY_RATES`` lookup in
  :data:`ecosystem.relationships.DECAY_RATES`; gating it at the DB
  catches a typo'd writer at insert time rather than letting it land
  a row the decay job will later silently skip.
* The symmetric-canonical CHECK lists the symmetric kinds inline so
  the migration is self-contained — adding a new symmetric kind is a
  coordinated migration + enum + Python frozenset change, not a
  silent runtime-only update.
* Both ``src_id`` and ``dst_id`` get their own indexes so the typical
  query pattern ("all edges this player is involved in", regardless of
  direction) stays O(log n) on either side of the pair.
* ``relationship_event`` cascades on ``ON DELETE CASCADE`` from the
  parent edge — an event without its edge is just noise, and orphaning
  rows would mask a buggy edge cleanup. Same FK posture as
  ``map_result -> player_match_stat``.
* ``last_updated_at`` is the anchor :func:`decay_strength` reads. The
  decay job is the canonical writer; ad-hoc writers set it explicitly.

Revision ID: 0010
Revises: 0009
Create Date: 2026-05-03
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0010"
down_revision: str | None = "0009"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


# Mirror of ``esports_sim.db.enums.RelationshipEdgeType``. Kept as
# literals (not imported) so the migration keeps applying even if the
# Python enum is later refactored — adding a value is a follow-up
# ALTER TYPE migration, same convention the BUF-6 enums use.
_RELATIONSHIP_EDGE_TYPES = (
    "teammate",
    "ex_teammate",
    "rival",
    "friend",
    "mentor",
    "manager_of",
)


def upgrade() -> None:
    postgresql.ENUM(
        *_RELATIONSHIP_EDGE_TYPES,
        name="relationship_edge_type",
    ).create(op.get_bind(), checkfirst=True)

    op.create_table(
        "relationship_edge",
        sa.Column("edge_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "src_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey(
                "entity.canonical_id",
                ondelete="CASCADE",
                name="fk_relationship_edge_src_id_entity",
            ),
            nullable=False,
        ),
        sa.Column(
            "dst_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey(
                "entity.canonical_id",
                ondelete="CASCADE",
                name="fk_relationship_edge_dst_id_entity",
            ),
            nullable=False,
        ),
        sa.Column(
            "edge_type",
            postgresql.ENUM(
                *_RELATIONSHIP_EDGE_TYPES,
                name="relationship_edge_type",
                create_type=False,
            ),
            nullable=False,
        ),
        sa.Column("strength", sa.Float(), nullable=False, server_default=sa.text("0")),
        sa.Column("sentiment", sa.Float(), nullable=False, server_default=sa.text("0")),
        sa.Column(
            "last_updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column(
            "extra",
            postgresql.JSONB,
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        # One row per (pair, kind). Symmetric kinds canonicalise endpoints
        # to ``src < dst`` so this is also one row per (unordered pair,
        # kind) for those types; asymmetric kinds keep direction.
        sa.UniqueConstraint(
            "src_id",
            "dst_id",
            "edge_type",
            name="uq_relationship_edge_src_dst_type",
        ),
        sa.CheckConstraint(
            "src_id <> dst_id",
            name="ck_relationship_edge_no_self_edge",
        ),
        # Symmetric kinds must be stored canonically (src < dst). The
        # set of symmetric kinds is mirrored in
        # ``esports_sim.db.enums.SYMMETRIC_RELATIONSHIP_EDGE_TYPES``;
        # adding a new symmetric kind requires updating both.
        sa.CheckConstraint(
            "edge_type NOT IN ('teammate', 'ex_teammate', 'rival', 'friend') " "OR src_id < dst_id",
            name="ck_relationship_edge_symmetric_canonical",
        ),
        sa.CheckConstraint(
            "strength >= 0 AND strength <= 1",
            name="ck_relationship_edge_strength_range",
        ),
        sa.CheckConstraint(
            "sentiment >= -1 AND sentiment <= 1",
            name="ck_relationship_edge_sentiment_range",
        ),
    )
    # Both endpoints are query-able pivots ("all edges where this
    # player is involved, regardless of direction"); index each
    # column standalone so the OR / UNION ALL pattern doesn't seq-scan
    # one side of the pair.
    op.create_index(
        "ix_relationship_edge_src_id",
        "relationship_edge",
        ["src_id"],
    )
    op.create_index(
        "ix_relationship_edge_dst_id",
        "relationship_edge",
        ["dst_id"],
    )
    # The decay job filters on ``edge_type`` to apply per-type rates;
    # the index keeps that O(log n) once the table is fully seeded.
    op.create_index(
        "ix_relationship_edge_edge_type",
        "relationship_edge",
        ["edge_type"],
    )

    op.create_table(
        "relationship_event",
        sa.Column("event_id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column(
            "edge_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey(
                "relationship_edge.edge_id",
                ondelete="CASCADE",
                name="fk_relationship_event_edge_id_relationship_edge",
            ),
            nullable=False,
        ),
        sa.Column("event_kind", sa.String(64), nullable=False),
        sa.Column("delta_strength", sa.Float(), nullable=False, server_default=sa.text("0")),
        sa.Column("delta_sentiment", sa.Float(), nullable=False, server_default=sa.text("0")),
        sa.Column(
            "occurred_at",
            sa.DateTime(timezone=True),
            nullable=False,
        ),
        sa.Column(
            "payload",
            postgresql.JSONB,
            nullable=False,
            server_default=sa.text("'{}'::jsonb"),
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    op.create_index(
        "ix_relationship_event_edge_id",
        "relationship_event",
        ["edge_id"],
    )
    op.create_index(
        "ix_relationship_event_occurred_at",
        "relationship_event",
        ["occurred_at"],
    )


def downgrade() -> None:
    op.drop_index("ix_relationship_event_occurred_at", table_name="relationship_event")
    op.drop_index("ix_relationship_event_edge_id", table_name="relationship_event")
    op.drop_table("relationship_event")

    op.drop_index("ix_relationship_edge_edge_type", table_name="relationship_edge")
    op.drop_index("ix_relationship_edge_dst_id", table_name="relationship_edge")
    op.drop_index("ix_relationship_edge_src_id", table_name="relationship_edge")
    op.drop_table("relationship_edge")

    op.execute("DROP TYPE IF EXISTS relationship_edge_type")
