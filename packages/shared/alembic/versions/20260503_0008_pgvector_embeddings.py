"""pgvector embedding tables for personality summaries + transcripts (BUF-28).

Lands the two tables that hold vector representations of:

* one personality summary per canonical entity (one row per entity, the
  paragraph-level summary BUF-25 produces), and
* one ~500-token chunk of a transcript per row (BUF-21's Whisper output
  sliced and embedded).

Both columns are ``vector(384)`` — the output width of
``sentence-transformers/all-MiniLM-L6-v2``, the local CPU/GPU embedder
this project standardised on per ADR-006 (kept off API budget; runs on
the project 5090). HNSW indexes use ``vector_cosine_ops`` because
similarity-of-meaning queries are scale-invariant; an L2 index would
sort by raw magnitude instead.

Why this lives in Postgres and not Qdrant: the queries we actually run
("similar to X *and* on an active T1 roster *and* plays duelist") are
cross-entity SQL JOINs, not vector-payload-filter searches. ADR-006
documents the trade. The base image is already
``pgvector/pgvector:pg16`` and migration 0001 ran ``CREATE EXTENSION
IF NOT EXISTS vector`` so this migration assumes the extension is
present and drops straight to ``CREATE TABLE``.

Schema rationale:

* ``personality_embedding`` keys on ``entity_id`` directly (PK + FK +
  ``ON DELETE CASCADE``). One personality per entity is the right
  cardinality — re-extraction is an UPSERT, not append-only.
* ``transcript_chunk_embedding`` does **not** FK to a media table.
  BUF-21's media table doesn't exist yet; declaring a FK against a
  non-existent table would block this migration. The follow-up
  migration after BUF-21 lands will add the FK + ``ON DELETE
  CASCADE``. Until then, the writer (Whisper pipeline) is the
  authority for cleanup — re-runs DELETE-by-media_id then INSERT.
* HNSW indexes use the pgvector defaults (``m=16``,
  ``ef_construction=64``). The 100k-vector latency target in the
  BUF-28 acceptance comfortably fits the defaults; we revisit
  parameters only if recall measurement shows a regression.
* ``model_version`` is per-row (not a global config) so a future
  embedder rotation can re-embed in place and the application can
  tell a mixed population apart from a clean one without losing
  the audit trail.

Revision ID: 0008
Revises: 0007
Create Date: 2026-05-03
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0008"
down_revision: str | None = "0007"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


# Embedding width must match the constant in ``esports_sim.db.models`` and
# the runtime ``Embedder``; a mismatch surfaces only at first insert as an
# opaque pgvector cast error. Hardcoded here (not imported) so the
# migration keeps working even if the application constant moves.
_EMBEDDING_DIM = 384


def upgrade() -> None:
    # ``CREATE EXTENSION`` ran in migration 0001; the BUF-6 review left it
    # there specifically so BUF-28 wouldn't need a maintenance window. If
    # an operator has manually dropped it in between, the CREATE TABLE
    # below will fail with "type vector does not exist", which is the
    # right error to investigate rather than a silent re-create here.

    op.create_table(
        "personality_embedding",
        sa.Column(
            "entity_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey(
                "entity.canonical_id",
                ondelete="CASCADE",
                name="fk_personality_embedding_entity_id_entity",
            ),
            primary_key=True,
        ),
        sa.Column("embedding", Vector(_EMBEDDING_DIM), nullable=False),
        sa.Column("model_version", sa.String(128), nullable=False),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )
    # HNSW index for cosine similarity. The pgvector docs recommend
    # building the index *after* bulk inserts when seeding; for a fresh
    # migration on an empty table this just lights up the access method.
    op.execute(
        "CREATE INDEX ix_personality_embedding_embedding_hnsw "
        "ON personality_embedding USING hnsw (embedding vector_cosine_ops)"
    )

    op.create_table(
        "transcript_chunk_embedding",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
        ),
        # Deliberately no FK constraint — see migration docstring.
        sa.Column("media_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("chunk_idx", sa.Integer(), nullable=False),
        sa.Column("chunk_text", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(_EMBEDDING_DIM), nullable=False),
        sa.Column("model_version", sa.String(128), nullable=False),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.UniqueConstraint(
            "media_id",
            "chunk_idx",
            name="uq_transcript_chunk_embedding_media_chunk",
        ),
    )
    # Index media_id for the cleanup path (DELETE WHERE media_id = $1
    # before re-inserting on a Whisper re-run); the unique constraint
    # above also covers ``media_id`` as the leading column but we
    # keep the standalone index so a future migration that drops the
    # unique constraint (e.g. allowing chunk versioning) doesn't
    # regress lookup performance.
    op.create_index(
        "ix_transcript_chunk_embedding_media_id",
        "transcript_chunk_embedding",
        ["media_id"],
    )
    op.execute(
        "CREATE INDEX ix_transcript_chunk_embedding_embedding_hnsw "
        "ON transcript_chunk_embedding USING hnsw (embedding vector_cosine_ops)"
    )


def downgrade() -> None:
    # HNSW indexes drop with their tables; explicit DROP INDEX is
    # cheap insurance against a future rename of the access method.
    op.execute("DROP INDEX IF EXISTS ix_transcript_chunk_embedding_embedding_hnsw")
    op.drop_index(
        "ix_transcript_chunk_embedding_media_id",
        table_name="transcript_chunk_embedding",
    )
    op.drop_table("transcript_chunk_embedding")

    op.execute("DROP INDEX IF EXISTS ix_personality_embedding_embedding_hnsw")
    op.drop_table("personality_embedding")

    # Deliberately *not* dropping the ``vector`` extension — migration 0001
    # owns it and a later migration may still need vector columns elsewhere.
