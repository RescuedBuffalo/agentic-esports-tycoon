"""Add media_record + transcript tables and back-fill the BUF-28 FK (BUF-21).

System 04 in the Systems-spec: local Whisper transcription. Two new
tables plus one constraint:

* ``media_record`` — one row per upstream audio/video artifact. The
  Whisper batch worker pulls rows that don't have a child
  :class:`~esports_sim.db.models.Transcript` and produces one. Re-
  ingest of the same Twitch VOD or YouTube upload UPSERTs on
  ``(source, source_uri)`` rather than minting a duplicate row.
* ``transcript`` — one row per ``media_id``. Holds the full text plus
  the per-segment list Whisper emits. The ``text`` column is what
  BUF-21's acceptance ("transcripts searchable via SQL on the
  transcript column") targets; a future migration can layer a
  tsvector + GIN on top if proper full-text search is needed.
* ``fk_transcript_chunk_embedding_media_id_media_record`` — the FK
  the BUF-28 docstring (migration 0009) explicitly deferred. Now
  that ``media_record`` exists, we can wire ``ON DELETE CASCADE``
  so a deleted media row drops its embedded chunks atomically.

Schema rationale:

* ``media_record.source_uri`` is ``VARCHAR(1024)`` rather than
  ``Text`` so the unique-index footprint stays bounded — Twitch and
  YouTube URLs sit well under 100 chars and tightening below 1024
  would risk truncating a redirect chain.
* ``media_record.entity_id`` is ``ON DELETE SET NULL``, not CASCADE.
  The audio file itself is still useful as raw corpus after the
  canonical it pointed at gets merged or deleted; CASCADE would
  drop the recorded transcription unnecessarily.
* ``transcript.media_id`` is the primary key (one transcript per
  media, ``ON DELETE CASCADE``). If a future workflow wants
  multiple transcripts per media (e.g., diarization A/B), promote
  ``(media_id, model_version)`` to the unique key in a follow-up.
* ``media_kind`` is a Postgres ENUM rather than a free-form string
  because the audio-vs-video distinction drives the worker's demux
  step; a typo'd writer would land a row the worker silently skips.
* The BUF-28 FK back-fill assumes ``transcript_chunk_embedding`` is
  empty in any environment that runs this migration — true today
  because BUF-28 landed days ago and the only producer is the
  Whisper worker this PR introduces. If a future deploy carries
  orphan rows, that's a data bug worth surfacing as a constraint
  failure rather than silently dropping them on add.

Revision ID: 0011
Revises: 0010
Create Date: 2026-05-03
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0011"
down_revision: str | None = "0010"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


# Mirror of ``esports_sim.db.enums.MediaKind``. Kept as literals (not
# imported) so the migration keeps applying even if the Python enum
# is later refactored — same convention the BUF-6 enums use.
_MEDIA_KINDS = ("audio", "video")


def upgrade() -> None:
    postgresql.ENUM(
        *_MEDIA_KINDS,
        name="media_kind",
    ).create(op.get_bind(), checkfirst=True)

    op.create_table(
        "media_record",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("source", sa.String(64), nullable=False),
        sa.Column("source_uri", sa.String(1024), nullable=False),
        sa.Column("local_path", sa.String(1024), nullable=False),
        sa.Column(
            "media_kind",
            postgresql.ENUM(
                *_MEDIA_KINDS,
                name="media_kind",
                create_type=False,
            ),
            nullable=False,
        ),
        sa.Column("duration_seconds", sa.Float(), nullable=True),
        sa.Column("language", sa.String(8), nullable=True),
        sa.Column(
            "entity_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey(
                "entity.canonical_id",
                ondelete="SET NULL",
                name="fk_media_record_entity_id_entity",
            ),
            nullable=True,
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
        # Dedup key: a re-ingest of the same Twitch VOD lands on the
        # same row. UPSERTing on this constraint is the runtime
        # contract the connector layer relies on.
        sa.UniqueConstraint(
            "source",
            "source_uri",
            name="uq_media_record_source_source_uri",
        ),
    )
    # ``source`` filter shows up in operator queries
    # (``WHERE source = 'twitch_vod'``); index it standalone so the
    # leading-column trick on the unique constraint doesn't have to
    # carry it.
    op.create_index(
        "ix_media_record_source",
        "media_record",
        ["source"],
    )
    # ``entity_id`` is the join the personality-summary feature
    # extractor reads ("all media for this player"); without an index
    # that join falls back to a seq-scan once the table grows past a
    # few thousand rows.
    op.create_index(
        "ix_media_record_entity_id",
        "media_record",
        ["entity_id"],
    )

    op.create_table(
        "transcript",
        sa.Column(
            "media_id",
            postgresql.UUID(as_uuid=True),
            sa.ForeignKey(
                "media_record.id",
                ondelete="CASCADE",
                name="fk_transcript_media_id_media_record",
            ),
            primary_key=True,
        ),
        sa.Column("language", sa.String(8), nullable=False),
        sa.Column("model_version", sa.String(64), nullable=False),
        # Full concatenated transcript text. The column BUF-21's
        # acceptance criterion targets — searchable with ILIKE today,
        # promotable to a tsvector + GIN later if usage demands it.
        sa.Column("text", sa.Text(), nullable=False),
        # Per-segment list as Whisper emits it; preserved as JSONB so
        # downstream feature extractors don't have to re-run the model.
        sa.Column("segments", postgresql.JSONB, nullable=False),
        sa.Column("duration_seconds", sa.Float(), nullable=False),
        sa.Column("wallclock_seconds", sa.Float(), nullable=False),
        sa.Column("transcript_path", sa.String(1024), nullable=True),
        sa.Column(
            "transcribed_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
    )

    # Back-fill the FK that migration 0009 deliberately deferred. The
    # constraint is added here rather than in 0009 because it points
    # at ``media_record``, which only exists after this migration's
    # ``create_table`` above.
    op.create_foreign_key(
        "fk_transcript_chunk_embedding_media_id_media_record",
        source_table="transcript_chunk_embedding",
        referent_table="media_record",
        local_cols=["media_id"],
        remote_cols=["id"],
        ondelete="CASCADE",
    )


def downgrade() -> None:
    # Drop the BUF-28 FK first so ``media_record`` becomes droppable
    # without the constraint blocking it.
    op.drop_constraint(
        "fk_transcript_chunk_embedding_media_id_media_record",
        "transcript_chunk_embedding",
        type_="foreignkey",
    )

    op.drop_table("transcript")

    op.drop_index("ix_media_record_entity_id", table_name="media_record")
    op.drop_index("ix_media_record_source", table_name="media_record")
    op.drop_table("media_record")

    op.execute("DROP TYPE IF EXISTS media_kind")
