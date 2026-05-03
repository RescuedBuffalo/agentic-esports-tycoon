"""Idempotent UPSERTs for the BUF-28 embedding tables.

Both writers (BUF-25 personality extraction, BUF-21 Whisper transcripts)
re-run on the same upstream input from time to time — a corrected
biography text, a re-transcribed audio file. The schema's uniqueness
keys (``personality_embedding.entity_id``,
``transcript_chunk_embedding.(media_id, chunk_idx)``) are the
idempotency anchors; these helpers wrap the dialect-specific
INSERT...ON CONFLICT pattern so call sites don't have to.

Single source of truth for the embedder identity: the row's
``model_version`` is taken from the :class:`Embedder` instance, not
from a constant — this is what lets a future model rotation re-embed
in place without touching the upsert helpers.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence

from sqlalchemy import delete, func
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from esports_sim.db.models import (
    EMBEDDING_DIM,
    PersonalityEmbedding,
    TranscriptChunkEmbedding,
)
from esports_sim.embeddings.embedder import Embedder


def _validate_dim(vector: Sequence[float], where: str) -> None:
    """Raise if a vector has the wrong width.

    pgvector's cast error at insert time is opaque ("expected N
    dimensions, not M"); a Python-side check produces a stack trace
    that points at the call site that fed the wrong embedder in.
    """
    if len(vector) != EMBEDDING_DIM:
        raise ValueError(
            f"{where}: expected vector(384), got vector({len(vector)}). "
            "Likely a mismatched embedder — every embedding column in this "
            "schema is 384-dim per ADR-006."
        )


def upsert_personality_embedding(
    session: Session,
    *,
    entity_id: uuid.UUID,
    text: str,
    embedder: Embedder,
) -> None:
    """Insert or replace the personality embedding for ``entity_id``.

    Conflicts on the ``entity_id`` primary key — re-running BUF-25 on
    the same player produces an UPDATE, not a duplicate. The row's
    ``updated_at`` is bumped server-side via the ``onupdate`` clause
    on the model.
    """
    [vector] = embedder.embed([text])
    _validate_dim(vector, "upsert_personality_embedding")
    stmt = (
        insert(PersonalityEmbedding)
        .values(
            entity_id=entity_id,
            embedding=vector,
            model_version=embedder.model_version,
        )
        .on_conflict_do_update(
            index_elements=["entity_id"],
            # ``updated_at`` is set explicitly because ON CONFLICT DO
            # UPDATE bypasses the ORM ``onupdate=func.now()`` hook —
            # that hook only fires for ORM-issued UPDATEs.
            set_={
                "embedding": vector,
                "model_version": embedder.model_version,
                "updated_at": func.now(),
            },
        )
    )
    session.execute(stmt)


def upsert_transcript_chunks(
    session: Session,
    *,
    media_id: uuid.UUID,
    chunks: Sequence[str],
    embedder: Embedder,
) -> int:
    """Insert/replace transcript-chunk embeddings for ``media_id``.

    Conflicts on ``(media_id, chunk_idx)``: a re-transcription that
    produces a slightly different chunking still upserts cleanly
    because ``chunk_idx`` is positional. After the UPSERT, any rows
    whose ``chunk_idx`` falls outside the new ``[0, len(chunks))``
    range are deleted, so a re-transcription that produces fewer
    chunks (after a Whisper segmentation change, for example)
    converges the table to the new state instead of leaving a stale
    tail that retrieval would surface as outdated text.

    Returns the number of chunks written. Empty input is a no-op
    that returns 0 — convenient for call sites that skip an empty
    transcript without branching. Note: empty input also leaves
    existing chunks untouched. A caller who genuinely wants to
    drop every chunk for a media (e.g., the upstream audio was
    pulled) must issue an explicit
    ``DELETE FROM transcript_chunk_embedding WHERE media_id = ...``
    so the intent is unambiguous.
    """
    if not chunks:
        return 0
    vectors = embedder.embed(chunks)
    if len(vectors) != len(chunks):
        # Defensive: a misbehaving embedder that drops or duplicates
        # rows would silently realign chunk_idx with the wrong text.
        raise RuntimeError(f"embedder returned {len(vectors)} vectors for {len(chunks)} chunks")
    payload = []
    for idx, (chunk_text, vector) in enumerate(zip(chunks, vectors, strict=True)):
        _validate_dim(vector, f"upsert_transcript_chunks chunk {idx}")
        payload.append(
            {
                "id": uuid.uuid4(),
                "media_id": media_id,
                "chunk_idx": idx,
                "chunk_text": chunk_text,
                "embedding": vector,
                "model_version": embedder.model_version,
            }
        )
    stmt = insert(TranscriptChunkEmbedding).values(payload)
    # Bind ``excluded.*`` against the same statement so the UPDATE
    # clause references the row Postgres prepared from ``payload``,
    # not a newly constructed Insert.
    stmt = stmt.on_conflict_do_update(
        constraint="uq_transcript_chunk_embedding_media_chunk",
        set_={
            "chunk_text": stmt.excluded.chunk_text,
            "embedding": stmt.excluded.embedding,
            "model_version": stmt.excluded.model_version,
            # Same caveat as in :func:`upsert_personality_embedding`:
            # ON CONFLICT bypasses the ORM ``onupdate`` hook, so the
            # bump is explicit here.
            "updated_at": func.now(),
        },
    )
    session.execute(stmt)
    # Drop any stale tail from a previous longer transcription.
    # ``chunk_idx`` is a contiguous 0-based positional index, so
    # anything at or above ``len(chunks)`` is from a prior run and
    # would otherwise be returned by retrieval as stale text. The
    # DELETE is bounded to one media_id and runs after the UPSERT
    # so a partial failure rolls back together inside the caller's
    # transaction.
    session.execute(
        delete(TranscriptChunkEmbedding).where(
            TranscriptChunkEmbedding.media_id == media_id,
            TranscriptChunkEmbedding.chunk_idx >= len(chunks),
        )
    )
    return len(payload)
