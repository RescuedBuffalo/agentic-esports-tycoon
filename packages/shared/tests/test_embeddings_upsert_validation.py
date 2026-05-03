"""Unit tests for the BUF-28 upsert helpers' input validation.

These don't need a live Postgres — the dim check fires before the
session is ever touched. Keep here so a refactor that loses the
guard surfaces as a fast-running unit-test fail rather than as an
opaque pgvector cast error against a real DB.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from typing import Any
from unittest.mock import MagicMock

import pytest
from esports_sim.embeddings.embedder import Embedder
from esports_sim.embeddings.upsert import (
    upsert_personality_embedding,
    upsert_transcript_chunks,
)


class _BadDimEmbedder:
    """Returns the wrong vector width to exercise the validator."""

    def __init__(self, dim: int) -> None:
        self._dim = dim

    @property
    def model_version(self) -> str:
        return "bad-dim@v1"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [[0.0] * self._dim for _ in texts]


class _CountMismatchEmbedder:
    """Returns a different number of vectors than texts."""

    @property
    def model_version(self) -> str:
        return "bad-count@v1"

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [[0.0] * 384 for _ in range(len(texts) + 1)]


def test_personality_upsert_rejects_wrong_dim() -> None:
    session: Any = MagicMock()
    embedder: Embedder = _BadDimEmbedder(dim=128)
    with pytest.raises(ValueError, match="vector\\(384\\)"):
        upsert_personality_embedding(
            session,
            entity_id=uuid.uuid4(),
            text="anything",
            embedder=embedder,
        )
    session.execute.assert_not_called()


def test_transcript_upsert_rejects_wrong_dim() -> None:
    session: Any = MagicMock()
    embedder: Embedder = _BadDimEmbedder(dim=512)
    with pytest.raises(ValueError, match="vector\\(384\\)"):
        upsert_transcript_chunks(
            session,
            media_id=uuid.uuid4(),
            chunks=["one chunk"],
            embedder=embedder,
        )
    session.execute.assert_not_called()


def test_transcript_upsert_rejects_count_mismatch() -> None:
    session: Any = MagicMock()
    embedder: Embedder = _CountMismatchEmbedder()
    with pytest.raises(RuntimeError, match="2 vectors for 1 chunks"):
        upsert_transcript_chunks(
            session,
            media_id=uuid.uuid4(),
            chunks=["one chunk"],
            embedder=embedder,
        )
    session.execute.assert_not_called()


def test_transcript_upsert_empty_input_short_circuits() -> None:
    """Empty chunks list returns 0 without ever calling embed/execute."""
    session: Any = MagicMock()

    class _ExplodingEmbedder:
        @property
        def model_version(self) -> str:
            return "should-not-load"

        def embed(self, texts: Sequence[str]) -> list[list[float]]:
            raise AssertionError("embed should not be called for empty chunks")

    embedder: Embedder = _ExplodingEmbedder()
    result = upsert_transcript_chunks(
        session,
        media_id=uuid.uuid4(),
        chunks=[],
        embedder=embedder,
    )
    assert result == 0
    session.execute.assert_not_called()
