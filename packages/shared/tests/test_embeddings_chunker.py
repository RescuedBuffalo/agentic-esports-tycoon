"""Unit tests for the BUF-28 transcript chunker."""

from __future__ import annotations

import pytest
from esports_sim.embeddings.chunker import DEFAULT_CHUNK_TOKENS, chunk_transcript


def test_empty_string_returns_no_chunks() -> None:
    assert chunk_transcript("") == []


def test_whitespace_only_returns_no_chunks() -> None:
    """A transcript that is purely whitespace produces no chunks (no words to split)."""
    assert chunk_transcript("   \n\t  ") == []


def test_single_chunk_when_under_budget() -> None:
    text = "the quick brown fox jumps over the lazy dog"
    [chunk] = chunk_transcript(text, chunk_tokens=20)
    # Whitespace-collapsed but content-preserving.
    assert chunk == "the quick brown fox jumps over the lazy dog"


def test_splits_into_chunks_of_target_size() -> None:
    text = " ".join(str(i) for i in range(1050))
    chunks = chunk_transcript(text, chunk_tokens=500)
    assert len(chunks) == 3
    assert len(chunks[0].split()) == 500
    assert len(chunks[1].split()) == 500
    assert len(chunks[2].split()) == 50  # remainder


def test_invalid_chunk_size_rejected() -> None:
    with pytest.raises(ValueError):
        chunk_transcript("anything", chunk_tokens=0)


def test_default_chunk_size_fits_minilm_context() -> None:
    """The default whitespace-token budget must stay under MiniLM's
    256-wordpiece input limit. English wordpiece-tokenises at ~1.3x
    word count; budgets above ~196 risk silent truncation that
    degrades retrieval quality (see BUF-28 PR #26 review).
    """
    assert DEFAULT_CHUNK_TOKENS <= 196
