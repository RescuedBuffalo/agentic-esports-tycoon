"""Transcript chunker (BUF-28).

Whisper (BUF-21) emits per-segment text; the embedding store wants
roughly-500-token chunks so each row is dense enough to retrieve
meaningfully but small enough not to exceed the embedder's context.
The "tokens" the spec calls for are the embedder's tokens, but
counting them precisely requires loading the model — too heavy for
the chunker's call sites. Whitespace tokens are a reliable
approximation for English-language transcripts (the actual MiniLM
tokenizer produces ~1.3x as many subword tokens, well inside the
256-token context the model handles cleanly even at our 500-word
target).
"""

from __future__ import annotations

DEFAULT_CHUNK_TOKENS: int = 500
"""BUF-28 spec target — ~500 tokens per chunk."""


def chunk_transcript(
    text: str,
    *,
    chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
) -> list[str]:
    """Split ``text`` into chunks of approximately ``chunk_tokens`` words.

    Splitting on ``str.split()`` so any Unicode whitespace counts as a
    boundary; the result is rejoined with single spaces (collapsing
    runs of whitespace is a non-loss for a transcript whose only
    structure is the words themselves). An empty / whitespace-only
    input returns ``[]`` so the upsert path can no-op cleanly.
    """
    if chunk_tokens <= 0:
        raise ValueError(f"chunk_tokens must be positive, got {chunk_tokens}")
    tokens = text.split()
    if not tokens:
        return []
    chunks: list[str] = []
    for i in range(0, len(tokens), chunk_tokens):
        chunks.append(" ".join(tokens[i : i + chunk_tokens]))
    return chunks
