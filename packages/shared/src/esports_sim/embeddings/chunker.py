"""Transcript chunker (BUF-28).

Whisper (BUF-21) emits per-segment text; the embedding store wants
chunks dense enough to retrieve meaningfully but short enough that
the embedder doesn't truncate them. The BUF-28 spec wrote "~500
tokens" assuming a long-context embedder — but ADR-006 picks
``sentence-transformers/all-MiniLM-L6-v2``, whose model card pins a
hard 256-wordpiece input limit (anything longer is silently
truncated). English-language transcripts wordpiece-tokenize at
~1.3x word count, so the safe whitespace-token budget is
``floor(256 / 1.3) ≈ 196``. We round down to 180 to leave headroom
for dense names + numbers that tokenize denser than running prose.
A future migration to a longer-context embedder can lift this
without changing the schema.
"""

from __future__ import annotations

DEFAULT_CHUNK_TOKENS: int = 180
"""Whitespace tokens per chunk — sized to fit MiniLM's 256-wordpiece limit
with margin (see module docstring)."""


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
