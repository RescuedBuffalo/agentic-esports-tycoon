"""Vector-similarity layer for personality summaries and transcripts (BUF-28).

ADR-006 puts these tables on the same Postgres instance as the rest of
the schema so cross-entity queries ("similar to X *and* on an active
T1 roster *and* plays duelist") are SQL JOINs rather than payload
filters in a separate vector store. The public surface here is small
on purpose:

* :class:`Embedder` — protocol the personality / Whisper pipelines
  hold to. The default :class:`SentenceTransformerEmbedder`
  lazy-loads ``sentence-transformers/all-MiniLM-L6-v2``; tests inject
  a deterministic stub instead of pulling 90 MB of model weights.
* :func:`chunk_transcript` — token-budgeted chunker the Whisper
  pipeline calls before embedding.
* :func:`upsert_personality_embedding` /
  :func:`upsert_transcript_chunks` — INSERT ... ON CONFLICT helpers
  so re-runs of the upstream extraction stay idempotent.
* :func:`similar_players` — the kNN-with-cross-entity-filter helper
  the BUF-28 acceptance criteria call out.
"""

from esports_sim.embeddings.chunker import chunk_transcript
from esports_sim.embeddings.embedder import (
    DEFAULT_MODEL_NAME,
    Embedder,
    SentenceTransformerEmbedder,
)
from esports_sim.embeddings.queries import (
    SimilarPlayer,
    SimilarPlayerNotFoundError,
    similar_players,
)
from esports_sim.embeddings.upsert import (
    upsert_personality_embedding,
    upsert_transcript_chunks,
)

__all__ = [
    "DEFAULT_MODEL_NAME",
    "Embedder",
    "SentenceTransformerEmbedder",
    "SimilarPlayer",
    "SimilarPlayerNotFoundError",
    "chunk_transcript",
    "similar_players",
    "upsert_personality_embedding",
    "upsert_transcript_chunks",
]
