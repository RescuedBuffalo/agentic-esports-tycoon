"""Embedder protocol + default sentence-transformers implementation (BUF-28).

The personality extractor (BUF-25) and the Whisper pipeline (BUF-21)
both call into an :class:`Embedder` to convert text into the 384-dim
vectors stored in ``personality_embedding`` / ``transcript_chunk_embedding``.

Why a Protocol rather than instantiating sentence-transformers
directly: every test that touches the embedding store would otherwise
have to load 90 MB of model weights, and we'd couple ``packages/shared``
to a heavy optional dependency at import time. Tests inject a
deterministic stub (e.g. a bag-of-characters → 384-dim hash) and the
production path uses :class:`SentenceTransformerEmbedder`.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    # Type-only: keep ``sentence_transformers`` out of the import graph
    # for callers that only need the protocol or the default factory's
    # signature.
    from sentence_transformers import SentenceTransformer


# The embedder identity recorded in ``model_version`` when the default
# sentence-transformers loader is used. ADR-006 standardises on this
# model; rotation is a deliberate operation that touches every row.
DEFAULT_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"


@runtime_checkable
class Embedder(Protocol):
    """The minimal surface a vector producer has to expose.

    ``model_version`` is the string written into the embedding row so
    a future model rotation can detect mixed populations. Callers
    must treat it as opaque — comparing it requires the same model
    semantics, not just the same string.
    """

    @property
    def model_version(self) -> str: ...

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one float vector per input text, in the same order."""


class SentenceTransformerEmbedder:
    """Default :class:`Embedder` backed by sentence-transformers.

    The constructor accepts an already-loaded
    :class:`~sentence_transformers.SentenceTransformer` for tests +
    ahead-of-time loading; if ``None``, the model is loaded on first
    :meth:`embed` call (lazy so import-time cost stays low).

    ``model_version`` is set from the ``model_name_or_path`` so a
    swap of the default model surfaces in the embedding rows the
    next time the upsert helpers are called.
    """

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_MODEL_NAME,
        model: SentenceTransformer | None = None,
    ) -> None:
        self._model_name = model_name
        self._model: SentenceTransformer | None = model

    @property
    def model_version(self) -> str:
        return self._model_name

    def _load(self) -> SentenceTransformer:
        if self._model is not None:
            return self._model
        # Lazy import: pulling ``sentence_transformers`` drags in
        # torch + transformers, both of which we don't want in the
        # ``packages/shared`` import graph for callers that only need
        # the schema models.
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self._model_name)
        return self._model

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        model = self._load()
        # ``encode`` returns numpy arrays; tolist() avoids leaking the
        # numpy type into the upsert path so the helper signature stays
        # ``list[float]`` (which pgvector binds cleanly).
        encoded = model.encode(
            list(texts),
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return [list(row) for row in encoded.tolist()]
