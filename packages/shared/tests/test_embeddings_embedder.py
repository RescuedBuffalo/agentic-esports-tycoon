"""Unit tests for the BUF-28 :class:`Embedder` protocol + default loader.

The actual sentence-transformers model isn't loaded here — we don't
want a 90 MB download in CI. The :class:`Embedder` protocol is the
contract the pipeline uses; we verify the protocol holds and that
the default loader's ``model_version`` plumbing is correct via an
injected fake model.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from esports_sim.embeddings.embedder import (
    DEFAULT_MODEL_NAME,
    Embedder,
    SentenceTransformerEmbedder,
)


class _FakeSentenceTransformer:
    """Stand-in for ``SentenceTransformer`` so tests don't pull torch in.

    Returns a deterministic unit-norm 384-dim vector per input — what
    matters here is the shape contract and the ``encode`` signature,
    not the embedding semantics.
    """

    def encode(
        self,
        texts: list[str],
        *,
        normalize_embeddings: bool = True,  # noqa: ARG002 - signature compat
        show_progress_bar: bool = False,  # noqa: ARG002 - signature compat
    ) -> Any:
        import numpy as np

        # ``encode`` returns numpy arrays in production; mimic that
        # shape so the embedder's ``.tolist()`` path is exercised.
        return np.array([[float(i % 7) / 7.0] * 384 for i in range(len(texts))])


def test_default_model_name_matches_adr() -> None:
    """ADR-006 standardises on all-MiniLM-L6-v2; rotation is deliberate."""
    assert DEFAULT_MODEL_NAME == "sentence-transformers/all-MiniLM-L6-v2"


def test_sentence_transformer_embedder_satisfies_protocol() -> None:
    embedder = SentenceTransformerEmbedder(model=_FakeSentenceTransformer())
    assert isinstance(embedder, Embedder)


def test_model_version_reflects_constructor_argument() -> None:
    embedder = SentenceTransformerEmbedder(
        model_name="custom/embedder@v3",
        model=_FakeSentenceTransformer(),
    )
    assert embedder.model_version == "custom/embedder@v3"


def test_embed_returns_one_vector_per_input() -> None:
    embedder = SentenceTransformerEmbedder(model=_FakeSentenceTransformer())
    out = embedder.embed(["alpha", "beta", "gamma"])
    assert len(out) == 3
    assert all(len(vec) == 384 for vec in out)


def test_embed_empty_input_short_circuits() -> None:
    """Empty input must not load the model — the upsert helpers depend
    on this so they can no-op cleanly without forcing a model download.
    """

    class _ExplodingLoader(SentenceTransformerEmbedder):
        def _load(self) -> Any:
            raise AssertionError("model should not be loaded for empty input")

    embedder = _ExplodingLoader()
    assert embedder.embed([]) == []


def test_protocol_accepts_lightweight_implementations() -> None:
    """Sanity check: a pipeline can build a deterministic test
    embedder without touching ``sentence-transformers`` at all.
    """

    class _Stub:
        @property
        def model_version(self) -> str:
            return "test-stub@v1"

        def embed(self, texts: Sequence[str]) -> list[list[float]]:
            return [[0.0] * 384 for _ in texts]

    stub: Embedder = _Stub()
    assert isinstance(stub, Embedder)
    assert stub.model_version == "test-stub@v1"
    assert stub.embed(["hi"]) == [[0.0] * 384]
