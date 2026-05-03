"""Integration tests for the BUF-28 pgvector store.

Skipped without ``TEST_DATABASE_URL``. The fixtures load the same
deterministic stub :class:`Embedder` each test uses, so the cosine
ordering checks are reproducible without any model weights.
"""

from __future__ import annotations

import math
import uuid
from collections.abc import Sequence

import pytest
from esports_sim.db.enums import EntityType, Platform
from esports_sim.db.models import (
    EMBEDDING_DIM,
    PersonalityEmbedding,
    TranscriptChunkEmbedding,
)
from esports_sim.embeddings import (
    SimilarPlayerNotFoundError,
    similar_players,
    upsert_personality_embedding,
    upsert_transcript_chunks,
)
from esports_sim.embeddings.embedder import Embedder
from fixtures import make_entity, make_entity_alias
from sqlalchemy import inspect, select, text

pytestmark = pytest.mark.integration


def _unit(values: Sequence[float]) -> list[float]:
    """L2-normalise a vector so cosine distance reduces to a clean dot."""
    norm = math.sqrt(sum(v * v for v in values))
    if norm == 0:
        return list(values)
    return [v / norm for v in values]


def _direction(*, axis: int, dim: int = EMBEDDING_DIM) -> list[float]:
    """Unit vector pointing along axis ``axis``."""
    vec = [0.0] * dim
    vec[axis] = 1.0
    return vec


class _DictEmbedder:
    """Deterministic stub embedder.

    Looks up ``texts`` in a dict; raises if any text is missing so a
    test misconfiguration doesn't silently embed everything to the
    same vector.
    """

    def __init__(self, table: dict[str, list[float]], *, version: str = "test@v1") -> None:
        self._table = table
        self._version = version

    @property
    def model_version(self) -> str:
        return self._version

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._table[t] for t in texts]


@pytest.fixture
def embedder() -> Embedder:
    """Default test embedder — six known texts on three orthogonal axes."""
    return _DictEmbedder(
        {
            "duelist-aggressive": _direction(axis=0),
            "duelist-mechanical": _unit([0.95, 0.31] + [0.0] * (EMBEDDING_DIM - 2)),
            "duelist-flashy": _unit([0.88, 0.0, 0.47] + [0.0] * (EMBEDDING_DIM - 3)),
            "controller-cerebral": _direction(axis=1),
            "sentinel-stoic": _direction(axis=2),
            "initiator-loud": _direction(axis=3),
            "transcript-chunk-a": _direction(axis=4),
            "transcript-chunk-b": _direction(axis=5),
        }
    )


def test_migration_creates_embedding_tables(db_engine) -> None:
    """`alembic upgrade head` lands both BUF-28 tables."""
    insp = inspect(db_engine)
    tables = set(insp.get_table_names())
    assert "personality_embedding" in tables
    assert "transcript_chunk_embedding" in tables


def test_hnsw_index_present_on_each_embedding(db_engine) -> None:
    """HNSW indexes on each vector column are the BUF-28 latency contract."""
    with db_engine.connect() as conn:
        rows = conn.execute(text("""
                SELECT indexname, indexdef
                FROM pg_indexes
                WHERE tablename IN ('personality_embedding', 'transcript_chunk_embedding')
                """)).all()
    by_name = {r[0]: r[1] for r in rows}
    assert "ix_personality_embedding_embedding_hnsw" in by_name
    assert "ix_transcript_chunk_embedding_embedding_hnsw" in by_name
    # Sanity-check the access method on each: a future migration that
    # accidentally drops to a sequential index would still register
    # under this name.
    for indexname, definition in by_name.items():
        if indexname.endswith("_embedding_hnsw"):
            assert "USING hnsw" in definition
            assert "vector_cosine_ops" in definition


def test_personality_embedding_upsert_idempotent(db_session, embedder: Embedder) -> None:
    """Re-running the personality extractor UPSERTs in place."""
    e = make_entity(entity_type=EntityType.PLAYER)
    db_session.add(e)
    db_session.flush()

    upsert_personality_embedding(
        db_session,
        entity_id=e.canonical_id,
        text="duelist-aggressive",
        embedder=embedder,
    )
    db_session.flush()

    # Second pass with a different text → same row, new vector.
    upsert_personality_embedding(
        db_session,
        entity_id=e.canonical_id,
        text="controller-cerebral",
        embedder=embedder,
    )
    db_session.flush()

    rows = db_session.execute(select(PersonalityEmbedding)).scalars().all()
    assert len(rows) == 1
    assert rows[0].entity_id == e.canonical_id
    assert rows[0].model_version == "test@v1"


def test_personality_cascade_deletes_with_entity(db_session, embedder: Embedder) -> None:
    """Removing a canonical entity drops its stale embedding so the
    HNSW index can't return a result that points at a missing row.
    """
    e = make_entity(entity_type=EntityType.PLAYER)
    db_session.add(e)
    db_session.flush()
    upsert_personality_embedding(
        db_session,
        entity_id=e.canonical_id,
        text="duelist-aggressive",
        embedder=embedder,
    )
    db_session.flush()

    db_session.delete(e)
    db_session.flush()

    remaining = db_session.execute(select(PersonalityEmbedding)).scalars().all()
    assert remaining == []


def test_transcript_chunks_upsert_by_media_idx(db_session, embedder: Embedder) -> None:
    """`(media_id, chunk_idx)` is the dedup anchor; re-runs UPSERT in place."""
    media_id = uuid.uuid4()
    written = upsert_transcript_chunks(
        db_session,
        media_id=media_id,
        chunks=["transcript-chunk-a", "transcript-chunk-b"],
        embedder=embedder,
    )
    db_session.flush()
    assert written == 2

    # Re-run with the same texts → no new rows, same UUIDs preserved.
    upsert_transcript_chunks(
        db_session,
        media_id=media_id,
        chunks=["transcript-chunk-a", "transcript-chunk-b"],
        embedder=embedder,
    )
    db_session.flush()
    rows = (
        db_session.execute(
            select(TranscriptChunkEmbedding).where(TranscriptChunkEmbedding.media_id == media_id)
        )
        .scalars()
        .all()
    )
    assert len(rows) == 2
    assert {r.chunk_idx for r in rows} == {0, 1}


def test_transcript_empty_input_is_noop(db_session, embedder: Embedder) -> None:
    written = upsert_transcript_chunks(
        db_session,
        media_id=uuid.uuid4(),
        chunks=[],
        embedder=embedder,
    )
    assert written == 0


def _seed_player(
    session,
    *,
    handle: str,
    text: str,
    embedder: Embedder,
    is_active: bool = True,
):
    entity = make_entity(entity_type=EntityType.PLAYER, is_active=is_active)
    make_entity_alias(
        entity=entity,
        platform=Platform.VLR,
        platform_id=f"vlr-{handle}",
        platform_name=handle,
    )
    session.add(entity)
    session.flush()
    upsert_personality_embedding(
        session,
        entity_id=entity.canonical_id,
        text=text,
        embedder=embedder,
    )
    session.flush()
    return entity


def test_similar_players_resolves_alias_and_orders_by_distance(
    db_session, embedder: Embedder
) -> None:
    """The headline BUF-28 acceptance: ``similar_players("aspas", k=5)``
    returns plausibly-similar duelists from the roster pool.

    The stub embedder maps the duelist-flavour strings to vectors
    that are close to each other on the first axis and orthogonal
    to controller / sentinel / initiator. So the kNN result for
    "aspas" must be the two other duelists (in cosine order),
    never the off-axis players.
    """
    target = _seed_player(
        db_session,
        handle="aspas",
        text="duelist-aggressive",
        embedder=embedder,
    )
    near1 = _seed_player(
        db_session,
        handle="tenz",
        text="duelist-mechanical",
        embedder=embedder,
    )
    near2 = _seed_player(
        db_session,
        handle="yay",
        text="duelist-flashy",
        embedder=embedder,
    )
    _seed_player(
        db_session,
        handle="marved",
        text="controller-cerebral",
        embedder=embedder,
    )
    _seed_player(
        db_session,
        handle="chronicle",
        text="sentinel-stoic",
        embedder=embedder,
    )

    results = similar_players(db_session, "aspas", k=5)

    assert results[0].entity_id == near1.canonical_id  # duelist-mechanical (axis-0 0.95)
    assert results[1].entity_id == near2.canonical_id  # duelist-flashy (axis-0 0.88)
    # Distances are monotonically non-decreasing.
    distances = [r.distance for r in results]
    assert distances == sorted(distances)
    # Target is excluded from the result.
    assert all(r.entity_id != target.canonical_id for r in results)


def test_similar_players_where_sql_filters_cross_entity(db_session, embedder: Embedder) -> None:
    """The cross-entity-filter acceptance bullet: a ``where_sql`` on
    a column the relational schema already exposes (``e.is_active``)
    constrains the result. Once roster + role tables land (BUF-87+),
    callers will hit the same path with ``role='duelist' AND active=true``.
    """
    _seed_player(
        db_session,
        handle="aspas",
        text="duelist-aggressive",
        embedder=embedder,
    )
    active_match = _seed_player(
        db_session,
        handle="tenz",
        text="duelist-mechanical",
        embedder=embedder,
        is_active=True,
    )
    inactive_match = _seed_player(
        db_session,
        handle="yay",
        text="duelist-flashy",
        embedder=embedder,
        is_active=False,
    )

    results = similar_players(
        db_session,
        "aspas",
        k=5,
        where_sql="e.is_active = true",
    )
    ids = {r.entity_id for r in results}
    assert active_match.canonical_id in ids
    assert inactive_match.canonical_id not in ids


def test_similar_players_unknown_handle_raises(db_session) -> None:
    with pytest.raises(SimilarPlayerNotFoundError):
        similar_players(db_session, "nobody-by-that-name", k=3)


def test_similar_players_target_without_embedding_raises(db_session, embedder: Embedder) -> None:
    """A handle that resolves to a canonical id but has no personality
    embedding yet must surface as a structured error so the caller
    knows to wait for BUF-25 to extract that player.
    """
    e = make_entity(entity_type=EntityType.PLAYER)
    make_entity_alias(
        entity=e,
        platform=Platform.VLR,
        platform_id="vlr-newbie",
        platform_name="newbie",
    )
    db_session.add(e)
    db_session.flush()

    with pytest.raises(SimilarPlayerNotFoundError):
        similar_players(db_session, "newbie", k=3)


def test_similar_players_uuid_target_must_be_player(db_session, embedder: Embedder) -> None:
    """A non-player UUID with a personality embedding (legal at the
    schema level — the FK only points at ``entity``) must not be
    accepted as a similar_players target. The function compares
    player-shaped vectors; mixing in a team's vector would silently
    rank players against the wrong centroid.
    """
    team = make_entity(entity_type=EntityType.TEAM)
    db_session.add(team)
    db_session.flush()
    upsert_personality_embedding(
        db_session,
        entity_id=team.canonical_id,
        text="duelist-aggressive",
        embedder=embedder,
    )
    db_session.flush()

    with pytest.raises(SimilarPlayerNotFoundError, match="not a player"):
        similar_players(db_session, team.canonical_id, k=3)


def test_similar_players_unknown_uuid_raises(db_session) -> None:
    """A canonical UUID that doesn't exist in `entity` must surface
    as a structured error — silently returning zero results would
    make a typo in a caller indistinguishable from a real no-match.
    """
    with pytest.raises(SimilarPlayerNotFoundError, match="does not exist"):
        similar_players(db_session, uuid.uuid4(), k=3)
