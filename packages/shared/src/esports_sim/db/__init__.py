"""Database layer (BUF-6 + BUF-13 + BUF-83 + BUF-28).

The canonical+staging substrate every other System (02–10) joins on:

* :class:`Entity` — canonical id + entity_type, the join key everyone uses.
* :class:`EntityAlias` — many-to-one platform handles, unique per
  ``(platform, platform_id)`` so two scrapers can't accidentally split a
  player into two canonical rows.
* :class:`StagingRecord` — scraped payloads parked for the resolver (BUF-7).
* :class:`RawRecord` — content-addressed blob store; ``content_hash`` keeps
  re-fetches idempotent.
* :class:`AliasReviewQueue` — items the fuzzy matcher couldn't decide; humans
  drain it via BUF-16's CLI.
* :class:`PatchNote` — versioned release-notes documents (BUF-83).
* :class:`PatchEra` — temporal-partition rows (BUF-13). Every record with
  a timestamp resolves to one via :func:`esports_sim.eras.assign_era`;
  the runtime guard against cross-major-shift aggregation is
  :class:`TemporalBleedError`.
* :class:`PersonalityEmbedding` / :class:`TranscriptChunkEmbedding` —
  pgvector-backed similarity store (BUF-28, ADR-006). One personality
  row per entity, many transcript chunks per media. The cross-entity
  ``similar_players`` helper lives in :mod:`esports_sim.embeddings`.

The ``Base`` metadata is exported so Alembic and tests can introspect the
schema. Migrations live at ``packages/shared/alembic/``.
"""

from esports_sim.db.base import Base
from esports_sim.db.enums import (
    SYMMETRIC_RELATIONSHIP_EDGE_TYPES,
    EntityType,
    MediaKind,
    Platform,
    RelationshipEdgeType,
    ReviewStatus,
    StagingStatus,
)
from esports_sim.db.models import (
    EMBEDDING_DIM,
    AliasReviewQueue,
    Entity,
    EntityAlias,
    MediaRecord,
    PatchEra,
    PatchIntent,
    PatchNote,
    PersonalityEmbedding,
    RawRecord,
    RelationshipEdge,
    RelationshipEvent,
    StagingInvariantError,
    StagingRecord,
    TemporalBleedError,
    Transcript,
    TranscriptChunkEmbedding,
)

__all__ = [
    "AliasReviewQueue",
    "Base",
    "EMBEDDING_DIM",
    "Entity",
    "EntityAlias",
    "EntityType",
    "MediaKind",
    "MediaRecord",
    "PatchEra",
    "PatchIntent",
    "PatchNote",
    "PersonalityEmbedding",
    "Platform",
    "RawRecord",
    "RelationshipEdge",
    "RelationshipEdgeType",
    "RelationshipEvent",
    "ReviewStatus",
    "StagingInvariantError",
    "StagingRecord",
    "StagingStatus",
    "SYMMETRIC_RELATIONSHIP_EDGE_TYPES",
    "TemporalBleedError",
    "Transcript",
    "TranscriptChunkEmbedding",
]
