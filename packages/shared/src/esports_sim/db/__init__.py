"""Database layer (BUF-6 + BUF-13 + BUF-83).

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

The ``Base`` metadata is exported so Alembic and tests can introspect the
schema. Migrations live at ``packages/shared/alembic/``.
"""

from esports_sim.db.base import Base
from esports_sim.db.enums import EntityType, Platform, ReviewStatus, StagingStatus
from esports_sim.db.models import (
    AliasReviewQueue,
    Entity,
    EntityAlias,
    PatchEra,
    PatchNote,
    RawRecord,
    StagingInvariantError,
    StagingRecord,
    TemporalBleedError,
)

__all__ = [
    "AliasReviewQueue",
    "Base",
    "Entity",
    "EntityAlias",
    "EntityType",
    "PatchEra",
    "PatchNote",
    "Platform",
    "RawRecord",
    "ReviewStatus",
    "StagingInvariantError",
    "StagingRecord",
    "StagingStatus",
    "TemporalBleedError",
]
