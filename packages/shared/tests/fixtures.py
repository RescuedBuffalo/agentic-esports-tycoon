"""Factory helpers for the BUF-6 schema.

Tiny by design — these are *seed* helpers, not a full Faker-style factory
library. Each function returns a model instance with sensible defaults that
callers override via kwargs.
"""

from __future__ import annotations

import hashlib
import uuid
from datetime import UTC, datetime
from typing import Any

from esports_sim.db.enums import EntityType, Platform, ReviewStatus, StagingStatus
from esports_sim.db.models import (
    AliasReviewQueue,
    Entity,
    EntityAlias,
    RawRecord,
    StagingRecord,
)


def make_entity(
    *,
    canonical_id: uuid.UUID | None = None,
    entity_type: EntityType = EntityType.PLAYER,
    is_active: bool = True,
) -> Entity:
    return Entity(
        canonical_id=canonical_id or uuid.uuid4(),
        entity_type=entity_type,
        is_active=is_active,
    )


def make_entity_alias(
    *,
    entity: Entity,
    platform: Platform = Platform.VLR,
    platform_id: str = "vlr-1234",
    platform_name: str = "TenZ",
    confidence: float = 1.0,
    verified_at: datetime | None = None,
    valid_from: datetime | None = None,
) -> EntityAlias:
    # Set ``canonical_id`` explicitly as well as the relationship — the FK
    # column is None on an unattached row otherwise, and the DTO requires it.
    return EntityAlias(
        id=uuid.uuid4(),
        entity=entity,
        canonical_id=entity.canonical_id,
        platform=platform,
        platform_id=platform_id,
        platform_name=platform_name,
        confidence=confidence,
        verified_at=verified_at,
        valid_from=valid_from or datetime.now(UTC),
    )


def make_staging_record(
    *,
    source: str = "vlr",
    entity_type: EntityType = EntityType.PLAYER,
    canonical_id: uuid.UUID | None = None,
    payload: dict[str, Any] | None = None,
    status: StagingStatus = StagingStatus.PENDING,
) -> StagingRecord:
    return StagingRecord(
        id=uuid.uuid4(),
        source=source,
        entity_type=entity_type,
        canonical_id=canonical_id,
        payload=payload or {"name": "TenZ"},
        status=status,
    )


def make_raw_record(
    *,
    source: str = "vlr",
    payload: dict[str, Any] | None = None,
    content_hash: str | None = None,
) -> RawRecord:
    body = payload or {"raw": "blob"}
    return RawRecord(
        id=uuid.uuid4(),
        source=source,
        payload=body,
        content_hash=content_hash
        or hashlib.sha256(repr(sorted(body.items())).encode()).hexdigest(),
    )


def make_review_queue_item(
    *,
    platform: Platform = Platform.VLR,
    platform_id: str = "vlr-9999",
    platform_name: str = "TenZ",
    candidates: list[dict[str, Any]] | None = None,
    reason: str = "fuzzy_below_auto_merge",
    status: ReviewStatus = ReviewStatus.PENDING,
) -> AliasReviewQueue:
    return AliasReviewQueue(
        id=uuid.uuid4(),
        platform=platform,
        platform_id=platform_id,
        platform_name=platform_name,
        candidates=candidates
        or [{"canonical_id": str(uuid.uuid4()), "name": "tenz", "score": 0.78}],
        reason=reason,
        status=status,
    )
