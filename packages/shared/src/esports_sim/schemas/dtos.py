"""Pydantic DTOs for the BUF-6 schema.

The Systems-spec rule: downstream code imports these DTOs and never reaches
into raw SQL. The :class:`pydantic.BaseModel` instances are frozen so a DTO
that escapes one layer cannot be mutated by another.

``model_config = from_attributes=True`` lets us instantiate a DTO directly
from a SQLAlchemy ORM row (``EntityDTO.model_validate(entity_row)``).
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from esports_sim.db.enums import EntityType, Platform, ReviewStatus, StagingStatus


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", from_attributes=True)


class EntityDTO(_Frozen):
    canonical_id: uuid.UUID
    entity_type: EntityType
    created_at: datetime
    is_active: bool = True


class EntityAliasDTO(_Frozen):
    id: uuid.UUID
    canonical_id: uuid.UUID
    platform: Platform
    platform_id: str = Field(min_length=1, max_length=255)
    platform_name: str = Field(min_length=1, max_length=255)
    # Confidence is the resolver's match score, bounded so a Liquipedia seed
    # at 1.0 and a fuzzy auto-merge at 0.92 are comparable across rows.
    confidence: float = Field(ge=0.0, le=1.0)
    verified_at: datetime | None = None
    valid_from: datetime


class StagingRecordDTO(_Frozen):
    id: uuid.UUID
    source: str = Field(min_length=1, max_length=64)
    entity_type: EntityType
    canonical_id: uuid.UUID | None = None
    payload: dict[str, Any]
    status: StagingStatus = StagingStatus.PENDING
    created_at: datetime


class RawRecordDTO(_Frozen):
    id: uuid.UUID
    source: str = Field(min_length=1, max_length=64)
    fetched_at: datetime
    payload: dict[str, Any]
    # SHA-256 hex is 64 chars; we keep the column wide enough for that and
    # narrower variants (BLAKE2b-256 etc.) so the producer picks the algo.
    content_hash: str = Field(min_length=1, max_length=64)


class AliasReviewQueueDTO(_Frozen):
    id: uuid.UUID
    platform: Platform
    platform_id: str = Field(min_length=1, max_length=255)
    platform_name: str = Field(min_length=1, max_length=255)
    candidates: list[dict[str, Any]]
    reason: str = Field(min_length=1, max_length=255)
    status: ReviewStatus = ReviewStatus.PENDING
    created_at: datetime


class PatchEraDTO(_Frozen):
    """BUF-13 patch-era partition. Half-open: ``[start_date, end_date)``."""

    era_id: uuid.UUID
    era_slug: str = Field(min_length=1, max_length=32)
    patch_version: str = Field(min_length=1, max_length=32)
    start_date: datetime
    end_date: datetime | None = None
    meta_magnitude: float = Field(ge=0.0, le=1.0)
    is_major_shift: bool = False
    created_at: datetime


__all__ = [
    "EntityDTO",
    "EntityAliasDTO",
    "StagingRecordDTO",
    "RawRecordDTO",
    "AliasReviewQueueDTO",
    "PatchEraDTO",
]
