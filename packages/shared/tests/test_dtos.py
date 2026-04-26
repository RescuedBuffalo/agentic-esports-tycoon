"""Unit tests for the BUF-6 Pydantic DTOs.

These don't need a database — they exercise the validation rules and the
``from_attributes=True`` path that lets us construct a DTO directly from a
SQLAlchemy ORM row.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest
from esports_sim.db.enums import EntityType, Platform, ReviewStatus, StagingStatus
from esports_sim.schemas.dtos import (
    AliasReviewQueueDTO,
    EntityAliasDTO,
    EntityDTO,
    RawRecordDTO,
    StagingRecordDTO,
)
from pydantic import ValidationError

from tests.fixtures import (
    make_entity,
    make_entity_alias,
    make_raw_record,
    make_review_queue_item,
    make_staging_record,
)


def test_entity_dto_from_orm() -> None:
    e = make_entity(entity_type=EntityType.PLAYER)
    # `from_attributes=True` lets us validate straight off the ORM row,
    # provided the attributes are populated. Server defaults aren't applied
    # until the row is flushed, so for this unit test we set created_at
    # manually.
    e.created_at = datetime.now(UTC)
    dto = EntityDTO.model_validate(e)
    assert dto.canonical_id == e.canonical_id
    assert dto.entity_type is EntityType.PLAYER


def test_entity_dto_is_frozen() -> None:
    dto = EntityDTO(
        canonical_id=uuid.uuid4(),
        entity_type=EntityType.PLAYER,
        created_at=datetime.now(UTC),
    )
    with pytest.raises(ValidationError):
        dto.is_active = False  # type: ignore[misc]


def test_entity_alias_confidence_bounds() -> None:
    base = {
        "id": uuid.uuid4(),
        "canonical_id": uuid.uuid4(),
        "platform": Platform.VLR,
        "platform_id": "vlr-1234",
        "platform_name": "TenZ",
        "valid_from": datetime.now(UTC),
    }
    EntityAliasDTO(**base, confidence=0.0)
    EntityAliasDTO(**base, confidence=1.0)
    with pytest.raises(ValidationError):
        EntityAliasDTO(**base, confidence=-0.01)
    with pytest.raises(ValidationError):
        EntityAliasDTO(**base, confidence=1.01)


def test_entity_alias_dto_from_orm() -> None:
    e = make_entity()
    a = make_entity_alias(entity=e)
    dto = EntityAliasDTO.model_validate(a)
    assert dto.platform is Platform.VLR
    assert dto.platform_id == "vlr-1234"


def test_staging_record_default_status() -> None:
    sr = make_staging_record()
    sr.created_at = datetime.now(UTC)
    dto = StagingRecordDTO.model_validate(sr)
    assert dto.status is StagingStatus.PENDING


def test_staging_record_canonical_id_optional() -> None:
    """Pre-resolution rows don't have a canonical_id; the DTO must allow that."""
    dto = StagingRecordDTO(
        id=uuid.uuid4(),
        source="vlr",
        entity_type=EntityType.PLAYER,
        canonical_id=None,
        payload={"name": "TenZ"},
        status=StagingStatus.REVIEW,
        created_at=datetime.now(UTC),
    )
    assert dto.canonical_id is None


def test_raw_record_dto_from_orm() -> None:
    r = make_raw_record()
    r.fetched_at = datetime.now(UTC)
    dto = RawRecordDTO.model_validate(r)
    assert dto.source == "vlr"
    assert len(dto.content_hash) > 0


def test_review_queue_dto_from_orm() -> None:
    item = make_review_queue_item()
    item.created_at = datetime.now(UTC)
    dto = AliasReviewQueueDTO.model_validate(item)
    assert dto.status is ReviewStatus.PENDING
    assert dto.candidates and isinstance(dto.candidates[0], dict)


def test_extra_fields_rejected() -> None:
    """All DTOs use extra='forbid' so an unknown column fails loudly."""
    with pytest.raises(ValidationError):
        EntityDTO(
            canonical_id=uuid.uuid4(),
            entity_type=EntityType.PLAYER,
            created_at=datetime.now(UTC),
            bogus=True,  # type: ignore[call-arg]
        )
