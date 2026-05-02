"""Direct tests for the ``Connector`` ABC + ``IngestionRecord`` DTO (BUF-9).

The runner trusts both contracts implicitly — `IngestionRecord` is the
DTO the connector hands the resolver, and `Connector` is the interface
the runner walks. Each connector test file exercises these by accident
through end-to-end pipelines, but a regression in the contract itself
(e.g. relaxing ``extra="forbid"``, dropping a column-length cap) would
silently shrink the validation surface across all connectors at once.
This file pins those contracts down.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timedelta
from typing import Any

import pytest
from data_pipeline.connector import Connector, IngestionRecord, RateLimit
from esports_sim.db.enums import EntityType, Platform
from pydantic import ValidationError

# --- IngestionRecord -----------------------------------------------------


class TestIngestionRecord:
    def test_round_trip_with_all_fields_set(self) -> None:
        rec = IngestionRecord(
            entity_type=EntityType.PLAYER,
            platform_id="riot:tenz",
            platform_name="TenZ",
            payload={"team": "SEN", "country": "CA"},
        )
        assert rec.entity_type == EntityType.PLAYER
        assert rec.platform_id == "riot:tenz"
        assert rec.platform_name == "TenZ"
        assert rec.payload == {"team": "SEN", "country": "CA"}

    def test_platform_id_min_length_rejected(self) -> None:
        # ``platform_id`` is the resolver's exact-match key; an empty
        # string would silently merge unrelated rows.
        with pytest.raises(ValidationError):
            IngestionRecord(
                entity_type=EntityType.PLAYER,
                platform_id="",
                platform_name="x",
                payload={},
            )

    def test_platform_id_max_length_matches_alias_column(self) -> None:
        # 255 chars is the alias-table cap; emitting longer surfaces here
        # rather than as a Postgres truncation that would lose context.
        with pytest.raises(ValidationError):
            IngestionRecord(
                entity_type=EntityType.PLAYER,
                platform_id="x" * 256,
                platform_name="x",
                payload={},
            )

    def test_platform_name_min_length_rejected(self) -> None:
        with pytest.raises(ValidationError):
            IngestionRecord(
                entity_type=EntityType.PLAYER,
                platform_id="riot:tenz",
                platform_name="",
                payload={},
            )

    def test_platform_name_max_length_matches_alias_column(self) -> None:
        with pytest.raises(ValidationError):
            IngestionRecord(
                entity_type=EntityType.PLAYER,
                platform_id="riot:tenz",
                platform_name="y" * 256,
                payload={},
            )

    def test_extra_fields_rejected(self) -> None:
        # ``model_config = ConfigDict(extra="forbid")``: a connector
        # silently dropping a column-name typo is exactly the regression
        # this catches.
        with pytest.raises(ValidationError):
            IngestionRecord(
                entity_type=EntityType.PLAYER,
                platform_id="riot:tenz",
                platform_name="TenZ",
                payload={},
                country="CA",  # type: ignore[call-arg]
            )

    def test_frozen_after_construction(self) -> None:
        rec = IngestionRecord(
            entity_type=EntityType.PLAYER,
            platform_id="riot:tenz",
            platform_name="TenZ",
            payload={},
        )
        with pytest.raises(ValidationError):
            rec.platform_id = "different"  # type: ignore[misc]

    def test_payload_accepts_arbitrary_json_shape(self) -> None:
        # ``payload`` is dict[str, Any] by design — the runner persists
        # it verbatim on raw_record / staging_record. No schema checks.
        rec = IngestionRecord(
            entity_type=EntityType.TEAM,
            platform_id="vlr:101",
            platform_name="Sentinels",
            payload={"nested": {"region": "americas"}, "rank": 1, "active": True},
        )
        assert rec.payload["nested"]["region"] == "americas"

    def test_entity_type_must_be_known_enum_value(self) -> None:
        # An unknown EntityType string surfaces as a ValidationError; the
        # alternative — a silent string passthrough — would let drifting
        # connectors mint rows the rest of the pipeline can't process.
        with pytest.raises(ValidationError):
            IngestionRecord(
                entity_type="not-a-real-type",  # type: ignore[arg-type]
                platform_id="x",
                platform_name="x",
                payload={},
            )


# --- RateLimit -----------------------------------------------------------


class TestRateLimit:
    """A handful of guard-rail tests; broader coverage lives in test_rate_limiter.py."""

    def test_capacity_below_one_rejected(self) -> None:
        with pytest.raises(ValueError, match="capacity"):
            RateLimit(capacity=0, refill_per_second=1.0)

    def test_negative_capacity_rejected(self) -> None:
        with pytest.raises(ValueError, match="capacity"):
            RateLimit(capacity=-5, refill_per_second=1.0)

    def test_refill_zero_rejected(self) -> None:
        with pytest.raises(ValueError, match="refill"):
            RateLimit(capacity=1, refill_per_second=0.0)

    def test_refill_negative_rejected(self) -> None:
        with pytest.raises(ValueError, match="refill"):
            RateLimit(capacity=1, refill_per_second=-1.0)

    def test_frozen_dataclass_cannot_be_mutated(self) -> None:
        rl = RateLimit(capacity=10, refill_per_second=1.0)
        with pytest.raises(Exception):  # noqa: B017 - FrozenInstanceError
            rl.capacity = 99  # type: ignore[misc]


# --- Connector ABC -------------------------------------------------------


def test_connector_cannot_be_instantiated_directly() -> None:
    """Connector is abstract; constructing it must raise TypeError."""
    with pytest.raises(TypeError):
        Connector()  # type: ignore[abstract]


def test_subclass_missing_required_property_cannot_instantiate() -> None:
    """A partial implementation should fail at construction.

    The runner has no business calling ``source_name`` on a connector
    that forgot to declare it; Python's ABCMeta enforces this for us.
    """

    class _Incomplete(Connector):
        # Missing source_name, platform, entity_types, cadence, rate_limit.
        def fetch(self, since: datetime) -> Iterable[dict[str, Any]]:
            return iter(())

        def validate(self, raw_payload: dict[str, Any]) -> dict[str, Any]:
            return raw_payload

        def transform(self, validated_payload: dict[str, Any]) -> Iterable[IngestionRecord]:
            return iter(())

    with pytest.raises(TypeError):
        _Incomplete()  # type: ignore[abstract]


def test_subclass_missing_required_method_cannot_instantiate() -> None:
    """A subclass that declares the metadata but omits ``fetch`` is also abstract."""

    class _NoFetch(Connector):
        @property
        def source_name(self) -> str:
            return "x"

        @property
        def platform(self) -> Platform:
            return Platform.VLR

        @property
        def entity_types(self) -> tuple[EntityType, ...]:
            return (EntityType.PLAYER,)

        @property
        def cadence(self) -> timedelta:
            return timedelta(hours=1)

        @property
        def rate_limit(self) -> RateLimit:
            return RateLimit(capacity=1, refill_per_second=1.0)

        # Missing fetch / validate / transform.

    with pytest.raises(TypeError):
        _NoFetch()  # type: ignore[abstract]


def test_fully_implemented_subclass_instantiates_and_exposes_metadata() -> None:
    """Smoke-test the ABC by walking the public surface end-to-end.

    A regression that drops a property from the ABC would surface as a
    missing attribute here, well before a real connector hits prod.
    """

    class _OK(Connector):
        @property
        def source_name(self) -> str:
            return "vlr"

        @property
        def platform(self) -> Platform:
            return Platform.VLR

        @property
        def entity_types(self) -> tuple[EntityType, ...]:
            return (EntityType.PLAYER, EntityType.TEAM)

        @property
        def cadence(self) -> timedelta:
            return timedelta(hours=6)

        @property
        def rate_limit(self) -> RateLimit:
            return RateLimit(capacity=10, refill_per_second=2.0)

        def fetch(self, since: datetime) -> Iterable[dict[str, Any]]:
            yield {"id": "1"}

        def validate(self, raw_payload: dict[str, Any]) -> dict[str, Any]:
            return raw_payload

        def transform(self, validated_payload: dict[str, Any]) -> Iterable[IngestionRecord]:
            yield IngestionRecord(
                entity_type=EntityType.PLAYER,
                platform_id=str(validated_payload["id"]),
                platform_name="x",
                payload=validated_payload,
            )

    connector = _OK()
    assert connector.source_name == "vlr"
    assert connector.platform is Platform.VLR
    assert EntityType.PLAYER in connector.entity_types
    assert connector.cadence == timedelta(hours=6)
    assert connector.rate_limit.capacity == 10

    payloads = list(connector.fetch(datetime(2026, 1, 1)))
    assert payloads == [{"id": "1"}]

    records = list(connector.transform(connector.validate(payloads[0])))
    assert len(records) == 1
    assert records[0].platform_id == "1"
    assert records[0].entity_type == EntityType.PLAYER
