"""Test doubles for the BUF-9 ingestion framework.

:class:`FakeConnector` is the shared scaffolding the runner tests drive.
It lets each test inject the upstream-payload sequence, the
validate/transform behaviour, and the rate-limit shape — so the same
runner harness exercises end-to-end happy path, schema drift, dedup,
and resolver errors without a real HTTP client.

Module is named ``pipeline_fixtures`` rather than the obvious
``fixtures`` to avoid colliding with ``packages/shared/tests/fixtures.py``
when both test directories end up on ``sys.path``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from datetime import datetime, timedelta
from typing import Any

from data_pipeline.connector import (
    Connector,
    IngestionRecord,
    RateLimit,
)
from esports_sim.db.enums import EntityType, Platform

# Module-level default keeps Bxxx happy (no function calls in argument
# defaults) and gives every FakeConnector that doesn't override the same
# generous bucket so tests don't have to think about timing.
_DEFAULT_RATE_LIMIT = RateLimit(capacity=100, refill_per_second=100.0)


class FakeConnector(Connector):
    """Minimal connector backed by an in-memory list of payloads.

    Each test wires its own ``validate`` / ``transform`` callables so
    behaviour shifts (drift, multi-record transforms, mocked HTTP) stay
    inline at the call site rather than buried in subclass overrides.
    """

    def __init__(
        self,
        *,
        payloads: list[dict[str, Any]],
        source_name: str = "fake",
        platform: Platform = Platform.VLR,
        entity_types: tuple[EntityType, ...] = (EntityType.PLAYER,),
        cadence: timedelta = timedelta(hours=1),
        rate_limit: RateLimit = _DEFAULT_RATE_LIMIT,
        validate_fn: Callable[[dict[str, Any]], dict[str, Any]] | None = None,
        transform_fn: Callable[[dict[str, Any]], Iterable[IngestionRecord]] | None = None,
    ) -> None:
        self._payloads = payloads
        self._source_name = source_name
        self._platform = platform
        self._entity_types = entity_types
        self._cadence = cadence
        self._rate_limit = rate_limit
        # Defaults are pass-through: validate is identity, transform reads
        # ``platform_id`` / ``platform_name`` straight off the payload.
        self._validate_fn = validate_fn or (lambda p: p)
        self._transform_fn = transform_fn or _default_transform

        # Surfaces for tests to assert against without poking internals.
        self.fetch_calls: list[datetime] = []

    @property
    def source_name(self) -> str:
        return self._source_name

    @property
    def platform(self) -> Platform:
        return self._platform

    @property
    def entity_types(self) -> tuple[EntityType, ...]:
        return self._entity_types

    @property
    def cadence(self) -> timedelta:
        return self._cadence

    @property
    def rate_limit(self) -> RateLimit:
        return self._rate_limit

    def fetch(self, since: datetime) -> Iterable[dict[str, Any]]:
        self.fetch_calls.append(since)
        # Yield (rather than ``return list``) so the runner observes lazy
        # iteration the way real connectors expose it.
        yield from self._payloads

    def validate(self, raw_payload: dict[str, Any]) -> dict[str, Any]:
        return self._validate_fn(raw_payload)

    def transform(self, validated_payload: dict[str, Any]) -> Iterable[IngestionRecord]:
        return self._transform_fn(validated_payload)


def _default_transform(payload: dict[str, Any]) -> Iterable[IngestionRecord]:
    """Identity transform: assume the payload already names a single record."""
    yield IngestionRecord(
        entity_type=EntityType(payload.get("entity_type", "player")),
        platform_id=payload["platform_id"],
        platform_name=payload["platform_name"],
        payload=payload,
    )


def make_player_payload(
    *,
    platform_id: str,
    platform_name: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a default-shaped upstream player payload.

    Tests override ``extra`` to differentiate hashes when they need
    multiple non-duplicate rows for the same handle (e.g. dedup tests).
    """
    body: dict[str, Any] = {
        "entity_type": "player",
        "platform_id": platform_id,
        "platform_name": platform_name,
    }
    if extra:
        body.update(extra)
    return body


__all__ = ["FakeConnector", "make_player_payload"]
