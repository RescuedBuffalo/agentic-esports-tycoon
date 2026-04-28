"""The :class:`Connector` ABC every scraper conforms to (BUF-9).

System 02's contract: a new source is a new ``Connector`` subclass — no
runner-side branching, no per-source orchestration. The lifecycle for one
upstream payload is::

    fetch  -> dedup -> validate -> transform -> resolve_entity -> staging

The connector owns the four hooks (``fetch``/``validate``/``transform``
plus the static metadata properties); the runner owns the rest. Splitting
``validate`` from ``transform`` keeps the schema-drift surface narrow:
``validate`` raises :class:`~data_pipeline.errors.SchemaDriftError` when
the upstream shape is wrong, ``transform`` is then free to assume the
shape it asked for.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from esports_sim.db.enums import EntityType, Platform
from pydantic import BaseModel, ConfigDict, Field


@dataclass(frozen=True)
class RateLimit:
    """Token-bucket parameters for a single upstream source.

    The runner constructs a :class:`~data_pipeline.rate_limiter.TokenBucket`
    from this and gates every ``fetch``-loop iteration through it. Kept as
    a frozen dataclass — not a pydantic model — because a connector's
    rate limit is a policy declaration, not an inbound payload.

    ``capacity`` controls the maximum burst size; ``refill_per_second``
    controls steady-state throughput. A 60-rpm endpoint with no burst
    tolerance is ``RateLimit(capacity=1, refill_per_second=1.0)``; one
    that allows a 10-call burst is ``RateLimit(capacity=10,
    refill_per_second=1.0)``.
    """

    capacity: int
    refill_per_second: float

    def __post_init__(self) -> None:
        if self.capacity < 1:
            raise ValueError("RateLimit.capacity must be >= 1")
        if self.refill_per_second <= 0:
            raise ValueError("RateLimit.refill_per_second must be > 0")


class IngestionRecord(BaseModel):
    """One resolver-ready unit a connector emits per upstream record.

    The runner doesn't reach into the upstream payload — it trusts the
    connector to project ``platform_id``/``platform_name``/``entity_type``
    deterministically. ``payload`` is whatever subset of the validated
    upstream blob the connector wants persisted on the staging row;
    typically the same shape downstream stages will read.

    A single upstream fetch can yield multiple :class:`IngestionRecord`s
    (e.g., one match payload yields ten player records), which is why
    ``transform`` returns an :class:`~collections.abc.Iterable`.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    entity_type: EntityType
    # Wider ranges than the alias schema's 255-char cap aren't useful — the
    # alias table will reject them anyway, surfacing the schema drift early.
    platform_id: str = Field(min_length=1, max_length=255)
    platform_name: str = Field(min_length=1, max_length=255)
    payload: dict[str, Any]


class Connector(ABC):
    """Single shape every scraper conforms to. Systems-spec System 02.

    Subclasses implement the abstract methods plus the four metadata
    properties. The runner is intentionally unaware of which connector
    it's driving — adding a source means dropping a new ``Connector``
    subclass and registering it; no edits to ``run_ingestion``.
    """

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Free-form identifier persisted on ``raw_record.source`` /
        ``staging_record.source``. Conventionally lowercase, snake_case,
        and stable across releases (changing it splits the audit trail)."""

    @property
    @abstractmethod
    def platform(self) -> Platform:
        """The :class:`Platform` enum the resolver should attribute aliases
        to. Distinct from ``source_name`` because two source crawlers can
        share a platform (e.g., a primary + backup VLR scraper)."""

    @property
    @abstractmethod
    def entity_types(self) -> tuple[EntityType, ...]:
        """Which entity classes this connector emits. Used by the
        scheduler/registry to pick connectors by what they produce; the
        runner doesn't enforce that emitted records stay inside this
        tuple, but a connector that drifts will surface in metrics."""

    @property
    @abstractmethod
    def cadence(self) -> timedelta:
        """How often the scheduler should re-invoke ``run_ingestion``.
        The runner itself is single-shot: this property is a hint for the
        outer orchestrator, not a sleep loop."""

    @property
    @abstractmethod
    def rate_limit(self) -> RateLimit:
        """Per-source HTTP rate limit. Honoured by the runner via a token
        bucket, not by the connector itself — keeping the limiter outside
        ``fetch`` means tests can swap a fake clock without per-connector
        plumbing."""

    @abstractmethod
    def fetch(self, since: datetime) -> Iterable[dict[str, Any]]:
        """Yield raw upstream payloads modified after ``since``.

        Implementations should be lazy (generator) so the runner can apply
        rate limiting per-yield rather than waiting for the whole crawl.
        Each yielded value must be JSON-serialisable: it lands verbatim in
        ``raw_record.payload`` for replay.
        """

    @abstractmethod
    def validate(self, raw_payload: dict[str, Any]) -> dict[str, Any]:
        """Reject malformed payloads.

        Return a (possibly normalised) payload on success; raise
        :class:`~data_pipeline.errors.SchemaDriftError` to skip this row
        and log ``SCHEMA_DRIFT`` against the connector. Anything else
        from this method bubbles up under a distinct ``CONNECTOR_ERROR``
        log event so the failure mode stays visible.
        """

    @abstractmethod
    def transform(self, validated_payload: dict[str, Any]) -> Iterable[IngestionRecord]:
        """Project a validated upstream payload into resolver inputs.

        One upstream record may produce zero or more
        :class:`IngestionRecord`s. ``platform_id`` must be deterministic
        — the resolver's exact-alias lookup keys on it, so a connector
        that flips between two ids for the same handle would split the
        canonical row.
        """


__all__ = ["Connector", "IngestionRecord", "RateLimit"]
