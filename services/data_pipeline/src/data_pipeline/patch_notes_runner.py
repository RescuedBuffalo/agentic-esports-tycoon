"""Patch-notes ingestion: connector ABC + runner (BUF-83).

Patch notes don't fit the ``Connector`` / ``IngestionRecord`` /
``run_ingestion`` flow. That pipeline is built around resolver-eligible
entities keyed by ``(platform, platform_id)`` aliases — it persists raw
blobs, runs entity resolution, and writes staging rows. Patch notes are
documents: keyed by ``patch_version``, never fuzzy-matched, never merged.
Squeezing them into ``IngestionRecord`` would mean either inventing a
fake ``platform_id`` or wiring zero-record transforms past the resolver,
both of which lose the typed-column queryability that ``patch_note``
gives BUF-24 downstream.

This module mirrors the BUF-9 connector contract for patch-note sources
and provides a single-pass orchestrator that writes ``PatchNote`` rows
directly. The metadata-property surface (``source_name`` / ``platform`` /
``cadence`` / ``rate_limit``) is identical to :class:`Connector` so the
scheduler can list the two ABCs uniformly; the per-record payload type
is the only thing that differs.

Design notes:

* :class:`PatchNoteConnector` is a parallel ABC rather than a
  ``Connector`` subclass — that keeps mypy happy without ``# type: ignore``
  shims. The two ABCs share *no* runtime state; the scheduler is expected
  to dispatch based on which type each registered connector implements.
* :class:`PatchNoteRecord` is local to this module rather than living in
  :mod:`data_pipeline.connector`. Locality wins here: the only callers
  are :class:`PatchNoteConnector` and :func:`run_patch_notes_ingestion`.
* The runner UPSERTs on ``patch_version`` so re-running the connector is
  idempotent — a re-fetch refreshes ``raw_html``/``body_text`` and bumps
  ``fetched_at`` rather than inserting a duplicate row.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import structlog
from esports_sim.db.models import PatchNote
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import select
from sqlalchemy.orm import Session
from structlog.stdlib import BoundLogger

from data_pipeline.connector import RateLimit
from data_pipeline.errors import SchemaDriftError, TransientFetchError
from data_pipeline.rate_limiter import TokenBucket


class PatchNoteRecord(BaseModel):
    """One patch-note article projected into the columns we persist.

    Mirrors the ``PatchNote`` SQLAlchemy model's writable fields. The
    runner uses this DTO as the in-memory shape between
    ``connector.transform`` and the ``PatchNote`` UPSERT — keeping the
    boundary typed means a connector that drops a column surfaces as a
    pydantic ``ValidationError`` rather than a None landing in the DB.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Match the column lengths in ``models.py`` so a connector emitting an
    # over-long version surfaces here, not as a Postgres ``StringDataRight
    # Truncation`` that would lose the offending row's context.
    patch_version: str = Field(min_length=1, max_length=32)
    published_at: datetime
    raw_html: str = Field(min_length=1)
    body_text: str = Field(min_length=1)
    url: str = Field(min_length=1, max_length=512)


class PatchNoteConnector(ABC):
    """Parallel of :class:`~data_pipeline.connector.Connector` for documents.

    Same metadata surface (``source_name`` / ``cadence`` / ``rate_limit``)
    so the scheduler can list patch-note connectors alongside entity
    connectors. ``platform`` is intentionally absent: patch notes aren't
    aliased, so attributing them to a :class:`Platform` enum value would
    be a category error. Subclasses declare ``source_name`` (e.g.
    ``"playvalorant"``) and the scheduler tags rows with that string.

    The fetch -> validate -> transform split mirrors the entity connector
    contract for the same reasons: ``validate`` raises
    :class:`~data_pipeline.errors.SchemaDriftError` when the upstream
    shape changes, ``transform`` is then free to assume the validated
    shape.
    """

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Free-form identifier persisted on ``patch_note``-adjacent logs.

        Conventionally lowercase, snake_case, and stable across releases.
        """

    @property
    @abstractmethod
    def cadence(self) -> timedelta:
        """How often the scheduler should re-invoke this connector."""

    @property
    @abstractmethod
    def rate_limit(self) -> RateLimit:
        """Per-source HTTP rate limit; honoured by the runner via a token bucket."""

    @abstractmethod
    def fetch(self, since: datetime) -> Iterable[dict[str, Any]]:
        """Yield raw upstream payloads modified after ``since``.

        Implementations should pre-filter by published-date metadata in
        the article-list page so we don't drill into article bodies for
        patches we already have. Each yielded dict is a JSON-serialisable
        envelope (e.g. ``{"url": ..., "html": ...}``) handed straight to
        ``validate``.
        """

    @abstractmethod
    def validate(self, raw_payload: dict[str, Any]) -> dict[str, Any]:
        """Reject malformed payloads.

        Return a (possibly normalised) payload on success; raise
        :class:`SchemaDriftError` to skip + log this row. Anything else
        bubbles up as a fatal connector error.
        """

    @abstractmethod
    def transform(self, validated_payload: dict[str, Any]) -> Iterable[PatchNoteRecord]:
        """Project a validated upstream payload into one or more
        :class:`PatchNoteRecord`s.

        Most patch-note connectors yield exactly one record per article,
        but the iterable shape leaves headroom for multi-version
        announcement posts.
        """


@dataclass
class PatchNotesStats:
    """Counters returned from :func:`run_patch_notes_ingestion`.

    ``upserted`` covers both inserts and updates — distinguishing them
    cleanly from one statement requires a ``RETURNING`` round-trip per
    row, which isn't worth the chattiness for a weekly-cadence connector.
    Operators who need the split can derive it from the row's
    ``fetched_at`` vs the current run's start timestamp.
    """

    fetched: int = 0
    schema_drifts: int = 0
    transient_errors: int = 0
    upserted: int = 0
    by_version: dict[str, int] = field(default_factory=dict)


def run_patch_notes_ingestion(
    connector: PatchNoteConnector,
    *,
    session: Session,
    since: datetime,
    rate_limiter: TokenBucket | None = None,
    logger: BoundLogger | None = None,
) -> PatchNotesStats:
    """Drive one patch-notes ingestion pass.

    The caller owns the transaction — the runner ``flush``es so dedup
    queries see fresh inserts but does not ``commit``. Wrap in
    ``session.begin()`` for all-or-nothing semantics; the default
    "commit-per-run" policy is the scheduler's job.

    Idempotency is enforced by UPSERT on ``patch_version``: a re-scrape of
    the same article overwrites ``raw_html``/``body_text``/``url`` and
    bumps ``fetched_at`` rather than inserting a duplicate row. This
    means re-running the connector against an unchanged source is safe
    and observable (``fetched_at`` advances).
    """
    rate_limiter = rate_limiter or TokenBucket.from_rate_limit(connector.rate_limit)
    base_logger = logger or structlog.get_logger("data_pipeline.patch_notes")
    bound = base_logger.bind(source=connector.source_name)

    stats = PatchNotesStats()
    bound.info("patch_notes.start", since=since.isoformat())

    # Bracket each ``next()`` with the limiter exactly the way the entity
    # runner does; same justification (the connector's HTTP call happens
    # in the body that runs *before* the next yield, so acquiring inside
    # the loop body would be after-the-fact).
    iterator = iter(connector.fetch(since))
    while True:
        rate_limiter.acquire()
        try:
            raw_payload = next(iterator)
        except StopIteration:
            break
        except TransientFetchError as exc:
            # The connector's ``fetch`` makes its HTTP call in the body
            # that runs *between* yields — i.e. during this very
            # ``next()`` call. A network blip on the article-list page
            # (or on an article body page) would otherwise escape this
            # loop and abort the whole patch-notes pass. Catching here
            # mirrors the post-yield handling below: count it, log it,
            # and keep iterating so the rest of the archive still
            # flows. The next scheduled run retries the same upstream
            # because no raw_record was written for the failed page.
            stats.transient_errors += 1
            bound.warning(
                "patch_notes.transient_error",
                code="TRANSIENT_ERROR",
                stage="fetch",
                detail=str(exc),
            )
            continue

        log = bound.bind(url=raw_payload.get("url"))

        try:
            validated = connector.validate(raw_payload)
            records = list(connector.transform(validated))
        except TransientFetchError as exc:
            stats.transient_errors += 1
            log.warning(
                "patch_notes.transient_error",
                code="TRANSIENT_ERROR",
                detail=str(exc),
            )
            continue
        except SchemaDriftError as exc:
            stats.schema_drifts += 1
            log.warning(
                "patch_notes.schema_drift",
                code="SCHEMA_DRIFT",
                detail=str(exc),
            )
            continue
        except Exception:
            # Unknown connector failure — log with full context, then
            # re-raise. Same fatal-by-default policy as the entity
            # runner: an unexpected failure mode must not silently
            # corrupt the document store.
            log.exception("patch_notes.connector_error", code="CONNECTOR_ERROR")
            raise

        stats.fetched += 1

        for record in records:
            _upsert_patch_note(session, source=connector.source_name, record=record)
            stats.upserted += 1
            stats.by_version[record.patch_version] = (
                stats.by_version.get(record.patch_version, 0) + 1
            )
            log.info(
                "patch_notes.upserted",
                source=connector.source_name,
                patch_version=record.patch_version,
                published_at=record.published_at.isoformat(),
            )

    bound.info(
        "patch_notes.done",
        fetched=stats.fetched,
        upserted=stats.upserted,
        schema_drifts=stats.schema_drifts,
        transient_errors=stats.transient_errors,
    )
    return stats


def _upsert_patch_note(
    session: Session,
    *,
    source: str,
    record: PatchNoteRecord,
) -> PatchNote:
    """UPSERT semantics on ``(source, patch_version)``.

    SELECT-then-INSERT-or-UPDATE rather than ``INSERT ... ON CONFLICT``
    because the latter would skip the SQLAlchemy ORM event hooks (none
    today on ``PatchNote``, but cheap insurance for the day there are).

    The dedup key is composite: two patch-note connectors (e.g. one
    per game) can legitimately emit the same ``patch_version`` string
    without colliding. The ``source`` argument comes from the calling
    connector's ``source_name``; the runner threads it through so the
    DTO (:class:`PatchNoteRecord`) stays connector-agnostic.

    ``fetched_at`` is server-default ``now()`` on insert and explicitly
    bumped to ``datetime.now(UTC)`` on update so operators can tell
    when each row was last touched.
    """
    existing: PatchNote | None = session.execute(
        select(PatchNote).where(
            PatchNote.source == source,
            PatchNote.patch_version == record.patch_version,
        )
    ).scalar_one_or_none()

    if existing is None:
        new = PatchNote(
            source=source,
            patch_version=record.patch_version,
            published_at=record.published_at,
            raw_html=record.raw_html,
            body_text=record.body_text,
            url=record.url,
        )
        session.add(new)
        session.flush()
        return new

    existing.published_at = record.published_at
    existing.raw_html = record.raw_html
    existing.body_text = record.body_text
    existing.url = record.url
    # Bump ``fetched_at`` on update so the freshness check observes the
    # re-scrape, even if the upstream content hasn't changed. We use the
    # process clock (UTC) rather than ``func.now()`` so the assignment is
    # visible on the in-memory ORM object before flush — the entity-runner
    # tests rely on this pattern too.
    existing.fetched_at = datetime.now(tz=UTC)
    session.flush()
    return existing


__all__ = [
    "PatchNoteConnector",
    "PatchNoteRecord",
    "PatchNotesStats",
    "run_patch_notes_ingestion",
]
