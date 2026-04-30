"""``run_ingestion`` — the orchestrator every connector flows through (BUF-9).

The pipeline for one connector pass::

    fetch -> dedup -> persist raw -> validate -> transform
                                   -> resolve_entity -> persist staging

The runner is intentionally the only place this sequence lives. A new
source is a new :class:`Connector` subclass; the runner is unchanged.

Per-record errors (schema drift, resolver failure, transient fetch error)
are logged and skipped: the rest of the run continues. Anything else
bubbles up — Systems-spec error taxonomy says fatal-by-default for the
unexpected, so an unknown failure mode doesn't quietly corrupt the
canonical store.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import structlog
from esports_sim.db.enums import StagingStatus
from esports_sim.db.models import RawRecord, StagingRecord
from esports_sim.resolver import ResolutionStatus, ResolveResult, resolve_entity
from sqlalchemy import select
from sqlalchemy.orm import Session
from structlog.stdlib import BoundLogger

from data_pipeline.connector import Connector, IngestionRecord
from data_pipeline.errors import IngestionError, SchemaDriftError, TransientFetchError
from data_pipeline.rate_limiter import TokenBucket


@dataclass
class IngestionStats:
    """Counters returned from :func:`run_ingestion` for observability.

    Per-record outcomes increment exactly one counter so the totals add
    up; ``processed`` further breaks down by resolver status to keep the
    auto-merge / pending / matched ratio visible without reaching into
    structured logs.
    """

    fetched: int = 0
    duplicates: int = 0
    schema_drifts: int = 0
    transient_errors: int = 0
    resolver_errors: int = 0
    processed: int = 0
    by_status: dict[str, int] = field(default_factory=dict)


def run_ingestion(
    connector: Connector,
    *,
    session: Session,
    since: datetime,
    rate_limiter: TokenBucket | None = None,
    logger: BoundLogger | None = None,
) -> IngestionStats:
    """Drive one ingestion pass for ``connector`` against the given session.

    The caller owns the transaction — the runner ``flush``es as it goes
    (so dedup queries see fresh inserts) but does not ``commit``. Wrap
    ``run_ingestion`` in your own ``session.begin()`` if you want all-or-
    nothing semantics; the default policy is "commit per run", which the
    scheduler implements by closing the session at the end.
    """
    rate_limiter = rate_limiter or TokenBucket.from_rate_limit(connector.rate_limit)
    base_logger = logger or structlog.get_logger("data_pipeline.ingestion")
    bound = base_logger.bind(source=connector.source_name)

    stats = IngestionStats()
    bound.info("ingestion.start", since=since.isoformat())

    # Rate-limit *around* each ``next()`` rather than inside the loop body
    # — a connector that issues its HTTP request before yielding (the
    # documented contract) would otherwise run unmetered, with the limiter
    # acting as after-the-fact bookkeeping. ``_rate_limited`` brackets each
    # iterator advance with an ``acquire`` so the bucket gates real upstream
    # traffic.
    for raw_payload in _rate_limited(connector.fetch(since), rate_limiter):
        # Scope the dedup hash by ``source``: two connectors emitting
        # byte-identical JSON for the same upstream record (e.g., a
        # primary VLR scraper plus a backup that happens to ship the
        # same shape) must each land their own raw row, otherwise the
        # second source is silently dropped before resolver/staging
        # ever see it.
        content_hash = _hash_payload(connector.source_name, raw_payload)
        log = bound.bind(content_hash=content_hash)

        if _content_hash_seen(session, content_hash):
            stats.duplicates += 1
            log.debug("ingestion.duplicate")
            continue

        # Run validate + transform *before* persisting raw. The split
        # matters for retryability:
        #   * SchemaDriftError is permanent (the upstream source has
        #     changed shape) — persist raw so a maintainer can re-parse
        #     offline against ``raw_record`` rather than re-fetching.
        #   * TransientFetchError is recoverable (timeout, 5xx, transient
        #     parse glitch) — must NOT persist raw, otherwise the
        #     content_hash dedups the same payload on the next run and
        #     the temporary blip becomes a permanent drop.
        # Materialise transform's output so an exception raised mid-
        # iteration is caught here rather than escaping the runner; once
        # transform raises, half-translated records would corrupt staging.
        try:
            validated = connector.validate(raw_payload)
            records = list(connector.transform(validated))
        except TransientFetchError as exc:
            stats.transient_errors += 1
            log.warning("ingestion.transient_error", code="TRANSIENT_ERROR", detail=str(exc))
            continue
        except SchemaDriftError as exc:
            _persist_raw(
                session,
                source=connector.source_name,
                payload=raw_payload,
                content_hash=content_hash,
            )
            stats.fetched += 1
            stats.schema_drifts += 1
            log.warning("ingestion.schema_drift", code="SCHEMA_DRIFT", detail=str(exc))
            continue
        except Exception:
            # Unknown connector failure (parser regression, KeyError on a
            # field that disappeared, etc.). Per the connector contract
            # this is fatal-by-default — but we still owe operators the
            # structured ``CONNECTOR_ERROR`` event with the same
            # ``source`` + ``content_hash`` context every other path
            # emits, otherwise postmortems lose the failing-payload trail.
            # Log first, then re-raise so the run stops.
            log.exception("ingestion.connector_error", code="CONNECTOR_ERROR")
            raise

        # Happy path: persist raw, then resolve and write a staging row
        # per emitted record.
        _persist_raw(
            session,
            source=connector.source_name,
            payload=raw_payload,
            content_hash=content_hash,
        )
        stats.fetched += 1

        for record in records:
            _process_record(
                session,
                connector=connector,
                record=record,
                stats=stats,
                log=log,
            )

    bound.info(
        "ingestion.done",
        fetched=stats.fetched,
        duplicates=stats.duplicates,
        schema_drifts=stats.schema_drifts,
        resolver_errors=stats.resolver_errors,
        processed=stats.processed,
    )
    return stats


# --- helpers ---------------------------------------------------------------


def _process_record(
    session: Session,
    *,
    connector: Connector,
    record: IngestionRecord,
    stats: IngestionStats,
    log: BoundLogger,
) -> None:
    """Resolve one record and write its staging row.

    Per-record skips are limited to the documented per-record family
    (:class:`IngestionError` and its subclasses). Anything else — most
    notably ``IntegrityError`` and other SQLAlchemy faults that
    ``resolve_entity`` deliberately re-raises — is a genuine data-layer
    failure and propagates so the run stops. Per the module docstring's
    "fatal-by-default for the unexpected" rule: silently swallowing a
    DB inconsistency would let a botched run continue and leave the
    canonical store half-written.
    """
    try:
        result = resolve_entity(
            session,
            platform=connector.platform,
            platform_id=record.platform_id,
            platform_name=record.platform_name,
            entity_type=record.entity_type,
        )
    except IngestionError as exc:
        # Connector-side IngestionError leaked out of validate — already
        # logged under ``RESOLVER_ERROR``; skip the staging write but
        # keep the run alive.
        stats.resolver_errors += 1
        log.warning(
            "ingestion.resolver_error",
            code="RESOLVER_ERROR",
            detail=str(exc),
            platform_id=record.platform_id,
        )
        return

    staging = _staging_row_from_result(
        source=connector.source_name,
        record=record,
        result=result,
    )
    staging.save(session)
    session.flush()
    stats.processed += 1
    stats.by_status[result.status.value] = stats.by_status.get(result.status.value, 0) + 1
    log.info(
        "ingestion.processed",
        status=result.status.value,
        canonical_id=str(result.canonical_id) if result.canonical_id else None,
        platform_id=record.platform_id,
    )


def _staging_row_from_result(
    *,
    source: str,
    record: IngestionRecord,
    result: ResolveResult,
) -> StagingRecord:
    """Map a :class:`ResolveResult` onto a staging row.

    PENDING (review-band fuzzy match) → status=review with null
    canonical_id; everything else → status=processed with the resolver's
    canonical_id. The :meth:`StagingRecord.save` invariant catches the
    one combination this mapping must never produce — processed + null —
    which would only happen if a resolver code path silently returned a
    null canonical for a non-PENDING status.
    """
    if result.status is ResolutionStatus.PENDING:
        return StagingRecord(
            source=source,
            entity_type=record.entity_type,
            canonical_id=None,
            payload=record.payload,
            status=StagingStatus.REVIEW,
        )
    return StagingRecord(
        source=source,
        entity_type=record.entity_type,
        canonical_id=result.canonical_id,
        payload=record.payload,
        status=StagingStatus.PROCESSED,
    )


def _rate_limited(
    iterable: Iterable[dict[str, Any]],
    limiter: TokenBucket,
) -> Iterator[dict[str, Any]]:
    """Bracket each ``next()`` on ``iterable`` with ``limiter.acquire()``.

    The runner can't put ``acquire`` inside a plain ``for`` body: Python
    advances the iterator (running the connector's ``fetch`` body up to
    the next ``yield``, which is where the upstream HTTP call happens)
    *before* the body runs. Acquiring here — between iterator advances
    — keeps the bucket gating actual upstream traffic, not after-the-
    fact bookkeeping.

    On end-of-stream, the final ``next()`` may have made an HTTP call
    that yielded nothing (paginated connectors legitimately fetch an
    empty page and ``return``; "is there new data?" pollers do the
    same). We deliberately do **not** refund the token on
    ``StopIteration`` — refunding would let repeated empty polls
    bypass the limiter and exceed provider quotas. The cost is one
    "wasted" token per pass for materialised-list connectors that do
    no HTTP after the last yield; that under-utilisation is an
    acceptable trade for never under-charging on real polls.
    """
    iterator = iter(iterable)
    while True:
        limiter.acquire()
        try:
            yield next(iterator)
        except StopIteration:
            return


def _hash_payload(source: str, payload: dict[str, Any]) -> str:
    """SHA-256 over (source, canonical JSON). Stable across Python runs.

    ``sort_keys=True`` and explicit separators give us a deterministic
    byte sequence — a payload with the same content but different key
    order must hash the same, otherwise dedup is useless.

    ``source`` is folded into the hash input so two connectors that
    happen to emit byte-identical JSON each get a distinct
    ``content_hash``. Without that, the second connector's payload
    would be deduplicated against the first and silently dropped
    before the resolver ran — losing the alias and staging rows it
    would have produced.
    """
    encoded = json.dumps(
        {"source": source, "payload": payload},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _content_hash_seen(session: Session, content_hash: str) -> bool:
    """Has a raw_record with this hash already been persisted?"""
    stmt = select(RawRecord.id).where(RawRecord.content_hash == content_hash).limit(1)
    return session.execute(stmt).first() is not None


def _persist_raw(
    session: Session,
    *,
    source: str,
    payload: dict[str, Any],
    content_hash: str,
) -> RawRecord:
    raw = RawRecord(source=source, payload=payload, content_hash=content_hash)
    session.add(raw)
    session.flush()
    return raw


__all__ = ["IngestionStats", "run_ingestion"]
