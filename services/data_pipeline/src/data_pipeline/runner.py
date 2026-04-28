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

    for raw_payload in connector.fetch(since):
        # Rate-limit per upstream record. Putting this on the *consumer*
        # side, not inside ``fetch``, means tests don't have to plumb a
        # fake clock through every connector subclass.
        rate_limiter.acquire()

        content_hash = _hash_payload(raw_payload)
        log = bound.bind(content_hash=content_hash)

        if _content_hash_seen(session, content_hash):
            stats.duplicates += 1
            log.debug("ingestion.duplicate")
            continue

        # Persist raw first. Even if validate/transform throws below, the
        # blob is on disk so a maintainer can re-run the parser offline
        # against ``raw_record`` rather than re-fetching upstream.
        _persist_raw(
            session,
            source=connector.source_name,
            payload=raw_payload,
            content_hash=content_hash,
        )
        stats.fetched += 1

        try:
            validated = connector.validate(raw_payload)
        except SchemaDriftError as exc:
            stats.schema_drifts += 1
            log.warning("ingestion.schema_drift", code="SCHEMA_DRIFT", detail=str(exc))
            continue
        except TransientFetchError as exc:
            stats.transient_errors += 1
            log.warning("ingestion.transient_error", code="TRANSIENT_ERROR", detail=str(exc))
            continue

        for record in connector.transform(validated):
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

    Resolver failures are logged and skipped, not raised: a single
    ambiguous handle shouldn't take down a 1000-row crawl. The raw row
    is already persisted, so the resolver can be re-run later.
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
        # Resolver shouldn't raise these, but a connector-side validate
        # could have leaked one out — bubble them as resolver errors so
        # the staging row isn't written.
        stats.resolver_errors += 1
        log.warning(
            "ingestion.resolver_error",
            code="RESOLVER_ERROR",
            detail=str(exc),
            platform_id=record.platform_id,
        )
        return
    except Exception as exc:
        # Catch-all for resolver bugs: log under a distinct event so
        # ``CONNECTOR_ERROR`` (connector misuse) and ``RESOLVER_ERROR``
        # (resolver crash) stay separable in dashboards.
        stats.resolver_errors += 1
        log.error(
            "ingestion.resolver_error",
            code="RESOLVER_ERROR",
            detail=repr(exc),
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


def _hash_payload(payload: dict[str, Any]) -> str:
    """SHA-256 over canonical JSON. Stable across Python runs.

    ``sort_keys=True`` and explicit separators give us a deterministic
    byte sequence — a payload with the same content but different key
    order must hash the same, otherwise dedup is useless.
    """
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
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
