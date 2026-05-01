"""Entity-resolution worker (BUF-12, Systems-spec System 03).

Three pieces sit on top of the BUF-7 :func:`resolve_entity` chokepoint:

1. :func:`process_staging_queue` — drains
   ``staging_record.status == pending``. For each row:

      a. Lock with ``SELECT ... FOR UPDATE SKIP LOCKED`` so two
         workers can't double-process the same row.
      b. Project ``(platform, platform_id, platform_name,
         entity_type)`` out of the row's payload using a small
         registry of source-specific extractors (currently
         ``liquipedia`` + ``vlr``; new sources register their own).
      c. Call :func:`resolve_entity` to identify or mint the canonical.
      d. Mark the row processed with ``canonical_id`` set.

2. :func:`merge_records` — pure source-priority resolver.
   ``riot_api > liquipedia > vlr > esportsearnings > twitch >
   twitter``. Returns the merged dict + a list of
   :class:`ConflictRecord`. Conflicts are logged at WARNING with both
   ``before`` and ``after`` values so an operator can audit which
   source won which field.

3. :func:`handle_rebrand` — append-only alias extension. Looks up the
   existing canonical via ``(platform, old_platform_id)``, then
   inserts a new alias under the **new** ``(platform_id,
   platform_name)`` on the same canonical with
   ``valid_from=effective_date``. Idempotent: a rebrand replayed with
   the same effective date is a no-op.

Plus the read-side query helper :func:`lookup_alias_at` so callers
that need to know "what was this player's handle as of date D" can
use it without re-implementing the most-recent-valid-from rule.

Why these three pieces sit together: System 03 is the single
authority for cross-source canonical state. Splitting merge
(field-level conflict resolution) from resolve (handle-to-canonical
mapping) keeps each function testable in isolation while letting the
worker compose them per row.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from esports_sim.db.enums import EntityType, Platform, StagingStatus
from esports_sim.db.models import EntityAlias, StagingRecord
from esports_sim.resolver.core import (
    ResolutionStatus,
    resolve_entity,
)

_logger = logging.getLogger("esports_sim.resolver.worker")

# Source priority from the BUF-12 spec. **Lower number wins.**
#
# A connector emits payloads tagged with one of these platforms; when
# two connectors disagree on a field for the same canonical, the lower
# number's value lands in the merged record and the loser is logged as
# a conflict. The order is policy: Riot's API is authoritative because
# it's the game maker's own data; Liquipedia next because their
# editors curate it; VLR after because its scraping shape is
# brittler; esportsearnings/twitch/twitter are softer signals.
SOURCE_PRIORITY: dict[Platform, int] = {
    Platform.RIOT_API: 0,
    Platform.LIQUIPEDIA: 1,
    Platform.VLR: 2,
    Platform.ESPORTSEARNINGS: 3,
    Platform.TWITCH: 4,
    Platform.TWITTER: 5,
}


@dataclass(frozen=True)
class ConflictRecord:
    """One field-level disagreement between two source payloads.

    ``before`` is the existing record's value; ``after`` is what
    actually landed in the merged dict (i.e. the higher-priority
    source's value, which may be the same as ``before`` if the
    incoming source lost). ``winning_source`` carries the name of the
    source whose value is in ``after`` so a log line is self-contained.
    """

    field_name: str
    before: Any
    after: Any
    losing_source: str
    winning_source: str

    def to_log_dict(self) -> dict[str, Any]:
        return {
            "field": self.field_name,
            "before": self.before,
            "after": self.after,
            "losing_source": self.losing_source,
            "winning_source": self.winning_source,
        }


@dataclass(frozen=True)
class MergeResult:
    """Outcome of one :func:`merge_records` call.

    ``merged`` is the post-resolution dict the caller should persist;
    ``conflicts`` is the list of every field where the two inputs
    disagreed. An empty ``conflicts`` list means the inputs agreed on
    every shared key.
    """

    merged: dict[str, Any]
    conflicts: list[ConflictRecord]


@dataclass
class WorkerStats:
    """Counters returned by :func:`process_staging_queue`.

    Mirrors the runner's ``IngestionStats`` shape so an operator can
    diff worker vs. ingest activity over a window. ``elapsed_seconds``
    lets the BUF-12 perf acceptance test assert against a wall-clock
    budget without a separate timer.
    """

    seen: int = 0
    processed: int = 0
    pending: int = 0
    review: int = 0
    blocked: int = 0
    extractor_misses: int = 0
    by_status: dict[str, int] = field(default_factory=dict)
    elapsed_seconds: float = 0.0


# A staging row's payload shape is connector-specific; the worker can't
# know how to project (platform_id, platform_name) without help. Each
# extractor returns ``None`` when its source isn't applicable so the
# worker can fall through to the next registered extractor.
PayloadExtractor = Callable[
    [str, EntityType, dict[str, Any]],
    "ExtractedHandle | None",
]


@dataclass(frozen=True)
class ExtractedHandle:
    """The (platform, platform_id, platform_name) triple a worker needs.

    ``source`` is the staging row's ``source`` string echoed back so
    the worker doesn't have to thread it through; useful for the
    structured log line.
    """

    source: str
    platform: Platform
    platform_id: str
    platform_name: str


# Default mapping from a staging row's free-form ``source`` string to
# the ``Platform`` the resolver should attribute its alias to. Workers
# in tests can pass their own mapping; production registers the same
# strings the connectors set as ``source_name``.
DEFAULT_SOURCE_PLATFORMS: dict[str, Platform] = {
    "liquipedia": Platform.LIQUIPEDIA,
    "vlr": Platform.VLR,
    "riot": Platform.RIOT_API,
    "esportsearnings": Platform.ESPORTSEARNINGS,
    "twitch": Platform.TWITCH,
    "twitter": Platform.TWITTER,
}


def default_payload_extractor(
    source: str,
    entity_type: EntityType,
    payload: dict[str, Any],
) -> ExtractedHandle | None:
    """Best-effort extractor for the connector payload shapes we ship today.

    Looks for ``platform_id`` / ``platform_name`` first (the explicit
    contract a future connector should adopt), then falls back to the
    common Liquipedia/VLR shape of ``slug`` + ``name``. Returns
    ``None`` if the payload doesn't carry either pattern, in which
    case the worker logs an extractor miss and skips the row.
    """
    platform = DEFAULT_SOURCE_PLATFORMS.get(source)
    if platform is None:
        return None

    # Explicit contract: a connector that wants worker compatibility
    # writes ``platform_id`` and ``platform_name`` directly into the
    # payload. The resolver's keys read straight from the payload then.
    explicit_id = payload.get("platform_id")
    explicit_name = payload.get("platform_name")
    if isinstance(explicit_id, str) and isinstance(explicit_name, str):
        return ExtractedHandle(
            source=source,
            platform=platform,
            platform_id=explicit_id,
            platform_name=explicit_name,
        )

    # Fallback: Liquipedia/VLR ship the upstream raw shape. A team or
    # player record carries ``slug`` + ``name``; tournaments the same.
    # ``previous_slug`` wins if present so a rebrand survives a worker
    # pass without forking a canonical.
    slug = payload.get("previous_slug") or payload.get("slug")
    name = payload.get("name") or payload.get("display_name")
    if isinstance(slug, str) and isinstance(name, str):
        return ExtractedHandle(
            source=source,
            platform=platform,
            platform_id=slug,
            platform_name=name,
        )
    return None


# --- merge_records --------------------------------------------------------


def merge_records(
    existing: dict[str, Any],
    incoming: dict[str, Any],
    *,
    existing_source: Platform,
    incoming_source: Platform,
    log_conflicts: bool = True,
) -> MergeResult:
    """Merge two payloads under :data:`SOURCE_PRIORITY`. Pure function.

    Returns the merged dict and a list of every field where the inputs
    disagreed. Fields only present on one side are taken verbatim;
    fields present on both go to whichever source has the lower
    ``SOURCE_PRIORITY`` number.

    A field where the two values are equal is **not** a conflict —
    only divergence counts. The merged dict's value for non-conflict
    fields is whatever both sides agreed on.

    ``log_conflicts=True`` (the default) emits one WARNING per
    divergent field with the before/after pair so an operator can
    audit who won what. Tests pass ``False`` when they want to assert
    against the conflict list directly without a noisy log.
    """
    if existing_source not in SOURCE_PRIORITY:
        raise ValueError(f"existing_source {existing_source!r} not in SOURCE_PRIORITY")
    if incoming_source not in SOURCE_PRIORITY:
        raise ValueError(f"incoming_source {incoming_source!r} not in SOURCE_PRIORITY")

    existing_priority = SOURCE_PRIORITY[existing_source]
    incoming_priority = SOURCE_PRIORITY[incoming_source]
    incoming_wins = incoming_priority < existing_priority

    merged: dict[str, Any] = dict(existing)
    conflicts: list[ConflictRecord] = []

    for key, incoming_value in incoming.items():
        if key not in merged:
            # New field — no disagreement to log.
            merged[key] = incoming_value
            continue
        existing_value = merged[key]
        if existing_value == incoming_value:
            continue

        # Divergent — apply the winner.
        if incoming_wins:
            merged[key] = incoming_value
            conflicts.append(
                ConflictRecord(
                    field_name=key,
                    before=existing_value,
                    after=incoming_value,
                    losing_source=existing_source.value,
                    winning_source=incoming_source.value,
                )
            )
        else:
            conflicts.append(
                ConflictRecord(
                    field_name=key,
                    before=existing_value,
                    after=existing_value,
                    losing_source=incoming_source.value,
                    winning_source=existing_source.value,
                )
            )

    if log_conflicts:
        for c in conflicts:
            _logger.warning(
                "resolver.merge_conflict field=%s winning_source=%s losing_source=%s "
                "before=%r after=%r",
                c.field_name,
                c.winning_source,
                c.losing_source,
                c.before,
                c.after,
            )

    return MergeResult(merged=merged, conflicts=conflicts)


# --- handle_rebrand -------------------------------------------------------


def handle_rebrand(
    session: Session,
    *,
    platform: Platform,
    old_platform_id: str,
    new_platform_id: str,
    new_platform_name: str,
    effective_date: datetime,
    confidence: float = 1.0,
) -> EntityAlias:
    """Extend an existing canonical's alias chain with a new handle.

    The canonical_id is determined by looking up the existing alias
    keyed by ``(platform, old_platform_id)``. A rebrand under a slug
    that doesn't yet exist is a programming error (no canonical to
    extend) and raises :class:`ValueError`.

    Idempotent: if ``(platform, new_platform_id)`` already exists and
    points at the same canonical, the existing alias is returned
    unchanged. If it exists but points at a *different* canonical,
    that's a real conflict (two canonicals both think they're the
    rebrand target) and we surface it as :class:`RebrandConflictError`
    rather than silently extending the wrong chain.

    The new alias's ``valid_from`` is ``effective_date``. Querying
    "what was this entity's handle as of date D" then becomes
    "the alias under canonical_id whose ``valid_from`` is the most
    recent value <= D" — see :func:`lookup_alias_at`.
    """
    if not old_platform_id:
        raise ValueError("old_platform_id must be non-empty")
    if not new_platform_id:
        raise ValueError("new_platform_id must be non-empty")
    if not new_platform_name:
        raise ValueError("new_platform_name must be non-empty")
    if effective_date.tzinfo is None:
        raise ValueError("effective_date must be timezone-aware")

    existing_old = session.execute(
        select(EntityAlias).where(
            EntityAlias.platform == platform,
            EntityAlias.platform_id == old_platform_id,
        )
    ).scalar_one_or_none()
    if existing_old is None:
        raise ValueError(
            f"handle_rebrand: no existing alias for ({platform.value}, {old_platform_id!r})"
        )

    canonical_id = existing_old.canonical_id

    existing_new = session.execute(
        select(EntityAlias).where(
            EntityAlias.platform == platform,
            EntityAlias.platform_id == new_platform_id,
        )
    ).scalar_one_or_none()
    if existing_new is not None:
        if existing_new.canonical_id != canonical_id:
            raise RebrandConflictError(
                f"({platform.value}, {new_platform_id!r}) already maps to "
                f"canonical {existing_new.canonical_id} but rebrand expects "
                f"{canonical_id}"
            )
        # Idempotent re-run of the same rebrand. Return the existing
        # alias so the caller's bookkeeping stays consistent.
        return existing_new

    new_alias = EntityAlias(
        canonical_id=canonical_id,
        platform=platform,
        platform_id=new_platform_id,
        platform_name=new_platform_name,
        confidence=confidence,
        valid_from=effective_date,
    )
    try:
        session.add(new_alias)
        session.flush()
    except IntegrityError as exc:
        # The unique key is on (platform, platform_id) — the only race
        # we know how to recover from. Re-fetch and verify the winner
        # is the same canonical; otherwise re-raise so the caller sees
        # the conflict.
        if "uq_entity_alias_platform_platform_id" not in str(exc):
            raise
        session.rollback()
        winner = session.execute(
            select(EntityAlias).where(
                EntityAlias.platform == platform,
                EntityAlias.platform_id == new_platform_id,
            )
        ).scalar_one()
        if winner.canonical_id != canonical_id:
            raise RebrandConflictError(
                f"race winner under ({platform.value}, {new_platform_id!r}) is "
                f"canonical {winner.canonical_id}; rebrand expected {canonical_id}"
            ) from exc
        return winner

    _logger.info(
        "resolver.handle_rebrand platform=%s old=%s new=%s canonical_id=%s effective=%s",
        platform.value,
        old_platform_id,
        new_platform_id,
        canonical_id,
        effective_date.isoformat(),
    )
    return new_alias


class RebrandConflictError(RuntimeError):
    """Raised when a rebrand would attach a new alias to the wrong canonical.

    Two distinct canonical entities both claiming the same destination
    handle is a real data problem — usually two seed runs forking on a
    near-miss, or an upstream source advertising a rebrand that
    overlaps a different player's existing handle. Surfaces here
    rather than being silently absorbed so an operator can decide.
    """


# --- lookup_alias_at -----------------------------------------------------


def lookup_alias_at(
    session: Session,
    *,
    platform: Platform,
    platform_id: str,
    at: datetime,
) -> EntityAlias | None:
    """Return the alias for ``(platform, platform_id)`` valid at ``at``.

    "Valid at ``at``" means: the alias whose ``valid_from`` is the
    largest value not greater than ``at``. Returns ``None`` when no
    alias under that key has a ``valid_from`` <= ``at`` (i.e. the
    handle didn't exist yet at that point).

    Today the schema doesn't carry a ``valid_to`` column, so this
    helper assumes "the current alias is valid until superseded by a
    later one with a more recent ``valid_from``". When BUF-?? adds
    explicit ``valid_to`` we'll tighten this query to honour it.
    """
    if at.tzinfo is None:
        raise ValueError("at must be timezone-aware")
    stmt = (
        select(EntityAlias)
        .where(
            EntityAlias.platform == platform,
            EntityAlias.platform_id == platform_id,
            EntityAlias.valid_from <= at,
        )
        .order_by(EntityAlias.valid_from.desc())
        .limit(1)
    )
    return session.execute(stmt).scalar_one_or_none()


# --- process_staging_queue ------------------------------------------------


def process_staging_queue(
    session: Session,
    *,
    batch_size: int = 100,
    payload_extractor: PayloadExtractor | None = None,
    max_batches: int | None = None,
) -> WorkerStats:
    """Drain ``staging_record.status == pending``. Single-shot, returns stats.

    The worker does **not** run forever; one call processes up to
    ``max_batches * batch_size`` rows and returns. The caller (a
    scheduler, a CLI, a test) decides whether to loop. This shape
    keeps the worker testable: a unit test can drive one batch and
    assert against the resulting staging-row state without a separate
    "stop the worker" affordance.

    Each row is locked with ``SELECT ... FOR UPDATE SKIP LOCKED`` so a
    second worker on the same DB picks up untouched rows in parallel.
    The lock is held for the duration of the resolver call; the
    enclosing transaction is the caller's, the worker only ``flush``es.

    A row whose payload doesn't carry an extractable
    ``(platform, platform_id, platform_name)`` triple is logged as
    ``extractor_miss`` and **stays in the queue** for re-processing
    after a fix to the extractor; we don't want a payload-shape bug
    to silently drain the queue.
    """
    extractor = payload_extractor or default_payload_extractor
    stats = WorkerStats()
    started = time.monotonic()

    batches_done = 0
    while True:
        if max_batches is not None and batches_done >= max_batches:
            break

        rows = (
            session.execute(
                select(StagingRecord)
                .where(StagingRecord.status == StagingStatus.PENDING)
                .order_by(StagingRecord.created_at)
                .limit(batch_size)
                .with_for_update(skip_locked=True)
            )
            .scalars()
            .all()
        )
        if not rows:
            break
        batches_done += 1

        for row in rows:
            stats.seen += 1
            handle = extractor(row.source, row.entity_type, row.payload)
            if handle is None:
                stats.extractor_misses += 1
                _logger.warning(
                    "resolver.worker.extractor_miss source=%s entity_type=%s id=%s",
                    row.source,
                    row.entity_type.value,
                    row.id,
                )
                continue

            result = resolve_entity(
                session,
                platform=handle.platform,
                platform_id=handle.platform_id,
                platform_name=handle.platform_name,
                entity_type=row.entity_type,
            )
            stats.by_status[result.status.value] = stats.by_status.get(result.status.value, 0) + 1

            if result.status is ResolutionStatus.PENDING:
                # Resolver couldn't auto-decide; staging row moves to
                # REVIEW so a human reviewer (BUF-16) takes over. The
                # canonical_id stays null — that's the documented
                # ``review`` lifecycle.
                row.status = StagingStatus.REVIEW
                stats.review += 1
            else:
                row.canonical_id = result.canonical_id
                row.status = StagingStatus.PROCESSED
                stats.processed += 1
            session.flush()

    stats.elapsed_seconds = time.monotonic() - started
    _logger.info(
        "resolver.worker.done seen=%d processed=%d review=%d misses=%d elapsed_s=%.3f",
        stats.seen,
        stats.processed,
        stats.review,
        stats.extractor_misses,
        stats.elapsed_seconds,
    )
    return stats


# --- helpers (re-exported for callers that need to monkey-patch) ----------


def _utcnow() -> datetime:
    return datetime.now(UTC)


__all__ = [
    "ConflictRecord",
    "DEFAULT_SOURCE_PLATFORMS",
    "ExtractedHandle",
    "MergeResult",
    "PayloadExtractor",
    "RebrandConflictError",
    "SOURCE_PRIORITY",
    "WorkerStats",
    "default_payload_extractor",
    "handle_rebrand",
    "lookup_alias_at",
    "merge_records",
    "process_staging_queue",
]


# Silence the unused-import lint for a uuid type hint we keep available
# for downstream callers writing extractors that need to read
# ``staging_record.id`` as a uuid.UUID.
_ = uuid
