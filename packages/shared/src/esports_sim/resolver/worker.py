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
    # Rebrand events the worker handled by extending the alias chain
    # with the new slug. ``rebrand_conflicts`` is the count of payloads
    # whose destination handle was already owned by a different
    # canonical — those rows still resolve to PROCESSED on the OLD
    # slug, but the rebrand alias extension is logged + skipped rather
    # than silently corrupted into the wrong canonical.
    rebrands_registered: int = 0
    rebrand_conflicts: int = 0
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
class RebrandTarget:
    """The new-handle half of a rebrand event detected during extraction.

    When an extractor surfaces this on an :class:`ExtractedHandle`, the
    worker resolves on the OLD handle (``platform_id`` of the parent
    ``ExtractedHandle``) and then calls :func:`handle_rebrand` to also
    register the new handle on the same canonical. Without this, a
    rebrand payload would resolve as MATCHED on the old slug and a
    later record carrying ONLY the new slug would fork into a
    duplicate canonical — same bug the seed had before round 2 of the
    PR review fixed it.

    ``effective_date`` lands as ``EntityAlias.valid_from`` on the new
    alias so :func:`lookup_alias_at` can answer "what was this handle
    pointing at on date D" correctly.
    """

    new_platform_id: str
    new_platform_name: str
    effective_date: datetime


@dataclass(frozen=True)
class ExtractedHandle:
    """The (platform, platform_id, platform_name) triple a worker needs.

    ``source`` is the staging row's ``source`` string echoed back so
    the worker doesn't have to thread it through; useful for the
    structured log line.

    ``rebrand_target`` is set when the payload describes a rebrand
    event. The worker resolves on ``platform_id`` (the OLD handle),
    then calls :func:`handle_rebrand` with the rebrand target's new
    handle on the same canonical.
    """

    source: str
    platform: Platform
    platform_id: str
    platform_name: str
    rebrand_target: RebrandTarget | None = None


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

    Rebrand detection: when a payload carries both ``slug`` and
    ``previous_slug`` (different values), an :class:`ExtractedHandle`
    is returned with ``platform_id=previous_slug`` (so the resolver
    matches the existing canonical) and a populated
    ``rebrand_target`` so the worker fires :func:`handle_rebrand`
    afterward to also register the new slug. Without this, a rebrand
    record would silently leave the new slug unattached and a later
    record carrying only the new slug would fork into a duplicate
    canonical — exactly the seed-side bug round 2 of the review
    caught.
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
    slug = payload.get("slug")
    previous_slug = payload.get("previous_slug")
    name = payload.get("name") or payload.get("display_name")
    if not isinstance(name, str):
        return None

    # Rebrand event: resolve on the OLD slug so the existing canonical
    # matches, then surface the new slug as a ``rebrand_target`` for
    # the worker to register via ``handle_rebrand``.
    if (
        isinstance(previous_slug, str)
        and previous_slug
        and isinstance(slug, str)
        and slug
        and previous_slug != slug
    ):
        return ExtractedHandle(
            source=source,
            platform=platform,
            platform_id=previous_slug,
            platform_name=name,
            rebrand_target=RebrandTarget(
                new_platform_id=slug,
                new_platform_name=name,
                effective_date=parse_renamed_at(payload.get("renamed_at")),
            ),
        )

    # Plain (no rebrand): use whichever slug is present.
    handle_id = previous_slug or slug
    if isinstance(handle_id, str) and handle_id:
        return ExtractedHandle(
            source=source,
            platform=platform,
            platform_id=handle_id,
            platform_name=name,
        )
    return None


def parse_renamed_at(value: Any) -> datetime:
    """Best-effort parse of a payload's ``renamed_at`` into a tz-aware datetime.

    Liquipedia ships dates as ISO ``YYYY-MM-DD`` (no time, no tz).
    A bare date is upgraded to UTC midnight; a full ISO datetime with
    no tz is also upgraded to UTC. Anything unparseable degrades to
    ``datetime.now(UTC)`` so the rebrand still records — losing the
    correct timestamp is better than dropping the alias.

    Lives in the worker module because both the seed (BUF-8) and the
    extractor (BUF-12) need to project a Liquipedia rebrand event's
    effective date the same way; centralising it keeps the two paths
    in lockstep.
    """
    if isinstance(value, str) and value:
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return datetime.now(UTC)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=UTC)
        return parsed
    return datetime.now(UTC)


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

    # Wrap the insert in a savepoint so the (platform, platform_id) race
    # recovers cleanly without trashing the caller's outer transaction.
    # Round 1 used ``session.rollback()`` here, which rolls back EVERY
    # uncommitted write in the session — fine in isolation, catastrophic
    # when ``handle_rebrand`` is one step inside a multi-step batch
    # (e.g. a worker pass that's already handled ten other rows). The
    # savepoint scopes the rollback to just the failed alias insert,
    # mirroring the pattern in :func:`resolve_entity`'s alias-race
    # recoveries.
    new_alias = EntityAlias(
        canonical_id=canonical_id,
        platform=platform,
        platform_id=new_platform_id,
        platform_name=new_platform_name,
        confidence=confidence,
        valid_from=effective_date,
    )
    try:
        with session.begin_nested():
            session.add(new_alias)
            session.flush()
    except IntegrityError as exc:
        # The unique key is on (platform, platform_id) — the only race
        # we know how to recover from. Re-fetch and verify the winner
        # is the same canonical; otherwise re-raise so the caller sees
        # the conflict.
        if "uq_entity_alias_platform_platform_id" not in str(exc):
            raise
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

    # Extractor-miss rows stay PENDING (so a fix to the extractor can
    # pick them up on the next worker pass) but MUST be excluded from
    # subsequent batch SELECTs in the *current* run — otherwise the
    # same rows come back every iteration and the loop never reaches
    # ``if not rows: break``. The set is bounded by the queue size in
    # one run, so it's cheap. UUIDs round-trip through psycopg as
    # native uuid.UUID objects, no string coercion needed.
    skipped_in_run: set[uuid.UUID] = set()

    batches_done = 0
    while True:
        if max_batches is not None and batches_done >= max_batches:
            break

        stmt = (
            select(StagingRecord)
            .where(StagingRecord.status == StagingStatus.PENDING)
            .order_by(StagingRecord.created_at)
            .limit(batch_size)
            .with_for_update(skip_locked=True)
        )
        if skipped_in_run:
            stmt = stmt.where(StagingRecord.id.notin_(skipped_in_run))
        rows = session.execute(stmt).scalars().all()
        if not rows:
            break
        batches_done += 1

        for row in rows:
            stats.seen += 1
            handle = extractor(row.source, row.entity_type, row.payload)
            if handle is None:
                stats.extractor_misses += 1
                skipped_in_run.add(row.id)
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
                # Rebrand extension: when the extractor surfaces a
                # rebrand_target the worker also registers the new
                # slug as an alias on the same canonical, mirroring
                # what the BUF-8 seed does on its side. Without this,
                # a later record carrying ONLY the new slug would
                # fork into a duplicate canonical.
                if handle.rebrand_target is not None and result.canonical_id is not None:
                    _apply_rebrand_target(
                        session,
                        handle=handle,
                        rebrand=handle.rebrand_target,
                        stats=stats,
                    )
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


def _apply_rebrand_target(
    session: Session,
    *,
    handle: ExtractedHandle,
    rebrand: RebrandTarget,
    stats: WorkerStats,
) -> None:
    """Extend the canonical's alias chain with the rebrand's new slug.

    Called by ``process_staging_queue`` after a successful
    ``resolve_entity`` on the OLD slug. Wraps :func:`handle_rebrand`
    with worker-level bookkeeping: a :class:`RebrandConflictError`
    (the destination handle is owned by a different canonical) is
    logged at WARNING and counted under ``rebrand_conflicts`` rather
    than aborting the worker pass — the row's primary resolution still
    succeeded, only the rebrand alias extension didn't, and an
    operator can decide what to do with the conflict.
    """
    try:
        handle_rebrand(
            session,
            platform=handle.platform,
            old_platform_id=handle.platform_id,
            new_platform_id=rebrand.new_platform_id,
            new_platform_name=rebrand.new_platform_name,
            effective_date=rebrand.effective_date,
        )
    except RebrandConflictError as exc:
        stats.rebrand_conflicts += 1
        _logger.warning(
            "resolver.worker.rebrand_conflict source=%s old=%s new=%s detail=%s",
            handle.source,
            handle.platform_id,
            rebrand.new_platform_id,
            exc,
        )
        return
    stats.rebrands_registered += 1


def _utcnow() -> datetime:
    return datetime.now(UTC)


__all__ = [
    "ConflictRecord",
    "DEFAULT_SOURCE_PLATFORMS",
    "ExtractedHandle",
    "MergeResult",
    "PayloadExtractor",
    "RebrandConflictError",
    "RebrandTarget",
    "SOURCE_PRIORITY",
    "WorkerStats",
    "default_payload_extractor",
    "handle_rebrand",
    "lookup_alias_at",
    "merge_records",
    "parse_renamed_at",
    "process_staging_queue",
]
