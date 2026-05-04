"""Batch worker: pull untranscribed ``MediaRecord`` rows, write ``Transcript`` (BUF-21).

The worker is the production glue between the scheduler, the
:class:`~data_pipeline.transcribe.engine.TranscriptionEngine`, and
the :class:`~esports_sim.db.models.Transcript` table:

* Selects ``MediaRecord`` rows that don't have a child
  ``Transcript`` (the BUF-21 spec's "where transcript IS NULL"). A
  caller can pass ``include_existing=True`` to force a re-
  transcription pass (e.g., after rotating to a newer model) — the
  UPSERT path replaces the existing row in place.
* Calls the engine for each, writes the ``Transcript`` row, and
  optionally drops a ``transcript.json`` sidecar so a downstream
  step (the BUF-28 chunk embedder, manual review) doesn't have to
  go through the database.
* Per-record errors are logged and skipped — a single corrupt audio
  file shouldn't take down a 10h batch.

Idempotency: a re-run on the same ``media_id`` deletes the prior
:class:`~esports_sim.db.models.Transcript` row and inserts a fresh
one. We don't UPSERT-in-place because ``segments`` is JSONB and the
shape can shift across model versions; a clean replace keeps the
rebuild semantics obvious.

The caller owns the transaction. The worker ``flush``es per row so
``transcribe_pending`` can be wrapped in either ``session.begin()``
(all-or-nothing) or per-row commits (the scheduler's default — a
crash mid-batch keeps the rows that already finished).
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import structlog
from esports_sim.db.models import MediaRecord, Transcript
from sqlalchemy import select
from sqlalchemy.orm import Session
from structlog.stdlib import BoundLogger

from data_pipeline.transcribe.engine import TranscriptionEngine, TranscriptionResult


@dataclass
class TranscribePendingStats:
    """Counters returned from :func:`transcribe_pending` for observability.

    Per-record outcomes increment exactly one counter so the totals
    add up. ``audio_seconds`` and ``wallclock_seconds`` are the
    aggregate numerator + denominator for the BUF-21 throughput
    acceptance ("≥20× realtime"); a successful 10h batch should land
    ``audio_seconds / wallclock_seconds >= 20``.
    """

    selected: int = 0
    transcribed: int = 0
    skipped_existing: int = 0
    file_missing: int = 0
    engine_errors: int = 0
    audio_seconds: float = 0.0
    wallclock_seconds: float = 0.0
    by_language: dict[str, int] = field(default_factory=dict)


def transcribe_pending(
    *,
    session: Session,
    engine: TranscriptionEngine,
    sidecar_root: Path | None = None,
    limit: int | None = None,
    include_existing: bool = False,
    logger: BoundLogger | None = None,
) -> TranscribePendingStats:
    """Drive one transcription pass against the queue of pending media.

    Parameters
    ----------
    session
        Open SQLAlchemy session. The caller owns the transaction;
        the worker ``flush``es per row but does not ``commit``.
    engine
        The :class:`TranscriptionEngine` to drive. Tests inject a
        deterministic stub; production wires
        :class:`~data_pipeline.transcribe.engine.FasterWhisperEngine`.
    sidecar_root
        Directory under which to write per-media ``transcript.json``
        sidecars. The full path is
        ``{sidecar_root}/{media_id}/transcript.json``. When ``None``,
        no sidecar is written and ``Transcript.transcript_path`` is
        left null. The directory is created on demand so the caller
        doesn't have to pre-stage it.
    limit
        Cap on the number of media rows to process this pass. ``None``
        means "all pending" — fine for a one-shot, but the scheduler
        passes a bounded number so a crash mid-batch doesn't lose
        more than a manageable chunk of work.
    include_existing
        When ``True``, also re-transcribe rows that already have a
        :class:`Transcript` child. Used after rotating to a newer
        model. Default is ``False`` so the steady-state pass is a
        no-op once every media has been transcribed once.
    logger
        Optional pre-bound structlog logger. The worker binds
        ``component="transcribe"`` plus per-row context.
    """
    base_logger = logger or structlog.get_logger("data_pipeline.transcribe")
    bound = base_logger.bind(component="transcribe", model_version=engine.model_version)

    stats = TranscribePendingStats()

    pending = _select_pending(session, limit=limit, include_existing=include_existing)
    bound.info(
        "transcribe.start",
        candidates=len(pending),
        include_existing=include_existing,
        limit=limit,
    )

    for media in pending:
        stats.selected += 1
        log = bound.bind(media_id=str(media.id), source=media.source)

        audio_path = Path(media.local_path)
        if not audio_path.exists():
            # Operational reality: some files referenced by media_record
            # may have been pruned from the worker host (cache eviction,
            # an aborted download). Log and skip — the row stays pending,
            # and a future pass after the file is restored picks it up.
            stats.file_missing += 1
            log.warning(
                "transcribe.file_missing",
                code="FILE_MISSING",
                local_path=str(audio_path),
            )
            continue

        try:
            result = engine.transcribe(audio_path, language=media.language)
        except FileNotFoundError:
            # Engine raised after our pre-check — race with file eviction.
            # Counted under file_missing so the operator sees a single
            # "file went away" bucket regardless of which layer noticed.
            stats.file_missing += 1
            log.warning(
                "transcribe.file_missing",
                code="FILE_MISSING",
                local_path=str(audio_path),
            )
            continue
        except Exception as exc:
            # Per-record skip. The engine layer wraps faster-whisper's
            # opaque CTranslate2 errors as plain ``Exception``; we log
            # under a structured code so the postmortem trail has
            # ``media_id`` + ``source`` and the run continues.
            stats.engine_errors += 1
            log.exception(
                "transcribe.engine_error",
                code="ENGINE_ERROR",
                detail=str(exc),
            )
            continue

        sidecar_path = _write_sidecar_if_configured(
            media_id=media.id,
            result=result,
            sidecar_root=sidecar_root,
        )

        _replace_transcript(
            session,
            media_id=media.id,
            result=result,
            sidecar_path=sidecar_path,
        )
        session.flush()

        stats.transcribed += 1
        stats.audio_seconds += result.duration_seconds
        stats.wallclock_seconds += result.wallclock_seconds
        stats.by_language[result.language] = stats.by_language.get(result.language, 0) + 1
        log.info(
            "transcribe.processed",
            language=result.language,
            audio_seconds=result.duration_seconds,
            wallclock_seconds=result.wallclock_seconds,
            sidecar=str(sidecar_path) if sidecar_path is not None else None,
        )

    bound.info(
        "transcribe.done",
        selected=stats.selected,
        transcribed=stats.transcribed,
        file_missing=stats.file_missing,
        engine_errors=stats.engine_errors,
        audio_seconds=stats.audio_seconds,
        wallclock_seconds=stats.wallclock_seconds,
        realtime_factor=_safe_ratio(stats.audio_seconds, stats.wallclock_seconds),
    )
    return stats


# --- helpers ---------------------------------------------------------------


def _select_pending(
    session: Session,
    *,
    limit: int | None,
    include_existing: bool,
) -> list[MediaRecord]:
    """Return media rows the worker should attempt this pass.

    Default: rows that don't have a child ``Transcript`` yet — the
    BUF-21 spec's "where transcript IS NULL" projected onto the
    transcript-as-separate-table schema.

    With ``include_existing=True``, return every media row regardless
    of transcript state, ordered by created-time. Used by the
    "re-transcribe everything with the new model" workflow that
    follows a model rotation.

    Concurrency: the SELECT takes a row-level ``FOR UPDATE SKIP
    LOCKED`` on ``media_record`` so two overlapping worker runs
    can't claim the same media. Without it, both workers would race
    to ``INSERT INTO transcript`` for the same ``media_id`` and the
    second one would crash on the PK constraint, rolling back its
    whole transaction. ``SKIP LOCKED`` is the right primitive here
    rather than ``NOWAIT``: a second worker should silently skip
    over claimed rows and pick the next unlocked batch, not fail
    with "could not obtain lock". The lock is held until the
    enclosing transaction commits, which the worker contract
    requires the caller to provide (the CLI wraps the pass in
    ``session.begin()``).

    ``of=`` pins the lock target to ``MediaRecord`` only — without
    it Postgres locks every row produced by the join, which on the
    LEFT JOIN side would mean trying to lock the (NULL) Transcript
    row and fail with "FOR UPDATE cannot be applied to the nullable
    side of an outer join".
    """
    base = select(MediaRecord)
    if include_existing:
        stmt = base.order_by(MediaRecord.created_at)
    else:
        # LEFT JOIN + WHERE transcript.media_id IS NULL is the
        # idiomatic "rows in A without a row in B" SQL pattern. Index
        # support: ``transcript.media_id`` is the PK, so the join is
        # an index-only lookup on the right side.
        stmt = (
            base.outerjoin(Transcript, Transcript.media_id == MediaRecord.id)
            .where(Transcript.media_id.is_(None))
            .order_by(MediaRecord.created_at)
        )
    stmt = stmt.with_for_update(skip_locked=True, of=MediaRecord)
    if limit is not None:
        stmt = stmt.limit(limit)
    return list(session.execute(stmt).scalars().all())


def _replace_transcript(
    session: Session,
    *,
    media_id: uuid.UUID,
    result: TranscriptionResult,
    sidecar_path: Path | None,
) -> Transcript:
    """Drop any existing ``Transcript`` for ``media_id`` and insert a fresh one.

    ORM-level ``session.delete`` + flush rather than a bulk SQL
    ``delete()`` so the identity map stays consistent: a re-run that
    targets the same media within one open session would otherwise
    leave a stale Transcript instance mapped to the same PK and the
    follow-up ``add`` would trip a primary-key collision at flush.

    DELETE-then-INSERT rather than UPSERT-in-place because
    ``segments`` is JSONB and the per-row shape can shift across
    model versions; a clean replace keeps the rebuild semantics
    obvious. The DELETE + INSERT live inside the caller's
    transaction, so a partial failure rolls back together.
    """
    existing = session.get(Transcript, media_id)
    if existing is not None:
        session.delete(existing)
        # Flush the DELETE before the INSERT so the same-PK collision
        # raised by Postgres becomes a Python-side ordering decision
        # rather than depending on SQLAlchemy's per-flush statement
        # ordering.
        session.flush()

    transcript = Transcript(
        media_id=media_id,
        language=result.language,
        model_version=result.model_version,
        text=result.text,
        segments=[seg.to_jsonable() for seg in result.segments],
        duration_seconds=result.duration_seconds,
        wallclock_seconds=result.wallclock_seconds,
        transcript_path=str(sidecar_path) if sidecar_path is not None else None,
    )
    session.add(transcript)
    return transcript


def _write_sidecar_if_configured(
    *,
    media_id: uuid.UUID,
    result: TranscriptionResult,
    sidecar_root: Path | None,
) -> Path | None:
    """Drop a ``transcript.json`` next to the media if a root is configured.

    Returns the path written, or ``None`` if no root was given. Callers
    that want a sidecar pass ``sidecar_root``; callers running purely
    against the database (the test suite, a one-off CLI invocation
    that just needs a row written) leave it ``None`` and skip the
    filesystem entirely.

    The sidecar shape is::

        {
            "media_id": "<uuid>",
            "language": "en",
            "model_version": "large-v3",
            "duration_seconds": 1234.5,
            "wallclock_seconds": 60.0,
            "text": "...",
            "segments": [{"start": 0.0, "end": 4.2, "text": "...", "speaker": null}, ...]
        }
    """
    if sidecar_root is None:
        return None
    media_dir = sidecar_root / str(media_id)
    media_dir.mkdir(parents=True, exist_ok=True)
    sidecar_path = media_dir / "transcript.json"
    payload = {
        "media_id": str(media_id),
        "language": result.language,
        "model_version": result.model_version,
        "duration_seconds": result.duration_seconds,
        "wallclock_seconds": result.wallclock_seconds,
        "text": result.text,
        "segments": [seg.to_jsonable() for seg in result.segments],
    }
    # Atomic-ish write: dump to tmp + rename so an interrupted
    # write doesn't leave a half-truncated JSON file the next
    # consumer would choke on.
    tmp_path = sidecar_path.with_suffix(".json.tmp")
    tmp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(sidecar_path)
    return sidecar_path


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    """Avoid division-by-zero when no rows were transcribed.

    Returns ``None`` if ``denominator`` is zero; the structured log
    consumer ignores nulls rather than treating a missing throughput
    number as a literal zero (which would page on a quiet pass).
    """
    if denominator <= 0:
        return None
    return numerator / denominator


__all__ = [
    "TranscribePendingStats",
    "transcribe_pending",
]
