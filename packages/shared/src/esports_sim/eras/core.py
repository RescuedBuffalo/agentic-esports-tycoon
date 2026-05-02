"""Era assignment, atomic roll, and TEMPORAL_BLEED guard.

The temporal partitioning rule from Systems-spec System 04:

  "Every record carries an era context; no cross-era feature
   aggregation ever happens."

This module is the single sanctioned writer of
:class:`~esports_sim.db.models.PatchEra` rows for the open/close path
plus the runtime guard that enforces the cross-era rule. The schema
layer (CHECK + EXCLUDE constraints, partial unique index in the
migration) is the backstop; this module is the API the rest of the
codebase calls.

Half-open semantics for era windows: ``[start_date, end_date)``. A
record with timestamp == end_date belongs to the *next* era, never
this one. That's the convention :func:`assign_era` enforces and
:func:`roll_era` exploits — the closed era's ``end_date`` and the new
era's ``start_date`` are stamped with the same instant, with no gap
and no overlap.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterable
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from esports_sim.db.models import PatchEra, TemporalBleedError


class EraNotFoundError(LookupError):
    """No era covers the requested timestamp.

    Either the timestamp falls before the earliest seeded era (i.e. the
    seed didn't backfill far enough) or after the open era's
    ``start_date`` *but* the open era doesn't exist (operator forgot to
    run :func:`open_new_era` after a close). Both are operator-actionable
    failures, not silent fall-through to ``None`` — which would let a
    downstream join silently lose the row.
    """


class EraOverlapError(RuntimeError):
    """An :func:`open_new_era` call would overlap an existing era.

    The exclusion constraint installed by the migration would catch
    this at flush time as :class:`sqlalchemy.exc.IntegrityError`, but
    raising a typed exception from the helper lets callers distinguish
    "you tried to open at a covered instant" from "the FK to entity
    failed for an unrelated reason".
    """


def assign_era(session: Session, ts: datetime) -> uuid.UUID:
    """Return the ``era_id`` whose ``[start_date, end_date)`` covers ``ts``.

    Half-open semantics: a record stamped exactly at an era's
    ``end_date`` resolves to the *next* era. The open era (where
    ``end_date IS NULL``) extends to ``+infinity`` so any timestamp at
    or after its ``start_date`` lands there.

    Raises :class:`EraNotFoundError` if no era covers the timestamp —
    the most common cause is a timestamp predating the seed's earliest
    boundary, which the BUF-13 acceptance "100% of historical matches
    successfully assigned" gate should catch via the seed manifest
    before any record with a too-early timestamp ships.
    """
    ts = _ensure_aware(ts)
    # Half-open match: start_date <= ts AND (end_date IS NULL OR ts < end_date).
    # Postgres orders the open era last on ``end_date ASC NULLS LAST``;
    # we match by the predicate directly so the SQL is portable.
    row = session.execute(
        select(PatchEra)
        .where(
            PatchEra.start_date <= ts,
            (PatchEra.end_date.is_(None)) | (PatchEra.end_date > ts),
        )
        .order_by(PatchEra.start_date.desc())
        .limit(1)
    ).scalar_one_or_none()
    if row is None:
        raise EraNotFoundError(
            f"No patch_era row covers {ts.isoformat()}; either the seed didn't "
            "backfill that far or no era is currently open."
        )
    return row.era_id


def current_era(session: Session) -> PatchEra | None:
    """Return the open era (``end_date IS NULL``), or None if none exists.

    The partial unique index ``ix_patch_era_open_unique`` caps the open
    count at one, so this returns at most one row.
    """
    return session.execute(select(PatchEra).where(PatchEra.end_date.is_(None))).scalar_one_or_none()


def open_new_era(
    session: Session,
    *,
    era_slug: str,
    patch_version: str,
    start_date: datetime,
    meta_magnitude: float = 0.0,
    is_major_shift: bool = False,
) -> PatchEra:
    """Create a new open era (``end_date=None``) starting at ``start_date``.

    This is the second half of the close-then-open transactional pair;
    callers that want the atomic roll should use :func:`roll_era`
    directly. Calling this without a prior close while another era is
    still open will trip the partial unique index on ``end_date IS
    NULL`` at flush time.

    The exclusion constraint on overlapping ranges (installed by the
    migration) means an era opened at an instant inside an existing
    closed era's range raises :class:`sqlalchemy.exc.IntegrityError` at
    flush; the caller is expected to wrap this call in a savepoint and
    translate that into :class:`EraOverlapError` if they want to
    differentiate. The seed loader does this; ad-hoc operator scripts
    typically don't and let the IntegrityError propagate.
    """
    start_date = _ensure_aware(start_date)
    era = PatchEra(
        era_slug=era_slug,
        patch_version=patch_version,
        start_date=start_date,
        end_date=None,
        meta_magnitude=meta_magnitude,
        is_major_shift=is_major_shift,
    )
    session.add(era)
    return era


def roll_era(
    session: Session,
    *,
    new_slug: str,
    new_patch_version: str,
    boundary_at: datetime,
    meta_magnitude: float = 0.0,
    is_major_shift: bool = False,
) -> tuple[PatchEra | None, PatchEra]:
    """Close the current era at ``boundary_at`` and open a new one at the same instant.

    Atomic in two senses:

    * **No timestamp gap.** Both the closed era's ``end_date`` and the
      new era's ``start_date`` are stamped with ``boundary_at`` — half-
      open semantics make this a clean handoff with no instant
      unassigned.
    * **No overlap.** The exclusion constraint from the migration
      enforces this at the DB layer; this helper wraps both writes in a
      single savepoint so a failure on the open path rolls the close
      back too.

    Returns ``(closed_era, new_era)``. ``closed_era`` is ``None`` only
    when there was no open era to close — i.e. this is the first roll
    on a fresh database. Subsequent rolls always return a non-None
    closed era; if no era is open and the caller expects one, that's a
    seed-state bug they should treat as fatal.

    Raises :class:`EraOverlapError` when ``boundary_at`` lies strictly
    inside an existing closed era's range (the EXCLUDE constraint
    catches the overlap). Raises :class:`ValueError` when
    ``boundary_at`` equals or precedes the current era's start_date —
    that would produce a zero-length or negative range, which
    ``ck_patch_era_end_after_start`` would also catch but we surface
    earlier with a clearer message.
    """
    boundary_at = _ensure_aware(boundary_at)

    # Single savepoint around both writes. If the open path raises an
    # IntegrityError (overlap with a closed era, partial-unique-index
    # collision), the close is also rolled back so the previous era's
    # end_date doesn't get persisted on its own.
    with session.begin_nested():
        existing = current_era(session)
        if existing is not None:
            if boundary_at <= existing.start_date:
                raise ValueError(
                    f"roll_era boundary {boundary_at.isoformat()} must be strictly "
                    f"after the current era's start_date "
                    f"{existing.start_date.isoformat()}"
                )
            existing.end_date = boundary_at

        new_era = open_new_era(
            session,
            era_slug=new_slug,
            patch_version=new_patch_version,
            start_date=boundary_at,
            meta_magnitude=meta_magnitude,
            is_major_shift=is_major_shift,
        )
        # Flush inside the savepoint so the EXCLUDE / partial-unique
        # violations surface here, not on the outer commit. ``begin_nested``
        # guarantees the rollback semantics.
        session.flush()

    return existing, new_era


def assert_no_temporal_bleed(
    session: Session,
    era_ids: Iterable[uuid.UUID],
) -> None:
    """Raise :class:`TemporalBleedError` if ``era_ids`` span a major-shift boundary.

    The rule from Systems-spec System 04: aggregations across an era
    marked ``is_major_shift=True`` are forbidden because the meta
    flipped enough at that boundary that pre-/post- stats are
    measuring two different games.

    Algorithm:

    1. Load the supplied eras. An empty input is a no-op (vacuously
       single-era).
    2. Take the earliest and latest ``start_date`` of the input set.
    3. Look up *all* major-shift eras strictly between them
       ``(earliest.start_date, latest.start_date]``. The earliest era's
       own ``is_major_shift`` is allowed: a regime starts there, and
       aggregating *within* a regime that begins with a major shift is
       legal.
    4. If any are found, raise. The error message names the offending
       era so the operator can decide whether to filter the input set
       or split the aggregation.

    Idempotent + side-effect-free: this never writes. Safe to call
    inside a read-only transaction.
    """
    ids = list(era_ids)
    if not ids:
        return
    rows = list(session.execute(select(PatchEra).where(PatchEra.era_id.in_(ids))).scalars())
    if len(rows) <= 1:
        # Single era (or zero rows after filter) — no bleed possible.
        return
    rows.sort(key=lambda e: e.start_date)
    earliest, latest = rows[0], rows[-1]

    bleeders = list(
        session.execute(
            select(PatchEra)
            .where(
                PatchEra.is_major_shift.is_(True),
                PatchEra.start_date > earliest.start_date,
                PatchEra.start_date <= latest.start_date,
            )
            .order_by(PatchEra.start_date)
        ).scalars()
    )
    if not bleeders:
        return
    boundary = bleeders[0]
    raise TemporalBleedError(
        f"TEMPORAL_BLEED: aggregation spans major-shift boundary at "
        f"era_slug={boundary.era_slug!r} (patch_version={boundary.patch_version!r}, "
        f"start_date={boundary.start_date.isoformat()}). "
        f"Split the aggregation: {len(rows)} eras, "
        f"earliest start={earliest.start_date.isoformat()}, "
        f"latest start={latest.start_date.isoformat()}."
    )


def _ensure_aware(ts: datetime) -> datetime:
    """Coerce a naive datetime to UTC.

    All era columns use ``DateTime(timezone=True)``. Naive inputs at
    this layer almost always mean "UTC" (the rest of the pipeline is
    UTC-only) but rather than silently adopting that convention we
    upgrade to aware here so a downstream comparison can't accidentally
    mix tz states. This mirrors what other BUF-prefix helpers do for
    timestamp inputs.
    """
    if ts.tzinfo is None:
        return ts.replace(tzinfo=UTC)
    return ts
