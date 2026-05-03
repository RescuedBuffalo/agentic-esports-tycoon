"""Persistence + scheduler hook for BUF-24 patch-intent extraction.

The split mirrors the BUF-83 pattern: the LLM call (in
:mod:`extractor`) is decoupled from the DB write so tests can drive
each path independently. ``upsert_patch_intent`` writes one row;
``extract_intent_for_pending`` is the Phase 0 scheduler hook —
enumerates ``PatchNote`` rows that don't yet have an intent for the
current ``PROMPT_VERSION`` and runs the extractor against each.

Idempotency: the dedup key on ``patch_intent`` is
``(patch_note_id, prompt_version)``. A re-run of the scheduler hook
against the same corpus is a no-op once every patch has its
intent for the current prompt; bumping ``PROMPT_VERSION`` produces a
new row per patch on the next pass.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from sqlalchemy import select

from esports_sim.budget.errors import BudgetExhausted
from esports_sim.db.models import PatchIntent, PatchNote
from esports_sim.patch_intent.extractor import (
    PROMPT_VERSION,
    ExtractionOutcome,
    extract_patch_intent,
)

if TYPE_CHECKING:
    import anthropic
    from sqlalchemy.orm import Session

    from esports_sim.budget import Governor


_logger = logging.getLogger("esports_sim.patch_intent")


@dataclass
class SchedulerStats:
    """Counters returned from :func:`extract_intent_for_pending`.

    ``inserted`` is the headline metric: how many new ``patch_intent``
    rows were written this pass. ``skipped_existing`` reports patches
    that already had an intent for the current prompt version (no LLM
    call made — the row was idempotently skipped). ``budget_exhausted``
    counts patches that were skipped because the governor rejected the
    pre-flight; useful so an operator can see the hook was healthy
    until the cap was hit.
    """

    inserted: int = 0
    updated: int = 0
    skipped_existing: int = 0
    budget_exhausted: int = 0
    by_patch_version: dict[str, str] = field(default_factory=dict)


def upsert_patch_intent(
    session: Session,
    *,
    patch_note: PatchNote,
    outcome: ExtractionOutcome,
) -> tuple[PatchIntent, str]:
    """Persist (or refresh) an extraction outcome against a patch note.

    SELECT-then-INSERT-or-UPDATE on ``(patch_note_id, prompt_version)``,
    matching the pattern :func:`data_pipeline.patch_notes_runner._upsert_patch_note`
    uses on its own dedup key. Returns the row plus an outcome tag of
    ``"inserted"`` or ``"updated"`` so the caller can attribute counters
    granularly.

    The caller owns the transaction. ``session.flush()`` runs after the
    insert/update so a follow-up SELECT in the same pass sees the row.
    """
    existing: PatchIntent | None = session.execute(
        select(PatchIntent).where(
            PatchIntent.patch_note_id == patch_note.id,
            PatchIntent.prompt_version == outcome.prompt_version,
        )
    ).scalar_one_or_none()

    result = outcome.result

    if existing is None:
        new = PatchIntent(
            patch_note_id=patch_note.id,
            prompt_version=outcome.prompt_version,
            model=outcome.model,
            primary_intent=result.primary_intent,
            pro_play_driven_score=result.pro_play_driven_score,
            agents_affected=list(result.agents_affected),
            maps_affected=list(result.maps_affected),
            econ_changed=result.econ_changed,
            expected_pickrate_shifts=[
                shift.model_dump() for shift in result.expected_pickrate_shifts
            ],
            community_controversy_predicted=result.community_controversy_predicted,
            confidence=result.confidence,
            reasoning=result.reasoning,
            input_tokens=outcome.input_tokens,
            output_tokens=outcome.output_tokens,
            usd_cost=outcome.usd_cost,
        )
        session.add(new)
        session.flush()
        return new, "inserted"

    existing.model = outcome.model
    existing.primary_intent = result.primary_intent
    existing.pro_play_driven_score = result.pro_play_driven_score
    existing.agents_affected = list(result.agents_affected)
    existing.maps_affected = list(result.maps_affected)
    existing.econ_changed = result.econ_changed
    existing.expected_pickrate_shifts = [
        shift.model_dump() for shift in result.expected_pickrate_shifts
    ]
    existing.community_controversy_predicted = result.community_controversy_predicted
    existing.confidence = result.confidence
    existing.reasoning = result.reasoning
    existing.input_tokens = outcome.input_tokens
    existing.output_tokens = outcome.output_tokens
    existing.usd_cost = outcome.usd_cost
    session.flush()
    return existing, "updated"


def extract_intent_for_pending(
    session: Session,
    *,
    governor: Governor,
    client: anthropic.Anthropic | None = None,
    source: str | None = None,
    limit: int | None = None,
) -> SchedulerStats:
    """Phase 0 scheduler hook: extract intents for every patch without one.

    Driven by the new-patch event in spirit but implemented as a poll:
    enumerate ``patch_note`` rows that don't yet have a ``patch_intent``
    for the current ``PROMPT_VERSION``, then run
    :func:`extract_patch_intent` for each. Idempotent — a subsequent
    call is a no-op once every patch has its intent.

    Stops cleanly on :class:`BudgetExhausted`: increments the counter,
    logs which patch was skipped, and returns. The next scheduled pass
    will pick up where this one left off once the weekly window rolls
    forward and credits free up.

    The selected ``prompt_version`` is always the module-level
    ``PROMPT_VERSION`` constant, mirroring what the extractor stamps on
    its outcome. There is no override knob: a "v1 classification" can
    only be produced by a v1 prompt rubric, and the prompt is the
    versioned artefact — bumping ``PROMPT_VERSION`` (and the rubric in
    :mod:`prompt`) is the way to land a new generation of rows on the
    next pass. An override that filtered on a different version while
    the extractor stamped the current one would silently UPSERT the
    default-version row repeatedly while the pending list never shrunk.

    Parameters
    ----------
    source:
        Optional connector source filter (``"playvalorant"``). Useful
        for narrow re-runs against one game's history.
    limit:
        Optional cap on the number of extractions per call. ``None``
        means "process every pending patch"; a finite cap is the
        operator-facing knob for "extract one patch and let me eyeball
        the result before draining the rest of the queue".
    """
    stats = SchedulerStats()

    pending = list(_select_pending_patches(session, source=source))
    _logger.info(
        "patch_intent.scheduler_start",
        extra={"pending_count": len(pending), "prompt_version": PROMPT_VERSION},
    )

    for patch_note in pending:
        if limit is not None and (stats.inserted + stats.updated) >= limit:
            _logger.info(
                "patch_intent.scheduler_limit_reached",
                extra={"limit": limit},
            )
            break
        try:
            outcome = extract_patch_intent(
                governor=governor,
                patch_notes_text=patch_note.body_text,
                client=client,
                notes=f"patch_note_id={patch_note.id} version={patch_note.patch_version}",
            )
        except BudgetExhausted as exc:
            stats.budget_exhausted += 1
            _logger.warning(
                "patch_intent.budget_exhausted",
                extra={
                    "patch_note_id": str(patch_note.id),
                    "patch_version": patch_note.patch_version,
                    "scope": exc.scope,
                },
            )
            # Stop the loop — every subsequent extraction would hit
            # the same cap. The next scheduled pass picks up here.
            break

        _, outcome_tag = upsert_patch_intent(session, patch_note=patch_note, outcome=outcome)
        if outcome_tag == "inserted":
            stats.inserted += 1
        else:
            stats.updated += 1
        stats.by_patch_version[patch_note.patch_version] = outcome.result.primary_intent
        _logger.info(
            "patch_intent.upsert",
            extra={
                "patch_note_id": str(patch_note.id),
                "patch_version": patch_note.patch_version,
                "outcome": outcome_tag,
                "primary_intent": outcome.result.primary_intent,
                "usd_cost": outcome.usd_cost,
            },
        )

    # The pending list was computed once at the top; rows that already
    # had an intent for the current prompt_version were excluded from
    # the iteration. Report them as ``skipped_existing`` so the on-call
    # dashboard sees a clean "nothing to do" on a quiet pass.
    stats.skipped_existing = _count_existing(session, source=source)
    _logger.info(
        "patch_intent.scheduler_done",
        extra={
            "inserted": stats.inserted,
            "updated": stats.updated,
            "skipped_existing": stats.skipped_existing,
            "budget_exhausted": stats.budget_exhausted,
        },
    )
    return stats


def _select_pending_patches(
    session: Session,
    *,
    source: str | None,
) -> list[PatchNote]:
    """Patches without an intent for the current ``PROMPT_VERSION``, oldest first.

    Oldest-first so a backfill processes history in chronological
    order — the cheapest path to noticing a regression where the model
    suddenly starts mis-classifying recent patches (e.g. after a
    prompt-version bump).
    """
    # Scalar subquery returning patch_note_ids that already have an
    # intent for the current prompt_version. The outer query excludes
    # them.
    classified = (
        select(PatchIntent.patch_note_id)
        .where(PatchIntent.prompt_version == PROMPT_VERSION)
        .scalar_subquery()
    )
    stmt = (
        select(PatchNote)
        .where(PatchNote.id.notin_(classified))
        .order_by(PatchNote.published_at.asc())
    )
    if source is not None:
        stmt = stmt.where(PatchNote.source == source)
    return list(session.execute(stmt).scalars())


def _count_existing(
    session: Session,
    *,
    source: str | None,
) -> int:
    """Number of patch notes that already have an intent for the current prompt."""
    stmt = (
        select(PatchIntent)
        .join(PatchNote, PatchIntent.patch_note_id == PatchNote.id)
        .where(PatchIntent.prompt_version == PROMPT_VERSION)
    )
    if source is not None:
        stmt = stmt.where(PatchNote.source == source)
    return len(list(session.execute(stmt).scalars()))


__all__ = [
    "SchedulerStats",
    "extract_intent_for_pending",
    "upsert_patch_intent",
]
