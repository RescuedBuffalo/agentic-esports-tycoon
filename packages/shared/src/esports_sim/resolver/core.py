"""Implementation of :func:`resolve_entity` and its return types.

The decision tree matches Systems-spec System 01 verbatim:

    1. exact lookup on (platform, platform_id) -> MATCHED
    2. fuzzy match on platform_name vs every alias under entities of the same
       entity_type
        - top score >= AUTO_MERGE_THRESHOLD -> AUTO_MERGED (insert alias)
        - top score >= REVIEW_THRESHOLD     -> PENDING (enqueue review)
    3. otherwise                            -> CREATED (new entity + alias 1.0)

We use :func:`rapidfuzz.fuzz.WRatio` because its weighted blend of partial /
token-set / token-sort ratios is the most resilient default for handles that
mix casing, spacing, and substrings ("TenZ" vs "tenz", "Sentinels" vs "Team
Sentinels"). The score is normalised from rapidfuzz's 0-100 range into
[0.0, 1.0] so it is directly comparable with the EntityAlias.confidence
column.
"""

from __future__ import annotations

import enum
import uuid
from dataclasses import dataclass, field

from rapidfuzz import fuzz, utils
from sqlalchemy import select
from sqlalchemy.orm import Session

from esports_sim.db.enums import EntityType, Platform, ReviewStatus, StagingStatus
from esports_sim.db.models import (
    AliasReviewQueue,
    Entity,
    EntityAlias,
    StagingRecord,
)

# Thresholds from the Systems-spec. Kept module-level constants (rather than
# call-site arguments with defaults) because they're a policy decision the
# whole pipeline must agree on — an experiment that wants different numbers
# should override them globally for the duration of the run, not call-by-call.
REVIEW_THRESHOLD: float = 0.70
AUTO_MERGE_THRESHOLD: float = 0.90

# rapidfuzz returns 0-100; the alias confidence column lives in [0,1]. Convert
# at the boundary so nothing downstream has to remember the scale.
_SCORE_DIVISOR: float = 100.0

# Cap on how many candidates we serialise into the review queue. The reviewer
# only ever needs the best handful; bloating the row hurts both DB and UI.
_MAX_REVIEW_CANDIDATES: int = 5


class ResolutionStatus(enum.StrEnum):
    """The four terminal states of :func:`resolve_entity`."""

    MATCHED = "matched"
    AUTO_MERGED = "auto_merged"
    PENDING = "pending"
    CREATED = "created"


@dataclass(frozen=True)
class ResolveCandidate:
    """A near-miss surfaced to the human reviewer.

    Stored verbatim in :class:`AliasReviewQueue.candidates` (as JSON) and also
    handed back to the caller so a CLI driver can render the same shortlist
    without re-querying.
    """

    canonical_id: uuid.UUID
    name: str
    score: float

    def to_json(self) -> dict[str, object]:
        return {
            "canonical_id": str(self.canonical_id),
            "name": self.name,
            "score": round(self.score, 4),
        }


@dataclass(frozen=True)
class ResolveResult:
    """Outcome of one :func:`resolve_entity` call.

    ``canonical_id`` is ``None`` only for :attr:`ResolutionStatus.PENDING`;
    every other status carries a non-null id the caller can use to set
    ``StagingRecord.canonical_id`` and proceed.
    """

    status: ResolutionStatus
    canonical_id: uuid.UUID | None
    confidence: float
    candidates: list[ResolveCandidate] = field(default_factory=list)


def resolve_entity(
    session: Session,
    *,
    platform: Platform,
    platform_id: str,
    platform_name: str,
    entity_type: EntityType,
) -> ResolveResult:
    """Map a scraper handle to the canonical record. Single chokepoint.

    See module docstring for the decision tree. The function performs all of
    its writes inside the caller's :class:`Session` and does *not* commit;
    the caller owns the transaction.
    """
    if not platform_id:
        # Defensive: an empty platform_id would collide on the (platform, "")
        # uniqueness key and silently merge unrelated rows. Fail loudly.
        raise ValueError("platform_id must be non-empty")
    if not platform_name:
        raise ValueError("platform_name must be non-empty")

    exact_hit = _lookup_exact_alias(session, platform=platform, platform_id=platform_id)
    if exact_hit is not None:
        return ResolveResult(
            status=ResolutionStatus.MATCHED,
            canonical_id=exact_hit.canonical_id,
            confidence=exact_hit.confidence,
        )

    candidates = _fuzzy_candidates(
        session,
        query_name=platform_name,
        entity_type=entity_type,
    )

    if candidates and candidates[0].score >= AUTO_MERGE_THRESHOLD:
        top = candidates[0]
        _insert_alias(
            session,
            canonical_id=top.canonical_id,
            platform=platform,
            platform_id=platform_id,
            platform_name=platform_name,
            confidence=top.score,
        )
        return ResolveResult(
            status=ResolutionStatus.AUTO_MERGED,
            canonical_id=top.canonical_id,
            confidence=top.score,
            candidates=candidates,
        )

    if candidates and candidates[0].score >= REVIEW_THRESHOLD:
        # Idempotent review enqueue: if a pending row already exists for this
        # (platform, platform_id) the second call must be a no-op so a retried
        # scraper doesn't pile up duplicate human-review work.
        existing = _find_pending_review(session, platform=platform, platform_id=platform_id)
        if existing is None:
            session.add(
                AliasReviewQueue(
                    platform=platform,
                    platform_id=platform_id,
                    platform_name=platform_name,
                    candidates=[c.to_json() for c in candidates],
                    reason="fuzzy_below_auto_merge",
                    status=ReviewStatus.PENDING,
                )
            )
            session.flush()
        return ResolveResult(
            status=ResolutionStatus.PENDING,
            canonical_id=None,
            confidence=candidates[0].score,
            candidates=candidates,
        )

    # No exact, no fuzzy hit — mint a brand-new canonical and seed alias 1.0.
    new_entity = Entity(entity_type=entity_type)
    session.add(new_entity)
    session.flush()  # populate canonical_id before we reference it
    _insert_alias(
        session,
        canonical_id=new_entity.canonical_id,
        platform=platform,
        platform_id=platform_id,
        platform_name=platform_name,
        confidence=1.0,
    )
    return ResolveResult(
        status=ResolutionStatus.CREATED,
        canonical_id=new_entity.canonical_id,
        confidence=1.0,
    )


# --- helpers ------------------------------------------------------------------


def _lookup_exact_alias(
    session: Session, *, platform: Platform, platform_id: str
) -> EntityAlias | None:
    stmt = select(EntityAlias).where(
        EntityAlias.platform == platform,
        EntityAlias.platform_id == platform_id,
    )
    return session.execute(stmt).scalar_one_or_none()


def _fuzzy_candidates(
    session: Session,
    *,
    query_name: str,
    entity_type: EntityType,
) -> list[ResolveCandidate]:
    """Score the query name against every alias under same-type entities.

    Returns the top-N candidates sorted by score descending, deduplicated by
    canonical_id (each entity contributes its single best-matching alias).
    Empty list when there are no entities of this type yet.
    """
    stmt = (
        select(EntityAlias.canonical_id, EntityAlias.platform_name)
        .join(Entity, Entity.canonical_id == EntityAlias.canonical_id)
        .where(Entity.entity_type == entity_type, Entity.is_active.is_(True))
    )
    rows = session.execute(stmt).all()
    if not rows:
        return []

    best_per_canonical: dict[uuid.UUID, ResolveCandidate] = {}
    for canonical_id, alias_name in rows:
        # default_process lowercases, strips non-alnum, and collapses whitespace.
        # Without it, "TenZ" vs "tenz" scores 0.50 and would land on the review
        # queue instead of auto-merging — the BUF-7 acceptance scenario fails.
        score = (
            fuzz.WRatio(query_name, alias_name, processor=utils.default_process) / _SCORE_DIVISOR
        )
        prior = best_per_canonical.get(canonical_id)
        if prior is None or score > prior.score:
            best_per_canonical[canonical_id] = ResolveCandidate(
                canonical_id=canonical_id, name=alias_name, score=score
            )

    ranked = sorted(best_per_canonical.values(), key=lambda c: c.score, reverse=True)
    return ranked[:_MAX_REVIEW_CANDIDATES]


def _insert_alias(
    session: Session,
    *,
    canonical_id: uuid.UUID,
    platform: Platform,
    platform_id: str,
    platform_name: str,
    confidence: float,
) -> EntityAlias:
    alias = EntityAlias(
        canonical_id=canonical_id,
        platform=platform,
        platform_id=platform_id,
        platform_name=platform_name,
        confidence=confidence,
    )
    session.add(alias)
    session.flush()
    return alias


def _find_pending_review(
    session: Session, *, platform: Platform, platform_id: str
) -> AliasReviewQueue | None:
    # ``alias_review_queue`` has no uniqueness constraint over
    # (platform, platform_id, status=pending) and the enqueue path is
    # non-atomic, so two concurrent resolver calls can each insert a pending
    # row before either notices the other. A third call's idempotency check
    # would then see two rows; ``.scalar_one_or_none()`` would crash with
    # ``MultipleResultsFound``, turning the retry path into a hard failure.
    # Take the oldest pending row instead: any pending match is enough to
    # treat the (platform, platform_id) as already queued.
    stmt = (
        select(AliasReviewQueue)
        .where(
            AliasReviewQueue.platform == platform,
            AliasReviewQueue.platform_id == platform_id,
            AliasReviewQueue.status == ReviewStatus.PENDING,
        )
        .order_by(AliasReviewQueue.created_at)
        .limit(1)
    )
    return session.execute(stmt).scalars().first()


# Re-export so callers writing ``from esports_sim.resolver.core import ...``
# can grab the model symbols too. Keeping them visible here makes the
# resolver's contract self-contained: every public name a caller might want
# to type-annotate against is reachable from one module.
__all__ = [
    "AUTO_MERGE_THRESHOLD",
    "REVIEW_THRESHOLD",
    "ResolutionStatus",
    "ResolveCandidate",
    "ResolveResult",
    "resolve_entity",
    "StagingRecord",  # re-exported so resolve callers don't dual-import
    "StagingStatus",
]
