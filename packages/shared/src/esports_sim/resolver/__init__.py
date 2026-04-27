"""Canonical-id resolver (BUF-7, Systems-spec System 01).

Every write to a canonical-keyed table funnels through :func:`resolve_entity`.
The resolver decides one of four outcomes for a ``(platform, platform_id,
platform_name)`` triple:

* **MATCHED** — an alias row already exists for ``(platform, platform_id)``;
  return its ``canonical_id`` unchanged.
* **AUTO_MERGED** — fuzzy similarity to an existing canonical's name is at or
  above :data:`AUTO_MERGE_THRESHOLD`; insert a new alias under the existing
  canonical at the match score and return.
* **PENDING** — fuzzy similarity sits in ``[REVIEW_THRESHOLD, AUTO_MERGE_THRESHOLD)``;
  enqueue an :class:`~esports_sim.db.models.AliasReviewQueue` row with the
  candidate list and return :class:`ResolutionStatus.PENDING`. No write to the
  canonical tables happens.
* **CREATED** — no candidate clears :data:`REVIEW_THRESHOLD`; mint a new
  :class:`~esports_sim.db.models.Entity` plus an alias at confidence 1.0.

The resolver is the *only* sanctioned entry point for inserting alias rows.
Direct inserts into ``entity_alias`` from a scraper are a layering violation;
the staging guard in :mod:`esports_sim.db.models` is the runtime backstop that
catches scrapers that try to bypass it.

Idempotency: a second call with the same ``(platform, platform_id)`` after a
prior MATCHED/AUTO_MERGED/CREATED returns :class:`ResolutionStatus.MATCHED`
with the same canonical_id. A second call after PENDING returns PENDING and
does not enqueue a duplicate review row.
"""

from esports_sim.resolver.core import (
    AUTO_MERGE_THRESHOLD,
    REVIEW_THRESHOLD,
    ResolutionStatus,
    ResolveCandidate,
    ResolveResult,
    resolve_entity,
)

__all__ = [
    "AUTO_MERGE_THRESHOLD",
    "REVIEW_THRESHOLD",
    "ResolutionStatus",
    "ResolveCandidate",
    "ResolveResult",
    "resolve_entity",
]
