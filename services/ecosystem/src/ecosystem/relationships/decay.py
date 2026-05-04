"""Exponential decay of :class:`RelationshipEdge` strength over time.

System 07's load-bearing claim: a relationship without fresh signal
fades. The math is the simplest model that respects the obvious
constraints (monotone decreasing, multiplicative composition over
sub-intervals, parameterised per edge kind):

.. math::

    s(t + \\Delta t) = s(t) \\cdot e^{-r \\cdot \\Delta t}

where ``r`` is :data:`DECAY_RATES` ``[edge_type]`` (units: ``1 /
week``) and ``Δt`` is the elapsed time in weeks.

Two decay reads of the same edge over disjoint intervals compose to
the same result as a single read across the union — that's what makes
the monthly job idempotent under jitter. The
:func:`run_monthly_decay` job advances ``last_updated_at`` to ``now``
so the next pass measures from the refreshed anchor.

Sentiment does **not** decay. Valence is a permanent feature of the
relationship; an ex-teammate keeps positive sentiment after their
strength has bled out, and a feud rival keeps negative sentiment
through the same window. Only the magnitude (``strength``) is
time-eroding.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from datetime import datetime

from esports_sim.db.enums import RelationshipEdgeType
from esports_sim.db.models import RelationshipEdge
from sqlalchemy import select
from sqlalchemy.orm import Session

_logger = logging.getLogger("ecosystem.relationships.decay")


class DecayError(ValueError):
    """Raised when :func:`decay_strength` is asked to operate on an unknown edge type.

    Better to fail loudly here than to fall back to a default rate and
    silently let edge weights drift in production. Adding a new edge
    kind requires a coordinated update to
    :class:`RelationshipEdgeType`, the migration's ENUM list, and
    :data:`DECAY_RATES`; missing the third surfaces here.
    """


# Per-edge-type decay rates in units of ``1 / week``. The half-life
# implied by each rate is ``ln(2) / r``:
#
# * teammate (0.020) ≈ 35-week half-life — a current-roster bond stays
#   strong through a season, slowly fades across the off-season if the
#   pair stops sharing scrims.
# * ex_teammate (0.010) ≈ 69-week half-life — bonds that survive a
#   roster move take more than a year to lose half their pull.
# * rival (0.015) ≈ 46-week half-life — rivalries cool faster than
#   ex-teammate bonds when no new feud signal lands.
# * friend (0.012) ≈ 58-week half-life — close to ex-teammate; the
#   social tie outlives the professional one.
# * mentor (0.005) ≈ 139-week half-life — coach / mentor bonds are
#   the slowest to fade; the relationship persists even in long
#   silences.
# * manager_of (0.008) ≈ 87-week half-life — operational bonds
#   between a manager and a team member fall in between.
#
# Tuning these is a downstream-feature event: the GNN's edge weights
# (:mod:`ecosystem.graph.builder`) read the decayed strengths, so a
# rate change that lands without re-decay produces a torn snapshot
# where some edges measured against the old rate and others against
# the new. Bump the schema version on
# :data:`ecosystem.graph.schema.SCHEMA_VERSION` when changing these.
DECAY_RATES: Mapping[RelationshipEdgeType, float] = {
    RelationshipEdgeType.TEAMMATE: 0.020,
    RelationshipEdgeType.EX_TEAMMATE: 0.010,
    RelationshipEdgeType.RIVAL: 0.015,
    RelationshipEdgeType.FRIEND: 0.012,
    RelationshipEdgeType.MENTOR: 0.005,
    RelationshipEdgeType.MANAGER_OF: 0.008,
}


# Number of seconds in one week. Hardcoded so the implementation
# doesn't pull in ``timedelta`` for what is effectively a constant; the
# accuracy of ``Δt`` in weeks is the only thing the math depends on.
_SECONDS_PER_WEEK: float = 7.0 * 24.0 * 3600.0


def weeks_between(earlier: datetime, later: datetime) -> float:
    """Return ``later - earlier`` in fractional weeks, never negative.

    A clock skew or out-of-order writer that produces ``later <
    earlier`` would otherwise grow strength on the next decay pass —
    explicitly clamped to zero so that case is a no-op rather than a
    silent corruption.
    """
    delta = (later - earlier).total_seconds()
    if delta <= 0:
        return 0.0
    return delta / _SECONDS_PER_WEEK


def decay_strength(
    strength: float,
    *,
    edge_type: RelationshipEdgeType,
    weeks: float,
) -> float:
    """Apply ``strength * exp(-rate * weeks)`` for ``edge_type``'s rate.

    Pure function — no side effects, no DB access. The regression
    test for BUF-26's acceptance pins this directly: an edge with
    ``strength = 1.0`` and ``weeks = 20`` and the TEAMMATE rate
    decays to ``exp(-0.4) ≈ 0.6703200460356393``.

    Negative ``weeks`` is treated as a no-op (returns the input
    unchanged) — the same clamp :func:`weeks_between` applies. The
    result is also clamped into ``[0, 1]`` so floating-point fuzz
    around the boundary cannot trip the column's CHECK constraint.
    """
    if not 0.0 <= strength <= 1.0:
        raise DecayError(
            f"strength={strength!r} is outside the [0, 1] range; "
            "the column's CHECK constraint guards this in the DB."
        )
    if weeks <= 0:
        return strength
    try:
        rate = DECAY_RATES[edge_type]
    except KeyError as e:
        raise DecayError(
            f"no DECAY_RATES entry for edge_type={edge_type!r}; "
            f"known: {sorted(t.value for t in DECAY_RATES)}"
        ) from e
    decayed = strength * math.exp(-rate * weeks)
    # Clamp to the column's CHECK range. ``exp(-rate * weeks)`` is
    # bounded in ``(0, 1]`` for non-negative inputs so this is a
    # belt-and-braces guard against floating-point fuzz.
    if decayed < 0.0:
        return 0.0
    if decayed > 1.0:
        return 1.0
    return decayed


def decay_edge(edge: RelationshipEdge, *, now: datetime) -> RelationshipEdge:
    """Apply decay to ``edge`` in place and advance ``last_updated_at``.

    The returned reference is the same instance — the helper is a
    mutator, not a constructor. ``sentiment`` is left untouched.
    Re-calling on the same edge is a no-op once the anchor has been
    moved forward to ``now``, which is what makes
    :func:`run_monthly_decay` idempotent under jitter.

    The caller is responsible for the ORM session bookkeeping — this
    function just mutates the in-memory row. ``run_monthly_decay``
    composes them.
    """
    weeks = weeks_between(edge.last_updated_at, now)
    if weeks == 0.0:
        # The anchor is already at-or-ahead of ``now``. Don't move it
        # backward; just no-op.
        return edge
    edge.strength = decay_strength(
        edge.strength,
        edge_type=edge.edge_type,
        weeks=weeks,
    )
    edge.last_updated_at = now
    return edge


def run_monthly_decay(
    session: Session,
    *,
    now: datetime,
) -> int:
    """Apply :func:`decay_edge` to every row in ``relationship_edge``.

    Returns the count of edges touched. Intended cadence is monthly,
    but the function itself is cadence-agnostic — calling it twice in
    quick succession is harmless because the anchor moves forward each
    pass and the second call sees zero elapsed weeks.

    The implementation walks the table in Python rather than issuing a
    bulk UPDATE for two reasons:

    1. ``DECAY_RATES`` is a Python dict; the per-type rate would have
       to be folded into the SQL via a CASE per kind, which doesn't
       generalise as the kind set grows.
    2. The monthly volume is small (one row per pair-kind, bounded by
       roster turnover) — Python-side iteration is operationally
       cheaper than maintaining a bespoke UPDATE statement.

    The caller owns the transaction. We do **not** commit here so the
    job can compose with sibling work in the same session.
    """
    rows = session.execute(select(RelationshipEdge)).scalars().all()
    touched = 0
    for edge in rows:
        previous_strength = edge.strength
        previous_anchor = edge.last_updated_at
        decay_edge(edge, now=now)
        # Count only rows where decay actually moved state. The
        # earlier ``or last_updated_at == now`` clause was true for
        # rows already anchored at ``now`` from a prior pass — an
        # immediate rerun would log work that didn't happen and
        # break the function's idempotence signal (Codex P2 on PR
        # #28).
        if edge.strength != previous_strength or edge.last_updated_at != previous_anchor:
            touched += 1
    _logger.info(
        "monthly_decay applied",
        extra={"edges_touched": touched, "total_edges": len(rows)},
    )
    session.flush()
    return touched
