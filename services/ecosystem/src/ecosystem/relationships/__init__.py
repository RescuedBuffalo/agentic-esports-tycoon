"""Pairwise relationships with sentiment + exponential decay (BUF-26, System 07).

The substrate for every "who likes whom" question the ecosystem sim
needs to answer: ex-teammate goodwill that lingers for years, a rivalry
that bleeds out without fresh feuds, a mentor-mentee bond whose
strength erodes when the two stop scrimming together. The schema
(:class:`esports_sim.db.models.RelationshipEdge` +
:class:`esports_sim.db.models.RelationshipEvent`) carries the data;
this package owns the decay math and the steady-state job that applies
it.

Exports:

* :data:`DECAY_RATES` — per-edge-type exponential decay rate, in units
  of ``1 / week``. The dict is the load-bearing lookup the spec calls
  for; bumping a value is a downstream-feature event because the
  decayed strengths feed the GNN's edge weights.
* :func:`decay_strength` — pure scalar function. Given a starting
  strength, an edge type, and a number of weeks elapsed, returns the
  decayed strength. This is the function the regression test pins.
* :func:`decay_edge` — applies :func:`decay_strength` to a
  :class:`RelationshipEdge` in place and bumps ``last_updated_at``.
  Sentiment is left untouched (valence does not decay).
* :func:`run_monthly_decay` — the operator-facing job that reads every
  edge, applies decay, and persists. Intended to run on a monthly
  cadence (cron / scheduler hook); idempotent to re-run because the
  ``last_updated_at`` anchor moves forward each pass.
"""

from __future__ import annotations

from ecosystem.relationships.decay import (
    DECAY_RATES,
    DecayError,
    decay_edge,
    decay_strength,
    run_monthly_decay,
    weeks_between,
)

__all__ = [
    "DECAY_RATES",
    "DecayError",
    "decay_edge",
    "decay_strength",
    "run_monthly_decay",
    "weeks_between",
]
