"""Patch-era partitioning (BUF-13, Systems-spec System 04).

Every record that lands in the canonical store carries an era context.
Cross-era feature aggregation is not allowed; the
:func:`assert_no_temporal_bleed` guard is the runtime backstop the
ticket's ``TEMPORAL_BLEED`` acceptance test exercises.

Public API:

* :func:`assign_era` — map a timestamp to the ``era_id`` whose
  ``[start_date, end_date)`` window covers it.
* :func:`current_era` — read the open era (the one with
  ``end_date IS NULL``); the partial unique index on the table
  guarantees at most one.
* :func:`roll_era` — atomic close-then-open transactional pair. Stamps
  the previous era's ``end_date`` and the new era's ``start_date`` with
  the *same* timestamp under one savepoint, so a crash mid-roll can't
  produce a gap or an overlap.
* :func:`assert_no_temporal_bleed` — raise
  :class:`~esports_sim.db.models.TemporalBleedError` if the supplied
  era_ids span any boundary marked ``is_major_shift=True``.
"""

from esports_sim.eras.core import (
    EraNotFoundError,
    EraOverlapError,
    assert_no_temporal_bleed,
    assign_era,
    current_era,
    open_new_era,
    roll_era,
)

__all__ = [
    "EraNotFoundError",
    "EraOverlapError",
    "assert_no_temporal_bleed",
    "assign_era",
    "current_era",
    "open_new_era",
    "roll_era",
]
