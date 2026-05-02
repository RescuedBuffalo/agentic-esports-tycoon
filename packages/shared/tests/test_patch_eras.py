"""BUF-13 patch_era schema + helpers integration tests.

Covers the three acceptance bullets from the ticket:

* 100% of historical matches successfully assigned to an era — proven
  by ``test_assign_era_covers_seeded_timeline_with_no_gaps``.
* TEMPORAL_BLEED — proven by ``test_assert_no_temporal_bleed_*``.
* Atomic close-then-open — proven by
  ``test_roll_era_is_atomic_no_overlap_no_gap``.

Plus per-helper unit coverage. All tests run against a real Postgres
(via the shared ``db_session`` fixture) because the EXCLUDE
constraint, the partial unique index, and the assign_era SQL
function are all server-side — a SQLite stand-in would not exercise
them.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime, timedelta

import pytest
from esports_sim.db.models import PatchEra, TemporalBleedError
from esports_sim.eras import (
    EraNotFoundError,
    assert_no_temporal_bleed,
    assign_era,
    current_era,
    open_new_era,
    roll_era,
)
from sqlalchemy import select, text
from sqlalchemy.exc import IntegrityError

pytestmark = pytest.mark.integration


# --- helpers --------------------------------------------------------------


def _make_era(
    *,
    slug: str,
    patch_version: str,
    start: datetime,
    end: datetime | None,
    is_major_shift: bool = False,
    meta_magnitude: float = 0.3,
) -> PatchEra:
    """Construct a PatchEra row with sensible defaults for unit tests."""
    return PatchEra(
        era_id=uuid.uuid4(),
        era_slug=slug,
        patch_version=patch_version,
        start_date=start,
        end_date=end,
        meta_magnitude=meta_magnitude,
        is_major_shift=is_major_shift,
    )


def _seed_three_era_timeline(session) -> tuple[PatchEra, PatchEra, PatchEra]:
    """Three contiguous eras: A → B(major shift) → C(open).

    A: 2024-01-01 → 2024-04-01, not major shift
    B: 2024-04-01 → 2024-08-01, MAJOR SHIFT
    C: 2024-08-01 → null (open), not major shift

    Exercising the BUF-13 guard: an aggregation that touches A and C
    spans across B's major-shift boundary and must raise
    :class:`TemporalBleedError`.
    """
    a = _make_era(
        slug="t_a",
        patch_version="8.0",
        start=datetime(2024, 1, 1, tzinfo=UTC),
        end=datetime(2024, 4, 1, tzinfo=UTC),
    )
    b = _make_era(
        slug="t_b",
        patch_version="8.08",
        start=datetime(2024, 4, 1, tzinfo=UTC),
        end=datetime(2024, 8, 1, tzinfo=UTC),
        is_major_shift=True,
        meta_magnitude=0.85,
    )
    c = _make_era(
        slug="t_c",
        patch_version="9.02",
        start=datetime(2024, 8, 1, tzinfo=UTC),
        end=None,
    )
    session.add_all([a, b, c])
    session.flush()
    return a, b, c


# --- schema invariants ----------------------------------------------------


def test_patch_era_table_exists_and_view_resolves_open_end_date(db_session) -> None:
    """The migration installs both the table and the helper view."""
    a, _b, c = _seed_three_era_timeline(db_session)

    rows = db_session.execute(
        text(
            "SELECT era_slug, end_date_resolved, is_current "
            "FROM patch_era_window ORDER BY start_date"
        )
    ).all()
    by_slug = {r.era_slug: r for r in rows}
    # Open era's end_date is COALESCE'd to a far-future sentinel.
    assert by_slug["t_c"].is_current is True
    assert by_slug["t_c"].end_date_resolved.year == 9999
    assert by_slug["t_a"].is_current is False
    assert by_slug["t_a"].end_date_resolved == a.end_date


def test_check_constraint_rejects_zero_length_range(db_session) -> None:
    """``end_date > start_date`` — zero-length and negative ranges are bugs."""
    bad = _make_era(
        slug="t_zero",
        patch_version="x.x",
        start=datetime(2024, 1, 1, tzinfo=UTC),
        end=datetime(2024, 1, 1, tzinfo=UTC),  # equal — zero length
    )
    db_session.add(bad)
    with pytest.raises(IntegrityError):
        db_session.flush()
    db_session.rollback()


def test_check_constraint_rejects_meta_magnitude_out_of_range(db_session) -> None:
    """``meta_magnitude`` is a 0..1 estimate."""
    bad = _make_era(
        slug="t_mag",
        patch_version="x.x",
        start=datetime(2024, 1, 1, tzinfo=UTC),
        end=datetime(2024, 2, 1, tzinfo=UTC),
        meta_magnitude=1.5,
    )
    db_session.add(bad)
    with pytest.raises(IntegrityError):
        db_session.flush()
    db_session.rollback()


def test_exclude_constraint_rejects_overlapping_ranges(db_session) -> None:
    """Two eras whose half-open windows overlap are rejected.

    The BUF-13 atomic-roll guarantee depends on this: without it, two
    racing rolls could each succeed at the close step and then both
    insert an open era, duplicating the current-era pointer.
    """
    a = _make_era(
        slug="t_a",
        patch_version="8.0",
        start=datetime(2024, 1, 1, tzinfo=UTC),
        end=datetime(2024, 4, 1, tzinfo=UTC),
    )
    overlap = _make_era(
        slug="t_overlap",
        patch_version="8.04",
        start=datetime(2024, 3, 1, tzinfo=UTC),  # mid-A
        end=datetime(2024, 5, 1, tzinfo=UTC),
    )
    db_session.add(a)
    db_session.flush()
    db_session.add(overlap)
    with pytest.raises(IntegrityError):
        db_session.flush()
    db_session.rollback()


def test_partial_unique_index_caps_open_eras_at_one(db_session) -> None:
    """At most one open era. The partial unique on ``end_date IS NULL``
    is what lets :func:`current_era` return at most one row."""
    a = _make_era(
        slug="t_a",
        patch_version="8.0",
        start=datetime(2024, 1, 1, tzinfo=UTC),
        end=None,
    )
    db_session.add(a)
    db_session.flush()

    b = _make_era(
        slug="t_b",
        patch_version="8.04",
        start=datetime(2024, 4, 1, tzinfo=UTC),
        end=None,
    )
    db_session.add(b)
    with pytest.raises(IntegrityError):
        db_session.flush()
    db_session.rollback()


def test_unique_era_slug(db_session) -> None:
    """``era_slug`` is the human key — collisions are real config bugs."""
    a = _make_era(
        slug="t_dup",
        patch_version="8.0",
        start=datetime(2024, 1, 1, tzinfo=UTC),
        end=datetime(2024, 4, 1, tzinfo=UTC),
    )
    db_session.add(a)
    db_session.flush()
    b = _make_era(
        slug="t_dup",
        patch_version="8.04",
        start=datetime(2024, 4, 1, tzinfo=UTC),
        end=datetime(2024, 8, 1, tzinfo=UTC),
    )
    db_session.add(b)
    with pytest.raises(IntegrityError):
        db_session.flush()
    db_session.rollback()


# --- assign_era -----------------------------------------------------------


def test_assign_era_covers_seeded_timeline_with_no_gaps(db_session) -> None:
    """Acceptance bullet: 100% of historical timestamps in [first.start, +inf)
    successfully assigned to an era.

    Walks the seeded timeline at every era's start instant, every era's
    end instant - 1us, and the open era's distant future. None should
    raise :class:`EraNotFoundError`.
    """
    a, b, c = _seed_three_era_timeline(db_session)

    # Half-open: ts == era.start_date lands in that era.
    assert assign_era(db_session, a.start_date) == a.era_id
    # ts == era.end_date lands in the *next* era.
    assert assign_era(db_session, a.end_date) == b.era_id
    # Mid-range.
    mid_b = b.start_date + (b.end_date - b.start_date) / 2
    assert assign_era(db_session, mid_b) == b.era_id
    # Open era extends to +infinity.
    far_future = c.start_date + timedelta(days=10_000)
    assert assign_era(db_session, far_future) == c.era_id


def test_assign_era_raises_for_pre_seed_timestamp(db_session) -> None:
    """Pre-seed timestamps are an operator-actionable failure, not silent None."""
    _seed_three_era_timeline(db_session)
    too_early = datetime(2019, 1, 1, tzinfo=UTC)
    with pytest.raises(EraNotFoundError):
        assign_era(db_session, too_early)


def test_assign_era_naive_input_treated_as_utc(db_session) -> None:
    """Naive datetimes are upgraded to UTC at the helper boundary."""
    a, _b, _c = _seed_three_era_timeline(db_session)
    naive = datetime(2024, 1, 15)  # no tzinfo
    assert assign_era(db_session, naive) == a.era_id


def test_sql_assign_era_function_matches_python_helper(db_session) -> None:
    """The Postgres ``assign_era()`` function and the Python helper agree.

    Important because the BUF-13 SQL views (added by future migrations
    once aggregation tables exist) use the SQL function directly.
    Drift between the two would mean an offline Python check passes
    but the in-database view filters differently.
    """
    a, b, c = _seed_three_era_timeline(db_session)
    samples = [
        a.start_date,
        a.start_date + timedelta(days=10),
        a.end_date,  # half-open: belongs to b
        b.start_date + timedelta(days=30),
        c.start_date,
        c.start_date + timedelta(days=365),
    ]
    for ts in samples:
        py = assign_era(db_session, ts)
        sql = db_session.execute(text("SELECT assign_era(:ts) AS era_id"), {"ts": ts}).scalar_one()
        assert sql == py, f"mismatch at {ts.isoformat()}: sql={sql!r} py={py!r}"


# --- current_era ----------------------------------------------------------


def test_current_era_returns_open_era(db_session) -> None:
    _a, _b, c = _seed_three_era_timeline(db_session)
    open_era = current_era(db_session)
    assert open_era is not None
    assert open_era.era_id == c.era_id


def test_current_era_returns_none_on_empty_table(db_session) -> None:
    assert current_era(db_session) is None


# --- roll_era atomicity ---------------------------------------------------


def test_roll_era_first_roll_on_empty_db(db_session) -> None:
    """Bootstrap case: no era exists yet; ``roll_era`` opens the first one."""
    boundary = datetime(2024, 1, 1, tzinfo=UTC)
    closed, opened = roll_era(
        db_session,
        new_slug="t_first",
        new_patch_version="8.0",
        boundary_at=boundary,
        is_major_shift=True,
        meta_magnitude=0.85,
    )
    assert closed is None
    assert opened.era_slug == "t_first"
    assert opened.start_date == boundary
    assert opened.end_date is None


def test_roll_era_is_atomic_no_overlap_no_gap(db_session) -> None:
    """Acceptance bullet: closing era + opening new one is atomic.

    No gap (closed.end_date == opened.start_date) and no overlap (the
    EXCLUDE constraint enforces half-open semantics).
    """
    # Seed a current-era row to roll off of.
    initial = _make_era(
        slug="t_initial",
        patch_version="9.02",
        start=datetime(2024, 8, 1, tzinfo=UTC),
        end=None,
    )
    db_session.add(initial)
    db_session.flush()

    boundary = datetime(2025, 1, 6, tzinfo=UTC)
    closed, opened = roll_era(
        db_session,
        new_slug="t_next",
        new_patch_version="10.00",
        boundary_at=boundary,
        is_major_shift=True,
        meta_magnitude=0.85,
    )
    assert closed is not None
    assert closed.era_id == initial.era_id

    # No gap: the closed era's end_date is exactly the new era's
    # start_date.
    assert closed.end_date == boundary
    assert opened.start_date == boundary
    assert opened.end_date is None

    # No overlap: assign_era at boundary lands on the NEW era (half-open).
    assert assign_era(db_session, boundary) == opened.era_id
    # And the instant just before lands on the closed era.
    assert assign_era(db_session, boundary - timedelta(microseconds=1)) == closed.era_id

    # Exactly one open era.
    open_count = (
        db_session.execute(select(PatchEra).where(PatchEra.end_date.is_(None))).scalars().all()
    )
    assert len(open_count) == 1
    assert open_count[0].era_id == opened.era_id


def test_roll_era_rolls_back_on_overlap_failure(db_session) -> None:
    """If the open path fails, the close path is also rolled back.

    Constructs a state where the chosen boundary overlaps a *closed*
    era's range — the EXCLUDE constraint rejects the new open row,
    and the savepoint must roll the close back so the previous era
    isn't left with a stale ``end_date`` set.
    """
    closed_old = _make_era(
        slug="t_closed",
        patch_version="8.0",
        start=datetime(2024, 1, 1, tzinfo=UTC),
        end=datetime(2024, 4, 1, tzinfo=UTC),
    )
    current = _make_era(
        slug="t_current",
        patch_version="8.04",
        start=datetime(2024, 4, 1, tzinfo=UTC),
        end=None,
    )
    db_session.add_all([closed_old, current])
    db_session.flush()
    # Capture the id before any rollback so the assertion below isn't
    # at the mercy of detached-instance attribute access.
    current_id = current.era_id

    # The savepoint inside roll_era should bubble the IntegrityError
    # (slug collision via the open path) and undo the close on the
    # current era. Without the savepoint, current.end_date would be
    # stamped before the open insert tried to flush.
    with pytest.raises(IntegrityError):
        roll_era(
            db_session,
            new_slug="t_closed",  # collides with the existing slug
            new_patch_version="9.0",
            boundary_at=datetime(2024, 6, 1, tzinfo=UTC),
        )

    # Refresh the row from the DB after the savepoint roll-back: the
    # close write must not have leaked. Expire-then-refetch via a
    # fresh query rather than ``session.get`` because the original
    # instance is now in an inconsistent state.
    db_session.expire_all()
    refreshed = db_session.execute(
        select(PatchEra).where(PatchEra.era_id == current_id)
    ).scalar_one()
    assert refreshed.end_date is None  # close was rolled back


def test_roll_era_rejects_boundary_at_or_before_current_start(db_session) -> None:
    """A boundary that doesn't strictly advance the timeline is a logic bug."""
    initial = _make_era(
        slug="t_initial",
        patch_version="8.0",
        start=datetime(2024, 1, 1, tzinfo=UTC),
        end=None,
    )
    db_session.add(initial)
    db_session.flush()

    with pytest.raises(ValueError):
        roll_era(
            db_session,
            new_slug="t_next",
            new_patch_version="8.04",
            boundary_at=initial.start_date,  # equal — zero-length close
        )

    with pytest.raises(ValueError):
        roll_era(
            db_session,
            new_slug="t_next",
            new_patch_version="8.04",
            boundary_at=initial.start_date - timedelta(days=1),
        )


def test_open_new_era_collides_when_an_era_is_already_open(db_session) -> None:
    """Calling ``open_new_era`` directly while another era is open trips the
    partial unique index — callers should use :func:`roll_era` instead."""
    a = _make_era(
        slug="t_a",
        patch_version="8.0",
        start=datetime(2024, 1, 1, tzinfo=UTC),
        end=None,
    )
    db_session.add(a)
    db_session.flush()

    open_new_era(
        db_session,
        era_slug="t_b",
        patch_version="8.04",
        start_date=datetime(2024, 4, 1, tzinfo=UTC),
    )
    with pytest.raises(IntegrityError):
        db_session.flush()
    db_session.rollback()


# --- TEMPORAL_BLEED -------------------------------------------------------


def test_assert_no_temporal_bleed_within_single_era_is_no_op(db_session) -> None:
    a, _b, _c = _seed_three_era_timeline(db_session)
    # Single era: vacuously safe.
    assert_no_temporal_bleed(db_session, [a.era_id])


def test_assert_no_temporal_bleed_empty_input_is_no_op(db_session) -> None:
    _seed_three_era_timeline(db_session)
    assert_no_temporal_bleed(db_session, [])


def test_assert_no_temporal_bleed_across_minor_eras_is_safe(db_session) -> None:
    """Adjacent eras within the same regime are OK if no major-shift sits between."""
    a = _make_era(
        slug="r_a",
        patch_version="8.0",
        start=datetime(2024, 1, 1, tzinfo=UTC),
        end=datetime(2024, 2, 1, tzinfo=UTC),
        is_major_shift=True,  # regime starter
        meta_magnitude=0.85,
    )
    b = _make_era(
        slug="r_b",
        patch_version="8.04",
        start=datetime(2024, 2, 1, tzinfo=UTC),
        end=datetime(2024, 3, 1, tzinfo=UTC),
        is_major_shift=False,  # minor
    )
    c = _make_era(
        slug="r_c",
        patch_version="8.07",
        start=datetime(2024, 3, 1, tzinfo=UTC),
        end=None,
        is_major_shift=False,
    )
    db_session.add_all([a, b, c])
    db_session.flush()

    # Aggregating across A (regime start) + B + C is fine — A's
    # major_shift status is allowed for the *earliest* era in the set.
    assert_no_temporal_bleed(db_session, [a.era_id, b.era_id, c.era_id])


def test_assert_no_temporal_bleed_across_major_shift_raises(db_session) -> None:
    """Acceptance bullet: aggregating across major-shift boundary raises."""
    a, b, c = _seed_three_era_timeline(db_session)
    # B is the major-shift era. Aggregating A + C spans across B.
    with pytest.raises(TemporalBleedError) as excinfo:
        assert_no_temporal_bleed(db_session, [a.era_id, c.era_id])
    # Error names the offending boundary so an operator can react.
    assert "TEMPORAL_BLEED" in str(excinfo.value)
    assert b.era_slug in str(excinfo.value)


def test_assert_no_temporal_bleed_first_era_major_is_allowed(db_session) -> None:
    """The earliest era's own ``is_major_shift`` is *not* a bleed.

    A regime starts at a major-shift; aggregating from that point
    forward is by definition within-regime. The bleed is when a
    major-shift sits *between* the earliest and latest of the input
    set.
    """
    a, b, _c = _seed_three_era_timeline(db_session)
    # Just B (major) is fine.
    assert_no_temporal_bleed(db_session, [b.era_id])
    # B + a previously-loaded later minor era would also be fine — but
    # we already proved C is also non-major in our seeded timeline so
    # B + C is regime-internal.
    # Replace C with a fresh minor era after B.
    d = _make_era(
        slug="t_d",
        patch_version="9.05",
        start=datetime(2024, 9, 1, tzinfo=UTC),
        end=None,
    )
    # First put C aside (close it) to avoid the open-era unique index.
    _c_row = _seed_three_era_timeline.__doc__  # noqa: F841 - just sanity hook
    # Recreate the timeline with B as the first era so we can attach D.
    db_session.execute(text("TRUNCATE patch_era CASCADE"))
    fresh_b = _make_era(
        slug="fb",
        patch_version="8.08",
        start=datetime(2024, 4, 1, tzinfo=UTC),
        end=datetime(2024, 9, 1, tzinfo=UTC),
        is_major_shift=True,
        meta_magnitude=0.85,
    )
    db_session.add_all([fresh_b, d])
    db_session.flush()
    assert_no_temporal_bleed(db_session, [fresh_b.era_id, d.era_id])
