"""BUF-13 patch_era seed tests.

The seed is a one-shot historical backfill. Two interesting properties
to nail down:

* **Coverage:** the seeded timeline tiles [first.start, +infinity) so
  any historical Valorant timestamp resolves to an era. The BUF-13
  acceptance "100% of historical matches successfully assigned" rests
  on this — we walk a year-by-year sweep at known patch boundaries
  and assert :func:`assign_era` returns a non-null era_id for each.
* **Idempotency:** the second invocation is a no-op. The manifest
  records ``existing`` vs. ``inserted`` separately so an operator can
  prove this without a manual diff.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from data_pipeline.seeds.patch_eras import (
    _VALORANT_ERAS,
    PatchEraSeedManifest,
    _build_planned_rows,
    seed_patch_eras,
)
from esports_sim.db.models import PatchEra
from esports_sim.eras import assign_era
from sqlalchemy import select

# --- pure-function unit tests (no DB) -------------------------------------


def test_planned_rows_form_contiguous_timeline() -> None:
    """Each spec's ``end_date`` is the next spec's ``start_date``.

    No gap, no overlap — same invariant the patch_era EXCLUDE
    constraint enforces at the DB level.
    """
    rows = _build_planned_rows(_VALORANT_ERAS)
    assert rows, "expected a non-empty Valorant patch-era timeline"

    # Earliest entry covers Valorant's launch (2020).
    assert rows[0]["start_date"].year == 2020

    # Each (i)th row's end_date equals the (i+1)th row's start_date.
    for prev, nxt in zip(rows, rows[1:], strict=False):
        assert prev["end_date"] is not None
        assert prev["end_date"] == nxt["start_date"]

    # Last row is the open era.
    assert rows[-1]["end_date"] is None


def test_planned_rows_has_at_least_one_major_shift() -> None:
    """Major-shift markers are what the BUF-13 guard keys on."""
    rows = _build_planned_rows(_VALORANT_ERAS)
    assert any(r["is_major_shift"] for r in rows)


def test_planned_rows_meta_magnitude_in_range() -> None:
    rows = _build_planned_rows(_VALORANT_ERAS)
    for r in rows:
        assert 0.0 <= r["meta_magnitude"] <= 1.0


def test_planned_rows_unique_slugs_and_strictly_increasing_starts() -> None:
    rows = _build_planned_rows(_VALORANT_ERAS)
    slugs = [r["era_slug"] for r in rows]
    assert len(slugs) == len(set(slugs)), "era_slug collision in seed dataset"
    starts = [r["start_date"] for r in rows]
    assert starts == sorted(starts)
    # Strictly increasing — no equal-pair dates.
    for prev, nxt in zip(starts, starts[1:], strict=False):
        assert prev < nxt


def test_build_planned_rows_rejects_unsorted_input() -> None:
    """A bug in the spec list (manual re-ordering) is caught early."""
    from datetime import date as ddate

    from data_pipeline.seeds.patch_eras import _EraSpec

    bad = [
        _EraSpec("a", "1.0", ddate(2024, 1, 1), True, 0.5),
        _EraSpec("b", "1.0", ddate(2023, 1, 1), False, 0.3),  # earlier than a
    ]
    with pytest.raises(ValueError):
        _build_planned_rows(bad)


# --- integration tests (require Postgres) ---------------------------------


pytestmark_integration = pytest.mark.integration


@pytest.mark.integration
def test_seed_inserts_all_planned_rows_first_run(db_session, tmp_path: Path) -> None:
    """First-run acceptance: all planned rows land, no existing rows."""
    manifest = seed_patch_eras(
        db_session,
        seeds_dir=tmp_path,
        today=datetime(2026, 5, 2, tzinfo=UTC).date(),
    )
    expected = len(_VALORANT_ERAS)
    assert manifest.counters.planned == expected
    assert manifest.counters.inserted == expected
    assert manifest.counters.existing == 0

    # One open era after the seed.
    open_rows = (
        db_session.execute(select(PatchEra).where(PatchEra.end_date.is_(None))).scalars().all()
    )
    assert len(open_rows) == 1
    assert open_rows[0].era_slug == manifest.open_era_slug


@pytest.mark.integration
def test_seed_is_idempotent_second_run_inserts_zero(db_session, tmp_path: Path) -> None:
    """Second-run acceptance: zero new rows, all marked ``existing``."""
    seed_patch_eras(db_session, seeds_dir=tmp_path, write_manifest=False)
    manifest = seed_patch_eras(db_session, seeds_dir=tmp_path, write_manifest=False)
    assert manifest.counters.inserted == 0
    assert manifest.counters.existing == len(_VALORANT_ERAS)


@pytest.mark.integration
def test_seed_writes_manifest_file(db_session, tmp_path: Path) -> None:
    """Manifest file lives at ``{seeds_dir}/patch_eras_seed_{date}.json``."""
    today = datetime(2026, 5, 2, tzinfo=UTC).date()
    manifest = seed_patch_eras(db_session, seeds_dir=tmp_path, today=today)
    target = tmp_path / f"patch_eras_seed_{today.isoformat()}.json"
    assert target.exists()
    payload = json.loads(target.read_text())
    assert payload["seed_date"] == today.isoformat()
    assert payload["counters"]["inserted"] == manifest.counters.inserted


@pytest.mark.integration
def test_seed_no_manifest_skips_file(db_session, tmp_path: Path) -> None:
    """``write_manifest=False`` is honoured — no file lands."""
    seed_patch_eras(db_session, seeds_dir=tmp_path, write_manifest=False)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.integration
def test_seeded_timeline_assigns_every_year_2020_to_present(db_session) -> None:
    """Acceptance bullet: 100% of historical matches resolve to an era.

    Walks a year-by-year cadence from June 2020 (Valorant launch) to
    several years past the seed's last entry. Every probe must
    resolve.
    """
    seed_patch_eras(db_session, write_manifest=False)
    probe = datetime(2020, 6, 2, tzinfo=UTC)
    end = datetime(2030, 1, 1, tzinfo=UTC)
    while probe < end:
        era_id = assign_era(db_session, probe)
        assert era_id is not None, f"no era covers {probe.isoformat()}"
        probe += timedelta(days=30)


@pytest.mark.integration
def test_manifest_to_json_roundtrips() -> None:
    """The manifest dataclass serialises cleanly for the on-disk file."""
    m = PatchEraSeedManifest(
        seed_date="2026-05-02",
        started_at="2026-05-02T00:00:00+00:00",
        finished_at="2026-05-02T00:00:01+00:00",
    )
    payload = m.to_json()
    # asdict flattens nested counters; json must round-trip.
    raw = json.dumps(payload)
    decoded = json.loads(raw)
    assert decoded["seed_date"] == "2026-05-02"
