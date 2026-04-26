"""SQLite ledger reads + writes."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from esports_sim.budget.ledger import Ledger


@pytest.fixture
def ledger(tmp_path: Path) -> Ledger:
    return Ledger(db_path=tmp_path / "budget.sqlite")


def test_record_and_read_back(ledger: Ledger) -> None:
    row_id = ledger.record(
        endpoint="messages.create",
        model="claude-opus-4-7",
        purpose="patch_intent",
        phase="post",
        input_tokens=100,
        output_tokens=50,
        usd_cost=0.0375,
        request_id="req_abc",
    )
    assert row_id == 1
    rows = ledger.all_entries()
    assert len(rows) == 1
    e = rows[0]
    assert e.purpose == "patch_intent"
    assert e.usd_cost == pytest.approx(0.0375)
    assert e.request_id == "req_abc"
    assert e.phase == "post"


def test_update_post_overwrites_pre_row(ledger: Ledger) -> None:
    """Pre-flight estimate is replaced by post-flight actual on update_post."""
    row_id = ledger.record(
        endpoint="messages.create",
        model="claude-opus-4-7",
        purpose="patch_intent",
        phase="pre",
        usd_cost=0.10,  # estimate
    )
    ledger.update_post(
        row_id,
        input_tokens=500,
        output_tokens=200,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
        usd_cost=0.0075,  # actual, lower than estimate
        request_id="req_xyz",
    )
    e = ledger.all_entries()[0]
    assert e.phase == "post"
    assert e.usd_cost == pytest.approx(0.0075)
    assert e.input_tokens == 500
    assert e.request_id == "req_xyz"


def test_weekly_spend_excludes_old_rows(ledger: Ledger) -> None:
    now = datetime(2026, 4, 26, 12, 0, tzinfo=UTC)
    # Old row (8 days ago) — outside the rolling 7-day window.
    ledger.record(
        endpoint="messages.create",
        model="claude-opus-4-7",
        purpose="x",
        phase="post",
        usd_cost=10.0,
        timestamp=now - timedelta(days=8),
    )
    # Recent row.
    ledger.record(
        endpoint="messages.create",
        model="claude-opus-4-7",
        purpose="x",
        phase="post",
        usd_cost=2.5,
        timestamp=now - timedelta(days=1),
    )
    spend = ledger.weekly_spend(now=now)
    assert spend == pytest.approx(2.5)


def test_weekly_spend_filtered_by_purpose(ledger: Ledger) -> None:
    now = datetime(2026, 4, 26, 12, 0, tzinfo=UTC)
    ledger.record(endpoint="m", model="x", purpose="a", phase="post", usd_cost=3.0, timestamp=now)
    ledger.record(endpoint="m", model="x", purpose="b", phase="post", usd_cost=1.0, timestamp=now)
    assert ledger.weekly_spend(purpose="a", now=now) == pytest.approx(3.0)
    assert ledger.weekly_spend(purpose="b", now=now) == pytest.approx(1.0)
    assert ledger.weekly_spend(now=now) == pytest.approx(4.0)


def test_blocked_rows_have_zero_cost_when_recorded_via_update(ledger: Ledger) -> None:
    """The governor records blocked rows with ``usd_cost=0`` so the rolling
    spend doesn't double-count an attempt that never went out."""
    ledger.record(
        endpoint="m",
        model="x",
        purpose="p",
        phase="blocked",
        usd_cost=0.0,
        notes="blocked scope=weekly",
    )
    assert ledger.weekly_spend() == 0.0


def test_concurrent_open_does_not_corrupt_db(tmp_path: Path) -> None:
    """Two Ledger instances on the same file see each other's writes."""
    db = tmp_path / "shared.sqlite"
    a = Ledger(db_path=db)
    b = Ledger(db_path=db)
    a.record(endpoint="m", model="x", purpose="p", phase="post", usd_cost=1.0)
    # b reads it back via its own connection.
    assert len(b.all_entries()) == 1
    b.record(endpoint="m", model="x", purpose="p", phase="post", usd_cost=2.0)
    # Round-trip through a sees both rows.
    assert len(a.all_entries()) == 2
