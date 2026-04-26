"""Daily-digest aggregation + CLI smoke-test."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from esports_sim.budget.caps import BudgetCaps
from esports_sim.budget.cli import main as cli_main
from esports_sim.budget.ledger import Ledger
from esports_sim.budget.report import format_digest, summarize_window


@pytest.fixture
def ledger(tmp_path: Path) -> Ledger:
    return Ledger(db_path=tmp_path / "budget.sqlite")


def _seed(ledger: Ledger, *, now: datetime) -> None:
    """Seed a few canonical rows over the last 24h."""
    ledger.record(
        endpoint="messages.create",
        model="claude-opus-4-7",
        purpose="patch_intent",
        phase="post",
        usd_cost=2.10,
        timestamp=now - timedelta(hours=2),
    )
    ledger.record(
        endpoint="messages.create",
        model="claude-haiku-4-5",
        purpose="personality",
        phase="post",
        usd_cost=0.73,
        timestamp=now - timedelta(hours=8),
    )
    # A blocked row — should appear in the count, not the spend.
    ledger.record(
        endpoint="messages.create",
        model="claude-opus-4-7",
        purpose="patch_intent",
        phase="blocked",
        usd_cost=0.0,
        notes="blocked scope=purpose",
        timestamp=now - timedelta(minutes=30),
    )


def test_summary_aggregates_by_purpose_and_model(ledger: Ledger) -> None:
    now = datetime(2026, 4, 26, 12, 0, tzinfo=UTC)
    _seed(ledger, now=now)

    s = summarize_window(
        ledger,
        caps=BudgetCaps(
            weekly_hard_cap_usd=30.0,
            purpose_caps_usd={"patch_intent": 3.0, "personality": 15.0},
        ),
        now=now,
    )

    assert s.weekly_spend_usd == pytest.approx(2.83)
    assert s.percent_of_weekly_cap == pytest.approx(2.83 / 30.0 * 100)
    assert s.blocked_count == 1

    by = {p.purpose: p for p in s.by_purpose}
    assert by["patch_intent"].spend_usd == pytest.approx(2.10)
    assert by["patch_intent"].cap_usd == 3.0
    assert by["patch_intent"].percent_of_cap == pytest.approx(2.10 / 3.0 * 100)
    assert by["patch_intent"].blocked == 1
    assert by["personality"].spend_usd == pytest.approx(0.73)

    # by_model dict groups by model name.
    assert s.by_model["claude-opus-4-7"] == pytest.approx(2.10)
    assert s.by_model["claude-haiku-4-5"] == pytest.approx(0.73)


def test_format_digest_includes_key_lines(ledger: Ledger) -> None:
    now = datetime(2026, 4, 26, 12, 0, tzinfo=UTC)
    _seed(ledger, now=now)
    s = summarize_window(
        ledger,
        caps=BudgetCaps(weekly_hard_cap_usd=30.0),
        now=now,
    )
    text = format_digest(s)
    assert "Last 7 days:" in text
    assert "$2.83" in text
    assert "Blocked calls: 1" in text
    assert "patch_intent" in text
    assert "claude-opus-4-7" in text


def test_cli_report_subcommand_prints_summary(
    ledger: Ledger,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """`nexus budget report --db /path/to/db` prints the digest and exits 0."""
    now = datetime.now(UTC)
    _seed(ledger, now=now)
    rc = cli_main(["budget", "report", "--db", str(ledger.db_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Last 7 days:" in out
    assert "patch_intent" in out


def test_cli_help_when_no_subcommand(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """`nexus` (no subcommand) errors out with usage — argparse default."""
    with pytest.raises(SystemExit):
        cli_main([])
