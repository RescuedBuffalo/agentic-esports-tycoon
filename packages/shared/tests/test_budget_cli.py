"""``nexus budget`` CLI surface (BUF-22).

Exercises the dispatcher (:func:`esports_sim.cli.main`) with the
``budget`` subcommand so the argparse wiring, default values, and exit
codes stay locked down independently of the underlying ``Ledger`` /
``summarize_window`` plumbing (which has its own tests).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from esports_sim.cli import main as cli_main


def test_budget_report_prints_digest_with_default_window(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``nexus budget report --db <path>`` exits 0 and prints a digest.

    No ledger entries yet — the digest is the empty case, but the
    command must succeed and the formatted output must hit stdout.
    """
    db = tmp_path / "budget.sqlite"
    rc = cli_main(["budget", "report", "--db", str(db)])
    assert rc == 0
    out = capsys.readouterr().out
    # ``format_digest`` always renders a header row mentioning the
    # weekly cap; assert we got *something* rendered rather than an
    # empty string.
    assert out.strip() != ""


def test_budget_report_accepts_custom_window(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``--days`` is forwarded to ``summarize_window`` as a timedelta."""
    db = tmp_path / "budget.sqlite"
    rc = cli_main(["budget", "report", "--db", str(db), "--days", "14"])
    assert rc == 0
    assert capsys.readouterr().out.strip() != ""


def test_budget_report_creates_db_parent_directory(tmp_path: Path) -> None:
    """Passing ``--db`` to a non-existent dir is fine — Ledger mkdirs.

    Lets an operator point the CLI at a fresh path without manually
    creating the directory first.
    """
    db = tmp_path / "nested" / "dir" / "budget.sqlite"
    rc = cli_main(["budget", "report", "--db", str(db)])
    assert rc == 0
    assert db.exists()


def test_budget_subcommand_requires_inner_command(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``nexus budget`` without ``report`` must fail loud, not run silently.

    The subparser is configured ``required=True``; argparse will exit
    with code 2 and print a usage error to stderr.
    """
    with pytest.raises(SystemExit) as exc_info:
        cli_main(["budget"])
    assert exc_info.value.code == 2
    err = capsys.readouterr().err
    assert "budget" in err.lower()


def test_unknown_budget_subcommand_rejected_by_argparse(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A typo in the subcommand triggers argparse's invalid-choice exit."""
    with pytest.raises(SystemExit) as exc_info:
        cli_main(["budget", "nope"])
    # argparse exits 2 for an invalid-choice error.
    assert exc_info.value.code == 2


def test_budget_report_help_lists_known_flags(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``nexus budget report --help`` lists ``--days`` and ``--db``.

    A regression that drops a flag should fail this test before it
    reaches an operator's terminal.
    """
    with pytest.raises(SystemExit) as exc_info:
        cli_main(["budget", "report", "--help"])
    # ``--help`` exits 0.
    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    assert "--days" in out
    assert "--db" in out


def test_top_level_help_lists_budget_subcommand(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The dispatcher's help text mentions ``budget`` so operators discover it."""
    with pytest.raises(SystemExit) as exc_info:
        cli_main(["--help"])
    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    assert "budget" in out
