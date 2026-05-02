"""``esports_sim.cli`` — the top-level ``nexus`` dispatcher.

Each subcommand has its own test file (``test_budget_cli`` /
``test_registry_cli``) but the dispatcher itself — wiring up the
``budget`` and ``run`` subparsers, exit codes, ``--help``, and
unknown-command handling — has no other coverage. A regression that
forgot to register a subparser, or that flipped the ``required=True``
bit on the top-level parser, would silently break the CLI without
failing any of the per-subcommand tests.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from esports_sim.cli import _build_parser
from esports_sim.cli import main as cli_main


def test_build_parser_registers_known_subcommands() -> None:
    """Both subcommands must be reachable from the top-level parser.

    Read the choices off the ``command`` subparser action — independent
    of help-text parsing.
    """
    parser = _build_parser()
    sub_actions = [
        a
        for a in parser._actions  # noqa: SLF001 - argparse public surface is thin
        if isinstance(a, type(parser._subparsers._group_actions[0]))  # type: ignore[union-attr]
    ]
    assert sub_actions, "expected a subparsers action on the top-level parser"
    choices = sub_actions[0].choices  # type: ignore[attr-defined]
    assert "budget" in choices
    assert "run" in choices


def test_main_with_no_argv_exits_two() -> None:
    """Empty argv -> argparse "the following arguments are required" -> exit 2.

    The dispatcher's subparser is ``required=True``; argparse converts
    that into a ``SystemExit(2)``.
    """
    with pytest.raises(SystemExit) as exc_info:
        cli_main([])
    assert exc_info.value.code == 2


def test_main_with_unknown_subcommand_exits_two(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        cli_main(["definitely-not-a-subcommand"])
    assert exc_info.value.code == 2
    err = capsys.readouterr().err
    assert "invalid choice" in err.lower()


def test_main_routes_budget_subcommand(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """``nexus budget report`` reaches the budget runner and exits 0."""
    db = tmp_path / "budget.sqlite"
    rc = cli_main(["budget", "report", "--db", str(db)])
    assert rc == 0
    # Some output was rendered — we don't pin the exact format here
    # (that's `test_budget_report.py`'s job).
    assert capsys.readouterr().out.strip() != ""


def test_main_routes_run_subcommand(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """``nexus run register`` reaches the registry runner and exits 0.

    Mirrors the registry CLI smoke test but lives here to assert
    *dispatch*, not registry behaviour.
    """
    cfg = tmp_path / "configs" / "graph" / "era_7.09.yaml"
    cfg.parent.mkdir(parents=True)
    cfg.write_text("kind: graph-snapshot\nera: 7.09\n", encoding="utf-8")

    rc = cli_main(
        [
            "run",
            "--db",
            str(tmp_path / "registry.db"),
            "--runs-dir",
            str(tmp_path / "runs"),
            "register",
            "--kind=graph-snapshot",
            f"--config={cfg}",
        ]
    )
    assert rc == 0
    assert capsys.readouterr().out.strip().startswith("graph-snapshot-")


def test_main_help_lists_both_subcommands(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``nexus --help`` advertises both ``budget`` and ``run`` so operators discover them."""
    with pytest.raises(SystemExit) as exc_info:
        cli_main(["--help"])
    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    assert "budget" in out
    assert "run" in out
