"""``nexus budget`` subcommand wiring.

The top-level dispatcher lives at :mod:`esports_sim.cli`; this module
exposes ``add_subparser(parsers)`` + ``run(args)`` so each subcommand
can register itself without the dispatcher growing import-coupling to
every command.
"""

from __future__ import annotations

import argparse
from datetime import timedelta

from esports_sim.budget.caps import BudgetCaps
from esports_sim.budget.ledger import Ledger
from esports_sim.budget.report import format_digest, summarize_window


def add_subparser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the ``budget`` subcommand on the top-level ``nexus`` parser."""
    budget = subparsers.add_parser(
        "budget",
        help="API budget tooling (BUF-22).",
        description="Inspect the Claude API budget ledger.",
    )
    budget_sub = budget.add_subparsers(dest="budget_command", required=True)

    report = budget_sub.add_parser(
        "report",
        help="Print the daily digest of Claude API spend.",
    )
    report.add_argument(
        "--days",
        type=int,
        default=7,
        help="Window size in days (default: 7).",
    )
    report.add_argument(
        "--db",
        default=None,
        help="Path to the SQLite ledger (default: $NEXUS_BUDGET_DB or .nexus/budget.sqlite).",
    )

    budget.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    """Dispatch a parsed ``nexus budget ...`` invocation."""
    if args.budget_command == "report":
        ledger = Ledger(db_path=args.db) if args.db else Ledger()
        summary = summarize_window(
            ledger,
            caps=BudgetCaps.from_env(),
            window=timedelta(days=args.days),
        )
        print(format_digest(summary))
        return 0
    return 2


# Backwards-compatible standalone entry point for the early dev period
# when `nexus = "esports_sim.budget.cli:main"` was the console script.
def main(argv: list[str] | None = None) -> int:  # pragma: no cover - shim
    from esports_sim.cli import main as top_main

    return top_main(argv)
