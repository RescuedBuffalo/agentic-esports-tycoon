"""``nexus`` CLI — currently exposes ``nexus budget``.

Kept stdlib-only (``argparse``) so the package doesn't pull in click/typer
just for one subcommand. As more verbs land, split this file up.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from datetime import timedelta

from esports_sim.budget.caps import BudgetCaps
from esports_sim.budget.ledger import Ledger
from esports_sim.budget.report import format_digest, summarize_window


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nexus",
        description="Nexus operator CLI.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    budget = sub.add_parser(
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

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "budget" and args.budget_command == "report":
        ledger = Ledger(db_path=args.db) if args.db else Ledger()
        summary = summarize_window(
            ledger,
            caps=BudgetCaps.from_env(),
            window=timedelta(days=args.days),
        )
        print(format_digest(summary))
        return 0

    parser.print_help()
    return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
