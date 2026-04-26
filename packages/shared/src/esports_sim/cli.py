"""Top-level ``nexus`` CLI dispatcher.

Each subcommand module (currently :mod:`esports_sim.budget.cli` and
:mod:`esports_sim.registry.cli`) registers its own subparser via
``add_subparser(parsers)``. Adding a new subcommand is one import + one
call here — no central enum, no magic strings.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from esports_sim.budget import cli as budget_cli
from esports_sim.registry import cli as registry_cli


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nexus",
        description="Nexus operator CLI.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    budget_cli.add_subparser(sub)
    registry_cli.add_subparser(sub)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    func = getattr(args, "func", None)
    if func is None:  # pragma: no cover - argparse ensures a subcommand
        parser.print_help()
        return 2
    return int(func(args))


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
