"""``python -m data_pipeline.seeds liquipedia`` operator entry point.

The ticket spec is "run ``seed_from_liquipedia()`` once" — this module
is the one-line invocation an operator types after standing up a fresh
Postgres. Reads ``DATABASE_URL`` from the environment and commits at
the end of the run; on failure rolls back and re-raises so a partial
seed never lands.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from data_pipeline.connectors.liquipedia import DEFAULT_BASE_URL
from data_pipeline.seeds.liquipedia import (
    DEFAULT_SEEDS_DIR,
    seed_from_liquipedia,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m data_pipeline.seeds",
        description="One-shot seed scripts for the canonical entity store.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    liq = sub.add_parser(
        "liquipedia",
        help="Bootstrap canonical entities from Liquipedia (BUF-8).",
    )
    liq.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"Liquipedia REST base URL (default: {DEFAULT_BASE_URL}).",
    )
    liq.add_argument(
        "--seeds-dir",
        type=Path,
        default=DEFAULT_SEEDS_DIR,
        help=f"Where to write the manifest (default: {DEFAULT_SEEDS_DIR}).",
    )
    liq.add_argument(
        "--no-manifest",
        action="store_true",
        help="Skip writing the manifest file (still printed on stdout).",
    )
    return parser


def _resolve_database_url() -> str:
    url = os.getenv("DATABASE_URL")
    if not url:
        raise SystemExit("DATABASE_URL is not set; cannot run the seed without a Postgres target.")
    # alembic + the sync engine want psycopg explicitly. Mirroring the
    # conftest helper so an operator's URL works whether they exported
    # the bare ``postgresql://`` form or the asyncpg one.
    if url.startswith("postgresql+asyncpg://"):
        return url.replace("postgresql+asyncpg://", "postgresql+psycopg://", 1)
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    args = _build_parser().parse_args(argv)
    if args.command != "liquipedia":  # pragma: no cover - argparse rejects others
        return 2

    engine = create_engine(_resolve_database_url(), future=True)
    try:
        with Session(engine) as session, session.begin():
            manifest = seed_from_liquipedia(
                session,
                base_url=args.base_url,
                seeds_dir=args.seeds_dir,
                write_manifest=not args.no_manifest,
            )
    finally:
        engine.dispose()

    print(  # noqa: T201 - operator-facing CLI
        f"liquipedia seed complete: created={manifest.total_canonical_created} "
        f"date={manifest.seed_date}"
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
