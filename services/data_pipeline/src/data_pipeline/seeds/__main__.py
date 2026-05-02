"""``python -m data_pipeline.seeds <command>`` operator entry point.

Two seeds wired here:

* ``vlr`` — bulk match history from a community-scraped VLR.gg CSV
  (BUF-8 v2). The replacement for the Liquipedia seed; reads a path,
  populates canonical TEAM + TOURNAMENT entities plus every match
  and map row.
* ``patch-eras`` — Valorant patch-era timeline (BUF-13).

Reads ``DATABASE_URL`` from the environment and commits at the end of
the run; on failure rolls back and re-raises so a partial seed never
lands.
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

from data_pipeline.seeds.patch_eras import seed_patch_eras
from data_pipeline.seeds.vlr import (
    DEFAULT_SEEDS_DIR,
    seed_from_vlr_csv,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m data_pipeline.seeds",
        description="One-shot seed scripts for the canonical entity store.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    vlr = sub.add_parser(
        "vlr",
        help="Bootstrap canonical entities + match history from a VLR.gg CSV (BUF-8 v2).",
    )
    vlr.add_argument(
        "csv_path",
        type=Path,
        help="Path to the VLR.gg map-level CSV (one row per map).",
    )
    vlr.add_argument(
        "--seeds-dir",
        type=Path,
        default=DEFAULT_SEEDS_DIR,
        help=f"Where to write the manifest (default: {DEFAULT_SEEDS_DIR}).",
    )
    vlr.add_argument(
        "--no-manifest",
        action="store_true",
        help="Skip writing the manifest file (still printed on stdout).",
    )

    eras = sub.add_parser(
        "patch-eras",
        help="Seed the patch_era table with the historical Valorant timeline (BUF-13).",
    )
    eras.add_argument(
        "--seeds-dir",
        type=Path,
        default=DEFAULT_SEEDS_DIR,
        help=f"Where to write the manifest (default: {DEFAULT_SEEDS_DIR}).",
    )
    eras.add_argument(
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

    engine = create_engine(_resolve_database_url(), future=True)
    try:
        with Session(engine) as session, session.begin():
            if args.command == "vlr":
                manifest = seed_from_vlr_csv(
                    session,
                    csv_path=args.csv_path,
                    seeds_dir=args.seeds_dir,
                    write_manifest=not args.no_manifest,
                )
                summary = (
                    f"vlr seed complete: "
                    f"teams_created={manifest.teams.created} "
                    f"tournaments_created={manifest.tournaments.created} "
                    f"matches_inserted={manifest.matches.matches_inserted} "
                    f"maps_inserted={manifest.matches.maps_inserted} "
                    f"date={manifest.seed_date}"
                )
            elif args.command == "patch-eras":
                era_manifest = seed_patch_eras(
                    session,
                    seeds_dir=args.seeds_dir,
                    write_manifest=not args.no_manifest,
                )
                summary = (
                    f"patch_eras seed complete: planned={era_manifest.counters.planned} "
                    f"inserted={era_manifest.counters.inserted} "
                    f"existing={era_manifest.counters.existing} "
                    f"open_era={era_manifest.open_era_slug}"
                )
            else:  # pragma: no cover - argparse rejects others
                return 2
    finally:
        engine.dispose()

    print(summary)  # noqa: T201 - operator-facing CLI
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
