"""``nexus-transcribe`` CLI — drive one transcription pass (BUF-21).

The transcription worker lives in :mod:`data_pipeline`, which is a
service-layer package; the cross-cutting ``nexus`` dispatcher lives
in :mod:`esports_sim.cli` and importing the worker from there would
invert the dependency arrow (shared depending on a service). So
BUF-21 ships its own ``nexus-transcribe`` console script — same
naming convention as ``nexus``, registered alongside it via the
data-pipeline ``[project.scripts]`` block.

Usage::

    nexus-transcribe run [--limit N] [--include-existing]
                          [--sidecar-root PATH] [--model SIZE]
                          [--device cuda|cpu] [--db DATABASE_URL]

Defaults: production model (``large-v3``), CUDA fp16, Silero VAD on,
sidecar disabled. Operators on a laptop pass ``--device=cpu --model=medium``
to get a slow-but-runnable pass.
"""

from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from data_pipeline.transcribe.engine import (
    DEFAULT_MODEL_SIZE,
    FasterWhisperEngine,
    TranscriptionEngine,
)
from data_pipeline.transcribe.worker import transcribe_pending


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="nexus-transcribe",
        description=(
            "Local Whisper transcription worker (BUF-21). Pulls media_record "
            "rows that don't have a transcript yet and runs faster-whisper "
            "against them on the local GPU."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Drive one transcription pass.")
    run.add_argument(
        "--db",
        default=None,
        help="Database URL (default: $DATABASE_URL).",
    )
    run.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap on rows processed this pass (default: all pending).",
    )
    run.add_argument(
        "--include-existing",
        action="store_true",
        help=(
            "Re-transcribe rows that already have a transcript. Used after "
            "rotating to a newer model; default is to skip them."
        ),
    )
    run.add_argument(
        "--sidecar-root",
        type=Path,
        default=None,
        help=(
            "Directory under which to write per-media transcript.json files. "
            "Default: no sidecars (database row only)."
        ),
    )
    run.add_argument(
        "--model",
        default=DEFAULT_MODEL_SIZE,
        help=f"Whisper model size (default: {DEFAULT_MODEL_SIZE}).",
    )
    run.add_argument(
        "--device",
        default="cuda",
        choices=("cuda", "cpu"),
        help="Compute device (default: cuda).",
    )
    run.add_argument(
        "--compute-type",
        default=None,
        help=(
            "CTranslate2 compute type (default: float16 on cuda, int8 on cpu). "
            "See faster-whisper docs for the valid set."
        ),
    )
    run.add_argument(
        "--no-vad",
        dest="vad",
        action="store_false",
        default=True,
        help="Disable Silero VAD (silence-skip). Use for already-trimmed input.",
    )
    run.set_defaults(func=_cmd_run)

    return parser


def _resolve_db_url(arg_url: str | None) -> str:
    """Pick the database URL: CLI arg wins, else $DATABASE_URL.

    Coerces ``postgresql://`` and ``postgresql+asyncpg://`` into the
    sync ``postgresql+psycopg://`` form the worker expects — same
    coercion the test conftest does for ``TEST_DATABASE_URL`` so an
    operator pasting the URL from .env doesn't have to remember the
    driver suffix.
    """
    url = arg_url or os.getenv("DATABASE_URL")
    if not url:
        raise SystemExit(
            "error: pass --db or set $DATABASE_URL — the transcription worker "
            "needs a Postgres connection string."
        )
    if url.startswith("postgresql+asyncpg://"):
        return url.replace("postgresql+asyncpg://", "postgresql+psycopg://", 1)
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url


def _build_engine(args: argparse.Namespace) -> TranscriptionEngine:
    """Construct the production engine from CLI args.

    Picks a sensible default ``compute_type`` per device when the
    operator didn't override it: ``float16`` on cuda (the spec'd
    setup, hits the throughput target), ``int8`` on cpu (faster-
    whisper's recommended cpu-friendly precision).
    """
    compute_type = args.compute_type
    if compute_type is None:
        compute_type = "float16" if args.device == "cuda" else "int8"
    return FasterWhisperEngine(
        model_size=args.model,
        device=args.device,
        compute_type=compute_type,
        vad_filter=args.vad,
    )


def _cmd_run(args: argparse.Namespace) -> int:
    db_url = _resolve_db_url(args.db)
    engine_db = create_engine(db_url, future=True)
    engine = _build_engine(args)

    # ``session.begin()`` wraps the whole pass in one transaction.
    # If you'd rather commit per-row (a long batch where a crash
    # mid-way should keep the rows that finished), call
    # ``transcribe_pending`` directly from your own loop and commit
    # between yields.
    with Session(engine_db) as session, session.begin():
        stats = transcribe_pending(
            session=session,
            engine=engine,
            sidecar_root=args.sidecar_root,
            limit=args.limit,
            include_existing=args.include_existing,
        )

    realtime = stats.audio_seconds / stats.wallclock_seconds if stats.wallclock_seconds else 0.0
    print(
        f"transcribed={stats.transcribed} "
        f"selected={stats.selected} "
        f"file_missing={stats.file_missing} "
        f"engine_errors={stats.engine_errors} "
        f"audio_s={stats.audio_seconds:.1f} "
        f"wallclock_s={stats.wallclock_seconds:.1f} "
        f"realtime={realtime:.1f}x"
    )
    return 0 if stats.engine_errors == 0 else 1


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
