"""``nexus run`` subcommand wiring.

Subcommands:

* ``register --kind=K --config=PATH [--data=PATH ...] [--notes=...]``
  → prints the (possibly idempotent) ``run_id`` on stdout
* ``ls [--kind=K] [--status=S]`` → tabular listing of registered runs
* ``show <run_id>`` → full row + resolved paths
* ``finalize <run_id> --status=completed|failed [--notes=...]`` → mark terminal

The dispatcher lives at :mod:`esports_sim.cli`; this module exposes
``add_subparser`` + ``run`` so the dispatcher stays thin.
"""

from __future__ import annotations

import argparse
from datetime import timedelta

from esports_sim.registry.db import Registry, RunStatus
from esports_sim.registry.errors import RegistryError, RunNotFoundError


def add_subparser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parent = subparsers.add_parser(
        "run",
        help="Experiment registry (BUF-69).",
        description=(
            "Register, list, and finalise runs. The registry is the "
            "single source of truth for run_id ↔ artifacts; downstream "
            "code resolves paths through Registry.get(), never by "
            "hardcoding runs/{run_id}/..."
        ),
    )
    parent.add_argument(
        "--db",
        default=None,
        help="Path to registry.db (default: $NEXUS_REGISTRY_DB or state/registry.db).",
    )
    parent.add_argument(
        "--runs-dir",
        default=None,
        help="Path to the runs/ artifact tree (default: $NEXUS_RUNS_DIR or runs/).",
    )

    run_sub = parent.add_subparsers(dest="run_command", required=True)

    register = run_sub.add_parser(
        "register",
        help="Register a new run; print the run_id on stdout.",
    )
    register.add_argument("--kind", required=True, help="Run kind (e.g. rl-train).")
    register.add_argument(
        "--config",
        required=True,
        help="Path to the config file driving this run.",
    )
    register.add_argument(
        "--data",
        action="append",
        default=[],
        help="Input data file/dir to fold into the data fingerprint. Repeatable.",
    )
    register.add_argument(
        "--data-fingerprint",
        default=None,
        help="Pre-computed fingerprint (mutually exclusive with --data).",
    )
    register.add_argument("--notes", default=None, help="Free-form annotation.")

    ls = run_sub.add_parser("ls", help="List registered runs (newest first).")
    ls.add_argument("--kind", default=None, help="Filter by kind.")
    ls.add_argument(
        "--status",
        default=None,
        choices=[s.value for s in RunStatus],
        help="Filter by status.",
    )

    show = run_sub.add_parser("show", help="Print full details for a run_id.")
    show.add_argument("run_id")

    finalize = run_sub.add_parser(
        "finalize",
        help="Mark a running run terminal.",
    )
    finalize.add_argument("run_id")
    finalize.add_argument(
        "--status",
        required=True,
        choices=[s.value for s in RunStatus if s is not RunStatus.RUNNING],
        help="Terminal status.",
    )
    finalize.add_argument("--notes", default=None, help="Free-form annotation to append.")

    parent.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:  # noqa: PLR0911 - one return per branch is fine
    registry = Registry(db_path=args.db, runs_dir=args.runs_dir)

    if args.run_command == "register":
        if args.data and args.data_fingerprint:
            print(
                "error: pass --data or --data-fingerprint, not both.",
                flush=True,
            )
            return 2
        try:
            run_id = registry.register(
                kind=args.kind,
                config_path=args.config,
                data_paths=args.data or None,
                data_fingerprint=args.data_fingerprint,
                notes=args.notes,
            )
        except (RegistryError, FileNotFoundError) as e:
            print(f"error: {e}", flush=True)
            return 1
        print(run_id)
        return 0

    if args.run_command == "ls":
        rows = registry.list_runs(kind=args.kind, status=args.status)
        if not rows:
            return 0
        # Tabular output: run_id | kind | status | started | duration | notes
        lines = [
            f"{'RUN_ID':<32} {'KIND':<20} {'STATUS':<10} {'STARTED':<20} {'DUR':>10}  NOTES",
        ]
        for r in rows:
            duration = r.duration_seconds()
            dur_str = _format_duration(timedelta(seconds=duration)) if duration is not None else "-"
            started_str = r.started_at.strftime("%Y-%m-%d %H:%M:%S")
            notes = (r.notes or "").replace("\n", " ")[:60]
            lines.append(
                f"{r.run_id:<32} {r.kind:<20} {r.status.value:<10} "
                f"{started_str:<20} {dur_str:>10}  {notes}"
            )
        print("\n".join(lines))
        return 0

    if args.run_command == "show":
        try:
            r = registry.get(args.run_id)
        except RunNotFoundError as e:
            print(f"error: {e}", flush=True)
            return 1
        print(f"run_id              : {r.run_id}")
        print(f"kind                : {r.kind}")
        print(f"status              : {r.status.value}")
        print(f"started_at          : {r.started_at.isoformat()}")
        if r.finished_at is not None:
            print(f"finished_at         : {r.finished_at.isoformat()}")
            dur_str = _format_duration(timedelta(seconds=r.duration_seconds() or 0.0))
            print(f"duration            : {dur_str}")
        print(f"git_sha             : {r.git_sha or '-'}")
        print(f"config_snapshot     : {r.config_snapshot}")
        print(f"config_hash         : {r.config_hash}")
        print(f"data_fingerprint    : {r.data_fingerprint or '-'}")
        print(f"run_dir             : {r.run_dir}")
        if r.notes:
            print(f"notes               : {r.notes}")
        return 0

    if args.run_command == "finalize":
        try:
            registry.finalize(args.run_id, status=args.status, notes=args.notes)
        except (RegistryError, RunNotFoundError) as e:
            print(f"error: {e}", flush=True)
            return 1
        return 0

    return 2


def _format_duration(td: timedelta) -> str:
    """Compact ``HH:MM:SS`` (or ``Dd HH:MM:SS`` past 24h) for ls output."""
    total = int(td.total_seconds())
    days, rem = divmod(total, 86400)
    hours, rem = divmod(rem, 3600)
    minutes, seconds = divmod(rem, 60)
    if days:
        return f"{days}d {hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
