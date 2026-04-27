"""Experiment registry — SQLite-backed run index (BUF-69, ADR-006).

Every training run, graph snapshot build, PPV computation, and world-model
pretrain registers itself here at start, gets a deterministic ``run_id``,
and finalises (or crashes) with a status. Without that, a year-in claim
like *"world model v3 was 12% better than v2"* has no meaning — we can't
reproduce the regime v2 was trained under.

Storage: SQLite WAL, mirroring the budget ledger (BUF-22). One file at
``state/registry.db`` (overridable via ``NEXUS_REGISTRY_DB``); WAL mode
so digest CLIs and live training writers don't block each other.

Artifact layout: every byte a run produces lives under ``runs/{run_id}/``
(overridable via ``NEXUS_RUNS_DIR``). Downstream code asks the registry
for paths via :meth:`Registry.get` — nothing should hardcode the
``runs/...`` prefix.
"""

from __future__ import annotations

import enum
import hashlib
import os
import re
import secrets
import shutil
import sqlite3
import subprocess
from collections.abc import Iterable, Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from esports_sim.registry.errors import (
    InvalidKindError,
    RegistryError,
    RunNotFoundError,
)
from esports_sim.registry.fingerprint import compute_fingerprint, hash_file

# Defaults follow the issue: state/registry.db + runs/. Both relative to
# the process CWD so dev environments and CI both Just Work without env
# wiring; production deployments override via env.
_DEFAULT_DB_PATH = Path("state") / "registry.db"
_DEFAULT_RUNS_DIR = Path("runs")

# kind validation: filesystem- and CLI-safe. Lowercase letters, digits,
# and -/_/.; no path separators, no whitespace. Anchored to whole string.
_KIND_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")

# How many times we retry register() on a run_id PK collision. Each retry
# generates a new {YYMMDD-HHMMSS-6hex} suffix; with 24 bits of entropy
# per second, five retries makes a true collision storm vanishingly
# unlikely while keeping the failure mode bounded.
_MAX_RUN_ID_RETRIES = 5


_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    git_sha TEXT,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    -- Absolute (or CWD-relative) path to the run's artifact directory at
    -- *register time*. We persist it so RunRecord.run_dir is stable across
    -- registry openings — opening with a different ``runs_dir`` later
    -- doesn't silently relocate the artifacts of an existing row.
    run_dir TEXT NOT NULL DEFAULT '',
    config_snapshot_path TEXT NOT NULL,
    -- sha256 of the *original* config file's bytes. Together with kind
    -- and data_fingerprint this is the natural key for idempotent
    -- registration: re-registering the same triple returns the same run_id.
    config_hash TEXT NOT NULL,
    data_fingerprint TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL,
    notes TEXT,
    UNIQUE(kind, config_hash, data_fingerprint)
);
CREATE INDEX IF NOT EXISTS ix_runs_kind ON runs(kind);
CREATE INDEX IF NOT EXISTS ix_runs_status ON runs(status);
CREATE INDEX IF NOT EXISTS ix_runs_started_at ON runs(started_at);
"""


class RunStatus(enum.StrEnum):
    """Lifecycle states for a run.

    ``crashed`` is computed (a ``running`` row that's stale by some
    operator-defined window), not stored — see
    :meth:`Registry.list_runs`. Storing it would race with the run's own
    finalize call.
    """

    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True)
class RunRecord:
    """A row from ``runs``, with helpers for path resolution.

    Downstream code asks for paths via this object instead of building
    ``runs/{run_id}/...`` strings inline — that way a future move of the
    artifact root only touches the registry.

    ``run_dir`` is the *stored* path persisted at register time, not a
    value derived from whoever opens the registry now. Opening with a
    different ``runs_dir`` later doesn't relocate the artifacts of an
    existing row — and downstream code that resumes a registered run
    always reads/writes under the directory the run was created in.
    """

    run_id: str
    kind: str
    git_sha: str | None
    started_at: datetime
    finished_at: datetime | None
    run_dir: Path
    config_snapshot_path: str
    config_hash: str
    data_fingerprint: str
    status: RunStatus
    notes: str | None

    @property
    def config_snapshot(self) -> Path:
        """Path to the registered (verbatim) copy of the input config."""
        return Path(self.config_snapshot_path)

    def artifact_path(self, *parts: str) -> Path:
        """Build an artifact path inside this run's directory.

        Use for checkpoints, logs, plots — anything a run produces. The
        directory is created on demand by callers; the registry only
        guarantees ``run_dir`` itself exists.
        """
        return self.run_dir.joinpath(*parts)

    def duration_seconds(self) -> float | None:
        """Wall-clock duration of the run, or ``None`` if still running."""
        if self.finished_at is None:
            return None
        return (self.finished_at - self.started_at).total_seconds()


class Registry:
    """Thin wrapper around the SQLite registry + the runs/ artifact tree."""

    def __init__(
        self,
        *,
        db_path: str | Path | None = None,
        runs_dir: str | Path | None = None,
    ) -> None:
        raw_db = Path(db_path) if db_path is not None else _resolve_db_path()
        raw_runs_dir = Path(runs_dir) if runs_dir is not None else _resolve_runs_dir()

        # ``:memory:`` is a magic string (not a real path) — leave it alone so
        # ``sqlite3.connect`` treats it as an in-memory database.
        self._is_memory_db = str(raw_db) == ":memory:"
        self.db_path: Path = raw_db if self._is_memory_db else raw_db.resolve()

        # ``runs_dir`` is resolved to an absolute path so rows we INSERT later
        # carry stable on-disk locations. A relative ``runs/`` written from
        # one CWD and read from another would otherwise resolve to two
        # different filesystem locations — exactly the cross-machine /
        # cross-CWD bug the registry is supposed to prevent.
        if not self._is_memory_db:
            raw_runs_dir.mkdir(parents=True, exist_ok=True)
            raw_db.parent.mkdir(parents=True, exist_ok=True)
        self.runs_dir: Path = raw_runs_dir.resolve()

        # In-memory SQLite databases are *per-connection* — tables created on
        # one connection are invisible to a fresh connection. Hold one
        # persistent connection for the lifetime of the Registry so
        # ``_init_schema`` and subsequent reads/writes share state. File-
        # backed databases keep the open-per-call pattern so concurrent
        # writers don't block each other.
        self._memory_conn: sqlite3.Connection | None = None
        if self._is_memory_db:
            self._memory_conn = self._open_connection()

        self._init_schema()

    # ---- connection helpers ------------------------------------------------

    def _open_connection(self) -> sqlite3.Connection:
        """Open a fresh sqlite3 connection with the registry's PRAGMAs applied."""
        # ``check_same_thread=False`` lets the persistent ``:memory:``
        # connection be reused across pytest's setup/teardown threads. Has
        # no effect on file-backed connections that close per call.
        conn = sqlite3.connect(
            self.db_path,
            isolation_level=None,
            timeout=30.0,
            check_same_thread=False,
        )
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=30000")
        conn.row_factory = sqlite3.Row
        return conn

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        if self._memory_conn is not None:
            # Shared connection — do NOT close on exit, the Registry owns it.
            yield self._memory_conn
            return
        conn = self._open_connection()
        try:
            yield conn
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA_SQL)
            # Migrate older DBs that pre-date the run_dir column. SQLite
            # raises OperationalError("duplicate column name") when the
            # column already exists; suppressing that is the idiomatic
            # idempotent ADD COLUMN.
            with suppress(sqlite3.OperationalError):
                conn.execute("ALTER TABLE runs ADD COLUMN run_dir TEXT NOT NULL DEFAULT ''")

    # ---- public API --------------------------------------------------------

    def register(
        self,
        *,
        kind: str,
        config_path: str | Path,
        data_paths: Iterable[Path | str] | None = None,
        data_fingerprint: str | None = None,
        notes: str | None = None,
    ) -> str:
        """Idempotently register a new run; returns the ``run_id``.

        Order of operations matters — we hash the inputs *first* so the
        natural-key lookup can short-circuit before we touch the
        filesystem. Re-registering an identical (kind, config, data)
        triple is therefore cheap and side-effect-free.

        Parameters
        ----------
        kind:
            Stable identifier for the run shape — ``graph-snapshot``,
            ``rl-train``, ``world-model-pretrain``. Becomes part of the
            ``run_id`` and a CLI filter, so keep it terse and machine-safe.
        config_path:
            Path to the YAML/JSON/TOML config that drives this run. The
            registry hashes its bytes for idempotency and copies the file
            verbatim to ``runs/{run_id}/config.yaml``.
        data_paths:
            Optional input files/directories. If supplied, the registry
            walks them and computes a SHA-256 fingerprint over their
            contents (see :func:`compute_fingerprint`). Folded into the
            idempotency key so the same config against new data mints a
            new ``run_id``.
        data_fingerprint:
            Pre-computed fingerprint, for callers with a cheaper digest
            than file SHA (e.g. a DuckDB row-count over a 100GB table).
            Mutually exclusive with ``data_paths``.
        notes:
            Free-form annotation persisted on the row.
        """
        if not _KIND_RE.match(kind):
            raise InvalidKindError(kind)
        if data_paths is not None and data_fingerprint is not None:
            raise RegistryError(
                "Pass either data_paths or data_fingerprint, not both — "
                "the registry can't reconcile two truth sources."
            )

        config = Path(config_path)
        if not config.exists():
            raise FileNotFoundError(f"config file not found: {config}")

        config_hash = hash_file(config)
        if data_fingerprint is None:
            data_fingerprint = compute_fingerprint(data_paths or [])

        # Idempotency: same triple → same row. Look up *before* any
        # filesystem mutation so the no-op path is genuinely no-op.
        existing = self._lookup_existing(
            kind=kind, config_hash=config_hash, data_fingerprint=data_fingerprint
        )
        if existing is not None:
            return existing

        # Two distinct IntegrityError scenarios collapse onto the same
        # exception type and have to be disambiguated by inspecting the DB:
        #
        #   (a) **Natural-key race**. UNIQUE(kind, config_hash,
        #       data_fingerprint) fired because another writer registered
        #       the same triple between our pre-flight lookup and our
        #       insert. The run is theirs; return their run_id.
        #
        #   (b) **run_id PK collision**. _generate_run_id mints
        #       ``{kind}-{YYMMDD-HHMMSS}-{6hex}`` — 24 bits of suffix
        #       entropy per second. Two concurrent registrations of
        #       *different* triples can land on the same id at sub-second
        #       resolution. Retry with a fresh id.
        #
        # The shape of the recovery is "look up the natural key, if found
        # → (a), otherwise → (b)" inside a bounded retry loop.
        last_error: sqlite3.IntegrityError | None = None
        for _attempt in range(_MAX_RUN_ID_RETRIES):
            run_id = _generate_run_id(kind)
            run_dir = self.runs_dir / run_id
            try:
                # ``exist_ok=False`` raises FileExistsError if the dir was
                # created between our id generation and now — by either
                # this process (extremely unlikely) or a sibling that just
                # registered with the same id. Don't pre-check with
                # ``exists()``: that's TOCTOU and the race fires under
                # exactly the workload this loop exists to handle.
                run_dir.mkdir(parents=True, exist_ok=False)
            except FileExistsError:
                continue
            snapshot_path = run_dir / f"config{config.suffix or '.yaml'}"
            shutil.copy2(config, snapshot_path)

            started_at = _now_utc()
            git_sha = _current_git_sha()

            try:
                with self._connect() as conn:
                    conn.execute(
                        """
                        INSERT INTO runs (
                            run_id, kind, git_sha, started_at, finished_at,
                            run_dir, config_snapshot_path, config_hash,
                            data_fingerprint, status, notes
                        ) VALUES (?, ?, ?, ?, NULL, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            run_id,
                            kind,
                            git_sha,
                            started_at.isoformat(),
                            str(run_dir),
                            str(snapshot_path),
                            config_hash,
                            data_fingerprint,
                            RunStatus.RUNNING.value,
                            notes,
                        ),
                    )
            except sqlite3.IntegrityError as e:
                last_error = e
                # Tear down the half-built dir before deciding which
                # branch we're in — both branches discard our work.
                shutil.rmtree(run_dir, ignore_errors=True)
                existing = self._lookup_existing(
                    kind=kind,
                    config_hash=config_hash,
                    data_fingerprint=data_fingerprint,
                )
                if existing is not None:
                    # (a) Natural-key race — adopt the other writer's id.
                    return existing
                # (b) run_id PK collision — try again with a fresh id.
                continue
            return run_id

        raise RegistryError(  # pragma: no cover - 5 collisions in a row is essentially unreachable
            f"Could not register run after {_MAX_RUN_ID_RETRIES} attempts due to "
            f"run_id collisions. Check the system clock and entropy source. "
            f"Last error: {last_error}"
        )

    def finalize(
        self,
        run_id: str,
        *,
        status: RunStatus | str,
        notes: str | None = None,
    ) -> None:
        """Mark a ``running`` row terminal.

        ``status`` must be one of :class:`RunStatus` or its string value.
        Idempotent: finalising a row that's already terminal is allowed
        (the row's existing ``status``/``finished_at``/``notes`` win).
        """
        status_enum = RunStatus(status) if isinstance(status, str) else status
        if status_enum is RunStatus.RUNNING:
            raise RegistryError(
                "finalize() cannot transition a row back to RUNNING. " "Pass COMPLETED or FAILED."
            )

        finished_at = _now_utc().isoformat()
        with self._connect() as conn:
            cur = conn.execute(
                """
                UPDATE runs
                SET status = :status,
                    finished_at = :finished_at,
                    notes = CASE
                        WHEN :note IS NULL THEN notes
                        WHEN notes IS NULL OR notes = '' THEN :note
                        ELSE notes || '; ' || :note
                    END
                WHERE run_id = :rid AND status = :prev_status
                """,
                {
                    "status": status_enum.value,
                    "finished_at": finished_at,
                    "prev_status": RunStatus.RUNNING.value,
                    "rid": run_id,
                    "note": notes,
                },
            )
            if cur.rowcount == 0:
                # Either the row doesn't exist or it's already terminal.
                # Distinguish so callers get a useful error.
                row = conn.execute("SELECT 1 FROM runs WHERE run_id = ?", (run_id,)).fetchone()
                if row is None:
                    raise RunNotFoundError(run_id)
                # Already terminal — idempotent no-op. (Tests pin this.)

    def get(self, run_id: str) -> RunRecord:
        """Return the full row for ``run_id`` or raise :class:`RunNotFoundError`."""
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            raise RunNotFoundError(run_id)
        return self._row_to_record(row)

    def list_runs(
        self,
        *,
        kind: str | None = None,
        status: RunStatus | str | None = None,
    ) -> list[RunRecord]:
        """List runs, newest-first.

        Filters compose with AND. Pass ``status="running"`` to find
        candidates for crash detection.
        """
        sql = "SELECT * FROM runs"
        clauses: list[str] = []
        params: list[object] = []
        if kind is not None:
            clauses.append("kind = ?")
            params.append(kind)
        if status is not None:
            status_str = status.value if isinstance(status, RunStatus) else str(status)
            clauses.append("status = ?")
            params.append(status_str)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY started_at DESC"
        with self._connect() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._row_to_record(r) for r in rows]

    # ---- internals ---------------------------------------------------------

    def _lookup_existing(self, *, kind: str, config_hash: str, data_fingerprint: str) -> str | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT run_id FROM runs
                WHERE kind = ? AND config_hash = ? AND data_fingerprint = ?
                """,
                (kind, config_hash, data_fingerprint),
            ).fetchone()
        return None if row is None else str(row["run_id"])

    def _row_to_record(self, row: sqlite3.Row) -> RunRecord:
        # ``run_dir`` is the *stored* path. Older rows (registered before
        # the column existed) carry an empty string — fall back to the
        # caller's ``runs_dir / run_id`` so the lookup still resolves,
        # while accepting that the fallback inherits the original bug
        # for those legacy rows. Fresh registrations always populate the
        # column.
        stored_run_dir = row["run_dir"]
        run_dir = Path(stored_run_dir) if stored_run_dir else self.runs_dir / str(row["run_id"])
        return RunRecord(
            run_id=str(row["run_id"]),
            kind=str(row["kind"]),
            git_sha=row["git_sha"],
            started_at=datetime.fromisoformat(row["started_at"]),
            finished_at=(
                datetime.fromisoformat(row["finished_at"]) if row["finished_at"] else None
            ),
            run_dir=run_dir,
            config_snapshot_path=str(row["config_snapshot_path"]),
            config_hash=str(row["config_hash"]),
            data_fingerprint=str(row["data_fingerprint"]),
            status=RunStatus(row["status"]),
            notes=row["notes"],
        )


# ---- module-level helpers --------------------------------------------------


def _resolve_db_path() -> Path:
    raw = os.getenv("NEXUS_REGISTRY_DB")
    return Path(raw) if raw else _DEFAULT_DB_PATH


def _resolve_runs_dir() -> Path:
    raw = os.getenv("NEXUS_RUNS_DIR")
    return Path(raw) if raw else _DEFAULT_RUNS_DIR


def _now_utc() -> datetime:
    return datetime.now(UTC)


def _generate_run_id(kind: str) -> str:
    """``{kind}-{YYMMDD-HHMMSS}-{6hex}`` — sortable, human-readable, unique.

    24 bits of suffix entropy keeps the per-second collision probability
    at ~1/16M for two simultaneous starts of the same kind. The
    ``register`` retry loop catches the residual collisions; bumping past
    24 bits would be belt-and-suspenders. (Original 16-bit suffix was
    flagged in PR review for being too tight under concurrent starts.)
    """
    ts = _now_utc().strftime("%y%m%d-%H%M%S")
    suffix = secrets.token_hex(3)  # 6 hex chars (24 bits)
    return f"{kind}-{ts}-{suffix}"


def _current_git_sha() -> str | None:
    """Return ``git rev-parse HEAD`` or ``None`` if not in a repo / git missing.

    Captured at register time so the row records the code state that was
    *intended* to run, not whatever ``HEAD`` happens to be at finalize.
    """
    try:
        out = subprocess.run(  # noqa: S603 - trusted invocation
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    sha = out.stdout.strip()
    return sha or None


def _hash_string(s: str) -> str:
    """SHA-256 hex digest of a UTF-8 string. Re-exported for tests."""
    return hashlib.sha256(s.encode("utf-8")).hexdigest()
