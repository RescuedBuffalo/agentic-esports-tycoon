"""Append-only SQLite ledger for Claude API spend (BUF-22, ADR-006).

Rationale for SQLite (not Postgres): the ledger is single-writer per
process group, append-only, and queried with simple aggregates. ADR-006
explicitly carves it out as a SQLite-with-WAL-mode workload — keeping it
out of Postgres means a budget query during a Postgres outage still works.

The schema is intentionally narrow: one row per Claude call, with the
fields needed to (a) reconcile against the Anthropic console (timestamp,
model, tokens, usd_cost) and (b) attribute spend (purpose, request_id).

Every connection sets ``PRAGMA journal_mode=WAL`` so concurrent readers
don't block the writer; ``synchronous=NORMAL`` is the WAL-recommended
level — durable across process crash, fast across power loss.
"""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

# Default DB location. Operators override via NEXUS_BUDGET_DB so prod, dev,
# and tests all point at distinct files.
_DEFAULT_DB_PATH = Path(".nexus") / "budget.sqlite"

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS api_ledger (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    -- ISO-8601 UTC, e.g. 2026-04-26T19:42:00.123456+00:00. Stored as TEXT
    -- so SQLite's lexicographic sort matches chronological order.
    timestamp TEXT NOT NULL,
    endpoint TEXT NOT NULL,
    model TEXT NOT NULL,
    input_tokens INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    cache_creation_input_tokens INTEGER NOT NULL DEFAULT 0,
    cache_read_input_tokens INTEGER NOT NULL DEFAULT 0,
    usd_cost REAL NOT NULL DEFAULT 0.0,
    purpose TEXT NOT NULL,
    -- request_id from Anthropic so we can cross-reference with the console.
    request_id TEXT,
    -- pre|post|blocked|error. ``pre`` is written before the call, ``post``
    -- after success, ``blocked`` when the governor refused, ``error`` when
    -- the call raised after the pre row.
    phase TEXT NOT NULL,
    notes TEXT
);
CREATE INDEX IF NOT EXISTS ix_api_ledger_timestamp ON api_ledger(timestamp);
CREATE INDEX IF NOT EXISTS ix_api_ledger_purpose ON api_ledger(purpose);
"""


@dataclass(frozen=True)
class LedgerEntry:
    """One row from ``api_ledger`` — what the SQLite query returns."""

    id: int
    timestamp: datetime
    endpoint: str
    model: str
    input_tokens: int
    output_tokens: int
    cache_creation_input_tokens: int
    cache_read_input_tokens: int
    usd_cost: float
    purpose: str
    request_id: str | None
    phase: str
    notes: str | None


def _resolve_db_path() -> Path:
    raw = os.getenv("NEXUS_BUDGET_DB")
    return Path(raw) if raw else _DEFAULT_DB_PATH


def _now_utc() -> datetime:
    return datetime.now(UTC)


class Ledger:
    """Thin wrapper around an append-only SQLite log."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        self.db_path = Path(db_path) if db_path is not None else _resolve_db_path()
        # Ensure the parent directory exists so callers can pass a fresh path
        # without ``mkdir``-ing it themselves.
        if str(self.db_path) != ":memory:":
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    # ---- connection helpers ------------------------------------------------

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, isolation_level=None)
        try:
            # WAL is recommended for one-writer-many-readers workloads; it
            # also keeps reads from blocking writes (a query for the daily
            # digest never blocks the next claude_call).
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.row_factory = sqlite3.Row
            yield conn
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA_SQL)

    # ---- writes ------------------------------------------------------------

    def record(
        self,
        *,
        endpoint: str,
        model: str,
        purpose: str,
        phase: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_creation_input_tokens: int = 0,
        cache_read_input_tokens: int = 0,
        usd_cost: float = 0.0,
        request_id: str | None = None,
        notes: str | None = None,
        timestamp: datetime | None = None,
    ) -> int:
        """Insert a row, return the new ``id``."""
        ts = (timestamp or _now_utc()).isoformat()
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO api_ledger (
                    timestamp, endpoint, model, input_tokens, output_tokens,
                    cache_creation_input_tokens, cache_read_input_tokens,
                    usd_cost, purpose, request_id, phase, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ts,
                    endpoint,
                    model,
                    int(input_tokens),
                    int(output_tokens),
                    int(cache_creation_input_tokens),
                    int(cache_read_input_tokens),
                    float(usd_cost),
                    purpose,
                    request_id,
                    phase,
                    notes,
                ),
            )
            row_id = cur.lastrowid
            assert row_id is not None
            return row_id

    def update_post(
        self,
        row_id: int,
        *,
        input_tokens: int,
        output_tokens: int,
        cache_creation_input_tokens: int,
        cache_read_input_tokens: int,
        usd_cost: float,
        request_id: str | None,
        phase: str = "post",
        notes: str | None = None,
    ) -> None:
        """Reconcile a ``pre`` row with actual usage after the call returns.

        The ledger is append-only in spirit, but the row created at pre-flight
        has no real usage yet — overwriting it with the post-flight numbers is
        cleaner than inserting two rows and joining them in every report query.
        We log the transition via ``phase`` so the audit trail shows what
        happened.
        """
        with self._connect() as conn:
            conn.execute(
                """
                UPDATE api_ledger
                SET input_tokens = ?,
                    output_tokens = ?,
                    cache_creation_input_tokens = ?,
                    cache_read_input_tokens = ?,
                    usd_cost = ?,
                    request_id = ?,
                    phase = ?,
                    notes = COALESCE(?, notes)
                WHERE id = ?
                """,
                (
                    int(input_tokens),
                    int(output_tokens),
                    int(cache_creation_input_tokens),
                    int(cache_read_input_tokens),
                    float(usd_cost),
                    request_id,
                    phase,
                    notes,
                    row_id,
                ),
            )

    # ---- reads -------------------------------------------------------------

    def total_spend_since(
        self,
        since: datetime,
        *,
        purpose: str | None = None,
    ) -> float:
        """Sum ``usd_cost`` for rows newer than *since*.

        Counts ``pre`` rows too so a long-running call already in flight
        doesn't get billed twice — the pre row is overwritten with actuals
        when ``update_post`` runs.
        """
        sql = "SELECT COALESCE(SUM(usd_cost), 0.0) AS total " "FROM api_ledger WHERE timestamp >= ?"
        params: list[object] = [since.astimezone(UTC).isoformat()]
        if purpose is not None:
            sql += " AND purpose = ?"
            params.append(purpose)
        with self._connect() as conn:
            row = conn.execute(sql, params).fetchone()
        return float(row["total"]) if row is not None else 0.0

    def weekly_spend(self, *, purpose: str | None = None, now: datetime | None = None) -> float:
        """Sum spend over the rolling 7-day window ending now."""
        end = now or _now_utc()
        return self.total_spend_since(end - timedelta(days=7), purpose=purpose)

    def entries_since(self, since: datetime) -> list[LedgerEntry]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM api_ledger WHERE timestamp >= ? ORDER BY timestamp",
                (since.astimezone(UTC).isoformat(),),
            ).fetchall()
        return [_row_to_entry(r) for r in rows]

    def all_entries(self) -> list[LedgerEntry]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM api_ledger ORDER BY timestamp").fetchall()
        return [_row_to_entry(r) for r in rows]


def _row_to_entry(row: sqlite3.Row) -> LedgerEntry:
    return LedgerEntry(
        id=int(row["id"]),
        timestamp=datetime.fromisoformat(row["timestamp"]),
        endpoint=str(row["endpoint"]),
        model=str(row["model"]),
        input_tokens=int(row["input_tokens"]),
        output_tokens=int(row["output_tokens"]),
        cache_creation_input_tokens=int(row["cache_creation_input_tokens"]),
        cache_read_input_tokens=int(row["cache_read_input_tokens"]),
        usd_cost=float(row["usd_cost"]),
        purpose=str(row["purpose"]),
        request_id=row["request_id"],
        phase=str(row["phase"]),
        notes=row["notes"],
    )
