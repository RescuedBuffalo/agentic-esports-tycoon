"""Regression tests for two PR-review fixes (round 2):

1. **Stable run_dir resolution** — ``RunRecord.run_dir`` must come from
   the DB row stored at register time, not from whichever ``runs_dir``
   the registry happens to be opened with later. Otherwise a run created
   under ``/path/A`` and looked up via a registry pointed at ``/path/B``
   would silently report ``/path/B/{run_id}`` as its artifact directory,
   stranding any artifacts already written under ``/path/A``.

2. **TOCTOU-safe run_dir creation** — between checking
   ``run_dir.exists()`` and calling ``run_dir.mkdir()``, a sibling
   process can create the directory. The retry loop now uses
   ``mkdir(exist_ok=False)`` and catches ``FileExistsError`` so the
   collision path stays robust under concurrent starts.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from esports_sim.registry import Registry
from esports_sim.registry import db as registry_db

# ---- stable run_dir from stored DB column ---------------------------------


def test_run_dir_is_stable_when_registry_reopened_with_different_runs_dir(
    tmp_path: Path,
) -> None:
    """The acceptance scenario the reviewer flagged.

    Register a run under ``runs_a/``. Open a *new* Registry instance
    pointing at the same DB but a different ``runs_dir`` (``runs_b/``).
    The looked-up run's ``run_dir`` must still resolve under ``runs_a/``
    — otherwise downstream code would write checkpoints in the wrong
    place and split artifacts across roots.
    """
    db = tmp_path / "registry.db"
    runs_a = tmp_path / "runs_a"
    runs_b = tmp_path / "runs_b"

    config = tmp_path / "config.yaml"
    config.write_text("kind: rl-train\n", encoding="utf-8")

    # Register under runs_a.
    reg_a = Registry(db_path=db, runs_dir=runs_a)
    run_id = reg_a.register(kind="rl-train", config_path=config)
    record_a = reg_a.get(run_id)
    assert record_a.run_dir == runs_a / run_id

    # Open a *different* Registry against the same DB but pointing at a
    # new artifact root. The looked-up record must still report the
    # original run_dir, NOT runs_b/{run_id}.
    reg_b = Registry(db_path=db, runs_dir=runs_b)
    record_b = reg_b.get(run_id)
    assert record_b.run_dir == runs_a / run_id, (
        "RunRecord.run_dir must come from the stored DB row, not the " "registry's current runs_dir"
    )
    # artifact_path() builds on top of that stored value, so any
    # downstream code asking for runs/{id}/checkpoint.pt also resolves
    # correctly.
    assert record_b.artifact_path("checkpoint.pt") == runs_a / run_id / "checkpoint.pt"


def test_legacy_rows_without_stored_run_dir_fall_back_to_current_runs_dir(
    tmp_path: Path,
) -> None:
    """Older DBs created before the run_dir column existed kept rows
    with no stored ``run_dir``. The reader falls back to
    ``self.runs_dir / run_id`` for those — accepting that legacy rows
    inherit the original bug, while fresh ones don't.
    """
    db = tmp_path / "registry.db"
    runs_dir = tmp_path / "runs"

    # Create a registry, register a run, then manually clear the stored
    # run_dir column to simulate a row from before the migration.
    reg = Registry(db_path=db, runs_dir=runs_dir)
    config = tmp_path / "config.yaml"
    config.write_text("kind: rl-train\n", encoding="utf-8")
    run_id = reg.register(kind="rl-train", config_path=config)

    import sqlite3

    with sqlite3.connect(db) as conn:
        conn.execute("UPDATE runs SET run_dir = '' WHERE run_id = ?", (run_id,))

    # Reopen and look up. With no stored run_dir the reader falls back
    # to runs_dir / run_id.
    reg2 = Registry(db_path=db, runs_dir=runs_dir)
    record = reg2.get(run_id)
    assert record.run_dir == runs_dir / run_id


def test_init_schema_is_idempotent_against_old_db_without_run_dir_column(
    tmp_path: Path,
) -> None:
    """The ALTER TABLE migration is run on every Registry init. Catching
    ``OperationalError("duplicate column name")`` makes that safe.
    """
    db = tmp_path / "registry.db"

    # Build a "legacy" DB by creating the table without the run_dir column.
    import sqlite3

    with sqlite3.connect(db) as conn:
        conn.executescript("""
            CREATE TABLE runs (
                run_id TEXT PRIMARY KEY,
                kind TEXT NOT NULL,
                git_sha TEXT,
                started_at TEXT NOT NULL,
                finished_at TEXT,
                config_snapshot_path TEXT NOT NULL,
                config_hash TEXT NOT NULL,
                data_fingerprint TEXT NOT NULL DEFAULT '',
                status TEXT NOT NULL,
                notes TEXT,
                UNIQUE(kind, config_hash, data_fingerprint)
            );
            """)

    # Opening a Registry should add the run_dir column without raising.
    Registry(db_path=db, runs_dir=tmp_path / "runs")

    # And opening it a *second* time must not fail on "duplicate column".
    Registry(db_path=db, runs_dir=tmp_path / "runs")

    # Confirm the column is now present.
    with sqlite3.connect(db) as conn:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(runs)").fetchall()}
    assert "run_dir" in cols


# ---- TOCTOU-safe mkdir ----------------------------------------------------


def test_register_recovers_from_concurrent_dir_creation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Simulate the race the reviewer pointed out.

    Force ``_generate_run_id`` to mint a deterministic id. Patch
    ``Path.mkdir`` so the *first* call (against the colliding id) raises
    ``FileExistsError`` — as if a sibling process won the race between
    the existence check and our own ``mkdir``. The register call must
    catch that, mint a fresh id, and try again.
    """
    config = tmp_path / "config.yaml"
    config.write_text("kind: rl-train\n", encoding="utf-8")

    reg = Registry(db_path=tmp_path / "registry.db", runs_dir=tmp_path / "runs")

    # Two ids in the queue: the first one will "lose" the mkdir race,
    # the second one wins.
    losing_id = "rl-train-260427-120000-loser"
    winning_id = "rl-train-260427-120001-winner"
    ids = iter([losing_id, winning_id])
    monkeypatch.setattr(registry_db, "_generate_run_id", lambda kind: next(ids))

    real_mkdir = Path.mkdir

    def flaky_mkdir(self: Path, *args: object, **kwargs: object) -> None:
        if losing_id in str(self):
            # Pretend a sibling process created the directory between
            # our id generation and our own mkdir. ``exist_ok=False``
            # would normally raise this — simulate the race directly.
            raise FileExistsError(self)
        real_mkdir(self, *args, **kwargs)  # type: ignore[arg-type]

    with patch.object(Path, "mkdir", flaky_mkdir):
        run_id = reg.register(kind="rl-train", config_path=config)

    # Caller saw the *winning* id, not an exception.
    assert run_id == winning_id
    record = reg.get(run_id)
    assert record.run_id == winning_id
    # The losing id was never persisted; only the winner has a row.
    assert reg.list_runs() == [record]
