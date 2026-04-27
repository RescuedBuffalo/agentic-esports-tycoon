"""Regression tests for two PR-review fixes (round 4):

1. **Narrow migration suppression** — ``_init_schema`` only swallows the
   "duplicate column name" OperationalError from the idempotent
   ALTER TABLE. Other OperationalErrors (database locked, real schema
   corruption) propagate so we don't silently leave the runs table
   without ``run_dir`` and have inserts fail later in harder-to-diagnose
   ways.

2. **Single-read config hash + snapshot** — ``register()`` reads the
   config file once and uses the same buffer for both the hash
   (idempotency key) and the snapshot copy. A concurrent writer can no
   longer mutate the file between the two reads and leave the stored
   ``config_hash`` describing different bytes than
   ``runs/{run_id}/config.yaml``.
"""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
from unittest.mock import patch

import pytest
from esports_sim.registry import Registry

# ---- narrow migration suppression ----------------------------------------


def test_init_schema_suppresses_only_duplicate_column_error(tmp_path: Path) -> None:
    """A non-"duplicate column" OperationalError from the ALTER TABLE
    must propagate — leaving the runs table mid-migration is exactly
    the failure mode we want to surface, not swallow.

    ``sqlite3.Connection.execute`` is on an immutable C type and can't
    be monkey-patched directly, so we wrap the connection in a proxy
    that intercepts the ALTER TABLE call.
    """
    db_path = tmp_path / "registry.db"
    runs_dir = tmp_path / "runs"

    class _FlakyConn:
        """Forwards everything except an ALTER TABLE that we make fail."""

        def __init__(self, real: sqlite3.Connection) -> None:
            self._real = real

        def __getattr__(self, name: str) -> object:
            return getattr(self._real, name)

        def execute(self, sql: str, *args: object, **kwargs: object) -> object:
            if "ALTER TABLE runs ADD COLUMN run_dir" in sql:
                # Simulates a real migration failure (e.g. a concurrent
                # writer holding the write lock long enough to time out).
                raise sqlite3.OperationalError("database is locked")
            return self._real.execute(sql, *args, **kwargs)

    real_open = Registry._open_connection

    def patched_open(self: Registry) -> object:
        return _FlakyConn(real_open(self))

    with (
        patch.object(Registry, "_open_connection", patched_open),
        pytest.raises(sqlite3.OperationalError, match="database is locked"),
    ):
        Registry(db_path=db_path, runs_dir=runs_dir)


def test_init_schema_silently_accepts_duplicate_column(tmp_path: Path) -> None:
    """The complementary case: opening a Registry against a DB that
    already has the run_dir column doesn't raise — that's the migration
    path the suppress was added for.
    """
    db_path = tmp_path / "registry.db"
    runs_dir = tmp_path / "runs"
    # First open creates the table + adds the column. Second open hits
    # the "duplicate column name" branch.
    Registry(db_path=db_path, runs_dir=runs_dir)
    Registry(db_path=db_path, runs_dir=runs_dir)
    # No exception.


# ---- single-read config hash + snapshot ----------------------------------


def test_snapshot_bytes_hash_matches_stored_config_hash(tmp_path: Path) -> None:
    """The invariant: bytes on disk under ``runs/{run_id}/config.yaml``
    must SHA-256 to ``record.config_hash``. With a static input file
    this is trivially true; the test pins the property so a future
    "we'll just read twice" regression is caught immediately.
    """
    config = tmp_path / "config.yaml"
    config.write_text("kind: rl-train\nseed: 1\n", encoding="utf-8")

    reg = Registry(db_path=tmp_path / "registry.db", runs_dir=tmp_path / "runs")
    run_id = reg.register(kind="rl-train", config_path=config)

    record = reg.get(run_id)
    snapshot_bytes = record.config_snapshot.read_bytes()
    assert hashlib.sha256(snapshot_bytes).hexdigest() == record.config_hash


def test_concurrent_config_mutation_does_not_split_hash_and_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The acceptance scenario from the review.

    Simulate a concurrent writer mutating the config file *between* the
    hash compute and the snapshot copy. Before the fix the registry
    read the file twice (once for ``hash_file``, once for
    ``shutil.copy2``); the second read would see the mutated bytes and
    the stored hash would no longer match the snapshot.

    The fix reads once into memory and uses the same buffer for both,
    so the invariant ``sha256(snapshot) == config_hash`` holds even
    under concurrent mutation.

    To force the bug deterministically we patch ``Path.read_bytes`` so
    the *first* call returns the original bytes, and a side effect
    rewrites the file to contain different bytes before the snapshot
    copy. With a streaming "read twice" implementation that would
    desync; with the single-read fix it stays consistent.
    """
    config = tmp_path / "config.yaml"
    original = b"kind: rl-train\nseed: 1\n"
    mutated = b"kind: rl-train\nseed: 999\n"
    config.write_bytes(original)

    real_read_bytes = Path.read_bytes
    seen_calls = {"n": 0}

    def racy_read_bytes(self: Path) -> bytes:
        if self == config:
            seen_calls["n"] += 1
            data = real_read_bytes(self)
            # Simulate another writer flipping the file *between* our
            # read and any subsequent file-system access. With the fix
            # there is no subsequent read, so this mutation is
            # invisible to the registry; without the fix it would be
            # picked up by shutil.copy2 / a second hash_file pass.
            config.write_bytes(mutated)
            return data
        return real_read_bytes(self)

    reg = Registry(db_path=tmp_path / "registry.db", runs_dir=tmp_path / "runs")

    with patch.object(Path, "read_bytes", racy_read_bytes):
        run_id = reg.register(kind="rl-train", config_path=config)

    # Exactly one read of the config — the single-buffer pattern.
    assert seen_calls["n"] == 1

    # Invariant: the stored snapshot's bytes hash to the row's config_hash.
    record = reg.get(run_id)
    snapshot_bytes = record.config_snapshot.read_bytes()
    assert hashlib.sha256(snapshot_bytes).hexdigest() == record.config_hash
    # And both describe the *original* bytes — the mutation slipped in
    # after our snapshot, not before it.
    assert snapshot_bytes == original


def test_idempotency_unaffected_by_pre_register_config_mutation(tmp_path: Path) -> None:
    """A second register against a mutated config still mints a new id.

    If the config file changes after the first register, the second
    register sees different bytes → different config_hash → new natural
    key → new run_id. (This is the existing idempotency behaviour; we
    re-assert it here to confirm the single-read refactor didn't
    accidentally cache the bytes across calls.)
    """
    config = tmp_path / "config.yaml"
    config.write_bytes(b"kind: rl-train\nseed: 1\n")

    reg = Registry(db_path=tmp_path / "registry.db", runs_dir=tmp_path / "runs")
    a = reg.register(kind="rl-train", config_path=config)

    config.write_bytes(b"kind: rl-train\nseed: 2\n")
    b = reg.register(kind="rl-train", config_path=config)

    assert a != b
