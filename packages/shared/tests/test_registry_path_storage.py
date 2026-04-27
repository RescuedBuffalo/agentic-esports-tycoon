"""Regression tests for two PR-review fixes (round 3):

1. **Absolute paths in stored rows** — ``register()`` previously stored
   whatever string the caller's ``runs_dir`` resolved to, so a relative
   ``runs/`` (the default) ended up persisted as a relative string.
   Opening the same DB from a different CWD then resolved that string
   against the new CWD and pointed ``record.run_dir`` at the wrong place.
   Fixed by resolving ``runs_dir`` (and therefore every derived path) to
   an absolute path in ``__init__``.

2. **`:memory:` mode actually works** — SQLite ``:memory:`` databases
   are per-connection. The old ``_connect()`` opened a fresh connection
   per call, so the schema created by ``_init_schema`` was invisible to
   every subsequent operation, surfacing as ``no such table: runs``. The
   Registry now holds one persistent connection for in-memory mode and
   yields it from ``_connect()`` without closing.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from esports_sim.registry import Registry, RunStatus

# ---- absolute path storage -----------------------------------------------


def test_relative_runs_dir_is_resolved_to_absolute_in_constructor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``Registry(runs_dir=Path("runs"))`` keeps ``self.runs_dir`` absolute.

    Before the fix the relative path leaked into INSERT statements; this
    test pins the resolution at the boundary so the rest of the registry
    works with absolute paths regardless of caller input.
    """
    monkeypatch.chdir(tmp_path)
    reg = Registry(db_path=tmp_path / "registry.db", runs_dir=Path("runs"))
    assert reg.runs_dir.is_absolute()
    assert reg.runs_dir == (tmp_path / "runs").resolve()


def test_record_run_dir_survives_cwd_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The acceptance scenario the reviewer flagged.

    Register a run from CWD ``A`` with a relative ``runs/`` argument.
    Open the same DB from CWD ``B``. ``record.run_dir`` must still
    resolve to the original location — not to ``B/runs/{run_id}``.
    """
    cwd_a = tmp_path / "cwd_a"
    cwd_b = tmp_path / "cwd_b"
    cwd_a.mkdir()
    cwd_b.mkdir()

    config = tmp_path / "config.yaml"
    config.write_text("kind: rl-train\n", encoding="utf-8")

    # Register from CWD A with a *relative* runs/ path.
    monkeypatch.chdir(cwd_a)
    reg_a = Registry(db_path=tmp_path / "registry.db", runs_dir=Path("runs"))
    run_id = reg_a.register(kind="rl-train", config_path=config)
    expected_run_dir = (cwd_a / "runs" / run_id).resolve()

    # Open from CWD B — same DB, but now the process default for
    # relative paths would resolve under cwd_b. The stored value must
    # win.
    monkeypatch.chdir(cwd_b)
    reg_b = Registry(db_path=tmp_path / "registry.db", runs_dir=Path("runs"))
    record = reg_b.get(run_id)

    assert record.run_dir == expected_run_dir
    assert record.run_dir != (cwd_b / "runs" / run_id).resolve()
    # config_snapshot, derived from config_snapshot_path, also points at
    # the original location.
    assert record.config_snapshot.parent == expected_run_dir


def test_artifact_path_is_absolute_for_cross_machine_use(tmp_path: Path) -> None:
    """``record.artifact_path("checkpoint.pt")`` is absolute, not relative.

    Downstream code (e.g. a Prefect flow on another machine) needs to
    open the file by path; a relative path would resolve against the
    flow's CWD, not the original.
    """
    config = tmp_path / "config.yaml"
    config.write_text("kind: rl-train\n", encoding="utf-8")
    reg = Registry(db_path=tmp_path / "registry.db", runs_dir=tmp_path / "runs")
    run_id = reg.register(kind="rl-train", config_path=config)

    record = reg.get(run_id)
    assert record.run_dir.is_absolute()
    assert record.artifact_path("checkpoint.pt").is_absolute()
    assert record.config_snapshot.is_absolute()


# ---- :memory: mode --------------------------------------------------------


def test_in_memory_registry_is_usable_end_to_end(tmp_path: Path) -> None:
    """``:memory:`` mode no longer fails with ``no such table: runs``.

    The fix holds one persistent connection so the schema and the
    subsequent register/list/get/finalize all see the same database. We
    drive a full lifecycle to confirm.
    """
    config = tmp_path / "config.yaml"
    config.write_text("kind: rl-train\n", encoding="utf-8")
    reg = Registry(db_path=":memory:", runs_dir=tmp_path / "runs")

    run_id = reg.register(kind="rl-train", config_path=config)

    record = reg.get(run_id)
    assert record.run_id == run_id
    assert record.status is RunStatus.RUNNING

    rows = reg.list_runs(kind="rl-train")
    assert len(rows) == 1
    assert rows[0].run_id == run_id

    reg.finalize(run_id, status=RunStatus.COMPLETED, notes="done")
    assert reg.get(run_id).status is RunStatus.COMPLETED


def test_in_memory_idempotency_works_within_one_registry(tmp_path: Path) -> None:
    """Re-registering the same triple in :memory: mode still no-ops.

    This is the stronger end-to-end assertion: idempotency exists *only*
    if subsequent reads see the previously inserted row, which requires
    the persistent-connection fix.
    """
    config = tmp_path / "config.yaml"
    config.write_text("kind: rl-train\n", encoding="utf-8")
    reg = Registry(db_path=":memory:", runs_dir=tmp_path / "runs")

    a = reg.register(kind="rl-train", config_path=config)
    b = reg.register(kind="rl-train", config_path=config)
    assert a == b
    assert len(reg.list_runs()) == 1


def test_in_memory_db_is_not_persisted_to_disk(tmp_path: Path) -> None:
    """Sanity: ``:memory:`` doesn't accidentally create a ``:memory:`` file
    on disk. (We previously special-cased it in ``__init__`` but a
    regression here would silently leak per-test files into CWD.)
    """
    cwd_before = set(os.listdir(tmp_path))
    Registry(db_path=":memory:", runs_dir=tmp_path / "runs")
    cwd_after = set(os.listdir(tmp_path))
    # Only the runs/ dir should appear; no `:memory:` file.
    assert cwd_after - cwd_before <= {"runs"}
