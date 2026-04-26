"""Registry — register / get / finalize / list / idempotency.

Acceptance scenarios from BUF-69:

* ``register`` returns a run_id, copies config verbatim, writes a DB row.
* ``ls`` filters by kind/status.
* Re-registering the same config + data returns the prior run_id (no-op).
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest
from esports_sim.registry import (
    InvalidKindError,
    Registry,
    RegistryError,
    RunNotFoundError,
    RunStatus,
)

# ---- shared fixtures ------------------------------------------------------


@pytest.fixture
def registry(tmp_path: Path) -> Registry:
    return Registry(
        db_path=tmp_path / "state" / "registry.db",
        runs_dir=tmp_path / "runs",
    )


@pytest.fixture
def config_file(tmp_path: Path) -> Path:
    p = tmp_path / "config.yaml"
    p.write_text("kind: graph-snapshot\nera: 7.09\n", encoding="utf-8")
    return p


# ---- register -------------------------------------------------------------


def test_register_returns_run_id_with_kind_prefix(registry: Registry, config_file: Path) -> None:
    run_id = registry.register(kind="graph-snapshot", config_path=config_file)
    assert run_id.startswith("graph-snapshot-")


def test_register_creates_run_dir_with_config_snapshot(
    registry: Registry, config_file: Path, tmp_path: Path
) -> None:
    """BUF-69 acceptance: ``runs/{run_id}/config.yaml`` exists, byte-identical."""
    run_id = registry.register(kind="graph-snapshot", config_path=config_file)
    run_dir = tmp_path / "runs" / run_id
    snapshot = run_dir / "config.yaml"

    assert run_dir.is_dir()
    assert snapshot.is_file()
    # Verbatim copy — same bytes as the input.
    assert snapshot.read_bytes() == config_file.read_bytes()


def test_register_records_db_row_with_running_status(registry: Registry, config_file: Path) -> None:
    run_id = registry.register(kind="graph-snapshot", config_path=config_file)
    record = registry.get(run_id)
    assert record.status is RunStatus.RUNNING
    assert record.finished_at is None
    assert record.kind == "graph-snapshot"


def test_register_with_data_paths_includes_them_in_fingerprint(
    registry: Registry, config_file: Path, tmp_path: Path
) -> None:
    data = tmp_path / "data.csv"
    data.write_bytes(b"some bytes")

    run_id = registry.register(kind="rl-train", config_path=config_file, data_paths=[data])
    record = registry.get(run_id)
    assert record.data_fingerprint != ""
    # Hex digest length, no slashes / spaces.
    assert len(record.data_fingerprint) == 64


def test_register_with_explicit_fingerprint(registry: Registry, config_file: Path) -> None:
    """Caller-supplied fingerprint is stored verbatim — useful when a
    DuckDB row-count digest is cheaper than a file SHA.
    """
    run_id = registry.register(
        kind="rl-train",
        config_path=config_file,
        data_fingerprint="duckdb-fingerprint-abcdef",
    )
    record = registry.get(run_id)
    assert record.data_fingerprint == "duckdb-fingerprint-abcdef"


def test_register_rejects_invalid_kind(registry: Registry, config_file: Path) -> None:
    with pytest.raises(InvalidKindError):
        registry.register(kind="not valid kind!", config_path=config_file)


def test_register_rejects_data_and_fingerprint_together(
    registry: Registry, config_file: Path, tmp_path: Path
) -> None:
    data = tmp_path / "d.txt"
    data.write_bytes(b"x")
    with pytest.raises(RegistryError, match="data_paths or data_fingerprint"):
        registry.register(
            kind="rl-train",
            config_path=config_file,
            data_paths=[data],
            data_fingerprint="precomputed",
        )


def test_register_missing_config_raises(registry: Registry, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        registry.register(kind="rl-train", config_path=tmp_path / "missing.yaml")


# ---- idempotency (the BUF-69 reproducibility test) ------------------------


def test_re_registering_same_config_returns_same_run_id(
    registry: Registry, config_file: Path
) -> None:
    """The acceptance criterion: same triple → no-op."""
    run_id_1 = registry.register(kind="graph-snapshot", config_path=config_file)
    run_id_2 = registry.register(kind="graph-snapshot", config_path=config_file)
    assert run_id_1 == run_id_2


def test_re_registering_with_same_data_fingerprint_returns_same_run_id(
    registry: Registry, config_file: Path, tmp_path: Path
) -> None:
    data = tmp_path / "data"
    data.mkdir()
    (data / "shard-0").write_bytes(b"shard 0 bytes")
    (data / "shard-1").write_bytes(b"shard 1 bytes")

    run_id_1 = registry.register(kind="rl-train", config_path=config_file, data_paths=[data])
    run_id_2 = registry.register(kind="rl-train", config_path=config_file, data_paths=[data])
    assert run_id_1 == run_id_2


def test_re_registering_with_different_kind_mints_new_run_id(
    registry: Registry, config_file: Path
) -> None:
    """Same config, different kind → distinct run."""
    a = registry.register(kind="graph-snapshot", config_path=config_file)
    b = registry.register(kind="rl-train", config_path=config_file)
    assert a != b


def test_re_registering_after_data_change_mints_new_run_id(
    registry: Registry, config_file: Path, tmp_path: Path
) -> None:
    data = tmp_path / "data.csv"
    data.write_bytes(b"v1")
    a = registry.register(kind="rl-train", config_path=config_file, data_paths=[data])
    data.write_bytes(b"v2")
    b = registry.register(kind="rl-train", config_path=config_file, data_paths=[data])
    assert a != b


def test_re_registering_after_config_change_mints_new_run_id(
    registry: Registry, config_file: Path
) -> None:
    a = registry.register(kind="graph-snapshot", config_path=config_file)
    config_file.write_text("kind: graph-snapshot\nera: 7.10\n", encoding="utf-8")
    b = registry.register(kind="graph-snapshot", config_path=config_file)
    assert a != b


def test_idempotent_register_does_not_create_extra_directories(
    registry: Registry, config_file: Path, tmp_path: Path
) -> None:
    """The no-op path is genuinely no-op — re-registration doesn't pollute
    the runs dir with empty per-call directories.
    """
    run_id = registry.register(kind="graph-snapshot", config_path=config_file)
    runs_root = tmp_path / "runs"
    dirs_before = sorted(p.name for p in runs_root.iterdir())
    # Same triple, four more times.
    for _ in range(4):
        same = registry.register(kind="graph-snapshot", config_path=config_file)
        assert same == run_id
    dirs_after = sorted(p.name for p in runs_root.iterdir())
    assert dirs_before == dirs_after


# ---- get / paths ----------------------------------------------------------


def test_get_unknown_run_id_raises(registry: Registry) -> None:
    with pytest.raises(RunNotFoundError):
        registry.get("nope-no-such-run")


def test_run_record_resolves_paths_without_hardcoding(
    registry: Registry, config_file: Path, tmp_path: Path
) -> None:
    """Downstream code: ``record.run_dir / "checkpoint.pt"``, never inline strings."""
    run_id = registry.register(kind="rl-train", config_path=config_file)
    record = registry.get(run_id)
    assert record.run_dir == tmp_path / "runs" / run_id
    assert record.config_snapshot == tmp_path / "runs" / run_id / "config.yaml"
    assert record.artifact_path("checkpoints", "step_1000.pt") == (
        tmp_path / "runs" / run_id / "checkpoints" / "step_1000.pt"
    )


# ---- finalize -------------------------------------------------------------


def test_finalize_marks_run_completed(registry: Registry, config_file: Path) -> None:
    run_id = registry.register(kind="rl-train", config_path=config_file)
    registry.finalize(run_id, status=RunStatus.COMPLETED, notes="trained 10M steps")
    record = registry.get(run_id)
    assert record.status is RunStatus.COMPLETED
    assert record.finished_at is not None
    assert record.duration_seconds() is not None
    assert record.duration_seconds() >= 0
    assert "trained 10M steps" in (record.notes or "")


def test_finalize_accepts_string_status(registry: Registry, config_file: Path) -> None:
    """``str`` is just as good as the enum at the boundary."""
    run_id = registry.register(kind="rl-train", config_path=config_file)
    registry.finalize(run_id, status="failed", notes="OOM")
    assert registry.get(run_id).status is RunStatus.FAILED


def test_finalize_with_running_status_raises(registry: Registry, config_file: Path) -> None:
    """``finalize`` is for terminal states only."""
    run_id = registry.register(kind="rl-train", config_path=config_file)
    with pytest.raises(RegistryError, match="RUNNING"):
        registry.finalize(run_id, status=RunStatus.RUNNING)


def test_finalize_unknown_run_raises(registry: Registry) -> None:
    with pytest.raises(RunNotFoundError):
        registry.finalize("nope", status=RunStatus.COMPLETED)


def test_finalize_is_idempotent_on_already_terminal_row(
    registry: Registry, config_file: Path
) -> None:
    """A second finalize on an already-terminal row is a no-op (per docstring)."""
    run_id = registry.register(kind="rl-train", config_path=config_file)
    registry.finalize(run_id, status=RunStatus.COMPLETED, notes="first")
    registry.finalize(run_id, status=RunStatus.FAILED, notes="ignored")
    # First-write-wins: status stays COMPLETED, "first" is preserved,
    # "ignored" never lands.
    record = registry.get(run_id)
    assert record.status is RunStatus.COMPLETED
    assert "first" in (record.notes or "")
    assert "ignored" not in (record.notes or "")


def test_finalize_appends_notes_rather_than_replacing(
    registry: Registry, config_file: Path
) -> None:
    """register-time notes survive finalize-time notes (same pattern as
    BUF-22's update_post: append, don't clobber)."""
    run_id = registry.register(kind="rl-train", config_path=config_file, notes="dataset=v3")
    registry.finalize(run_id, status=RunStatus.COMPLETED, notes="reward=0.87")
    record = registry.get(run_id)
    assert "dataset=v3" in (record.notes or "")
    assert "reward=0.87" in (record.notes or "")


# ---- list -----------------------------------------------------------------


def test_list_runs_returns_newest_first(registry: Registry, config_file: Path) -> None:
    a = registry.register(kind="rl-train", config_path=config_file)
    config_file.write_text("v2", encoding="utf-8")
    b = registry.register(kind="rl-train", config_path=config_file)
    rows = registry.list_runs()
    # Newest (b) before oldest (a).
    assert [r.run_id for r in rows].index(b) < [r.run_id for r in rows].index(a)


def test_list_runs_filters_by_kind(registry: Registry, config_file: Path) -> None:
    snap = registry.register(kind="graph-snapshot", config_path=config_file)
    config_file.write_text("v2", encoding="utf-8")
    train = registry.register(kind="rl-train", config_path=config_file)

    snap_rows = registry.list_runs(kind="graph-snapshot")
    train_rows = registry.list_runs(kind="rl-train")
    assert {r.run_id for r in snap_rows} == {snap}
    assert {r.run_id for r in train_rows} == {train}


def test_list_runs_filters_by_status(registry: Registry, config_file: Path) -> None:
    """BUF-69 mentions ``ls --status=`` as a filter — exercise it."""
    a = registry.register(kind="rl-train", config_path=config_file)
    config_file.write_text("v2", encoding="utf-8")
    b = registry.register(kind="rl-train", config_path=config_file)
    registry.finalize(b, status=RunStatus.COMPLETED)

    running = registry.list_runs(status=RunStatus.RUNNING)
    completed = registry.list_runs(status="completed")
    assert {r.run_id for r in running} == {a}
    assert {r.run_id for r in completed} == {b}


def test_run_record_duration_is_none_while_running(registry: Registry, config_file: Path) -> None:
    run_id = registry.register(kind="rl-train", config_path=config_file)
    assert registry.get(run_id).duration_seconds() is None


def test_run_record_duration_is_positive_after_finalize(
    registry: Registry, config_file: Path
) -> None:
    run_id = registry.register(kind="rl-train", config_path=config_file)
    registry.finalize(run_id, status=RunStatus.COMPLETED)
    duration = registry.get(run_id).duration_seconds()
    assert duration is not None
    # Sanity: registers and finalises in the same test should be < 60s.
    assert timedelta(seconds=0) <= timedelta(seconds=duration) < timedelta(minutes=1)
