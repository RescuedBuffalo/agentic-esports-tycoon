"""Regression tests for the empty-input vs no-input fingerprint collision.

The bug: ``compute_fingerprint([])`` returned ``""`` and so did
``compute_fingerprint([empty_dir])``, because ``_iter_files`` walked the
empty directory, found nothing, and the helper short-circuited on the
empty file list. ``Registry.register`` then treated *"caller passed an
empty data directory"* as the same natural key as *"caller passed no
data at all"*, silently reusing a prior ``run_id`` for what was meant
to be a distinct experiment registration.

The fix: preserve legacy hashing for non-empty file sets, but when the
input list is non-empty and the walk yields zero files, hash a stable
manifest of top-level input basenames instead. ``[]`` still returns
``""``; ``[empty_dir]`` now produces a real digest derived from the
directory's basename.
"""

from __future__ import annotations

from pathlib import Path

from esports_sim.registry import Registry, compute_fingerprint

# ---- fingerprint level ---------------------------------------------------


def test_empty_path_list_returns_empty_string_sentinel() -> None:
    """The "no data provided" sentinel is preserved — only ``[]`` returns
    the empty string. Required so :class:`Registry` can use ``""`` as
    its "config-only natural key" marker.
    """
    assert compute_fingerprint([]) == ""


def test_empty_directory_does_not_collide_with_empty_input(tmp_path: Path) -> None:
    """The acceptance scenario from the review.

    A registration with ``data_paths=[empty_dir]`` must not produce the
    same fingerprint as one with ``data_paths=None`` (which the registry
    represents as ``compute_fingerprint([])``). Otherwise the natural
    key collapses and the registry reuses a stale ``run_id``.
    """
    empty = tmp_path / "shards"
    empty.mkdir()

    digest_no_data = compute_fingerprint([])
    digest_empty_dir = compute_fingerprint([empty])

    assert digest_no_data == ""
    assert digest_empty_dir != ""
    assert len(digest_empty_dir) == 64
    assert digest_empty_dir != digest_no_data


def test_two_empty_directories_with_distinct_basenames_differ(tmp_path: Path) -> None:
    """Different basenames → different digests for empty dirs.

    The input manifest folds in the *basenames* of the inputs, so an
    empty ``shards/`` and an empty ``replays/`` produce distinct
    fingerprints even though both walked to zero files.
    """
    a = tmp_path / "shards"
    b = tmp_path / "replays"
    a.mkdir()
    b.mkdir()

    assert compute_fingerprint([a]) != compute_fingerprint([b])


def test_empty_dir_fingerprint_is_deterministic(tmp_path: Path) -> None:
    """Same empty dir → same digest across calls."""
    empty = tmp_path / "shards"
    empty.mkdir()
    assert compute_fingerprint([empty]) == compute_fingerprint([empty])


def test_empty_dir_then_populated_dir_changes_fingerprint(tmp_path: Path) -> None:
    """Adding a file to a previously-empty dir changes the digest.

    Confirms the manifest-prefix doesn't accidentally swallow file
    content — the file pairs still contribute when present.
    """
    d = tmp_path / "shards"
    d.mkdir()
    digest_empty = compute_fingerprint([d])

    (d / "file_0").write_bytes(b"some bytes")
    digest_populated = compute_fingerprint([d])

    assert digest_empty != digest_populated


def test_populated_directory_fingerprint_is_still_permutation_invariant(
    tmp_path: Path,
) -> None:
    """Regression guard: the manifest prefix didn't break the existing
    invariance for non-empty inputs.

    Two distinct files with the same basename in different parent dirs
    must still produce the same digest regardless of caller order.
    """
    a = tmp_path / "a" / "data.csv"
    b = tmp_path / "b" / "data.csv"
    a.parent.mkdir()
    b.parent.mkdir()
    a.write_bytes(b"alpha")
    b.write_bytes(b"bravo")

    assert compute_fingerprint([a, b]) == compute_fingerprint([b, a])


def test_non_empty_inputs_keep_legacy_fingerprint_shape(tmp_path: Path) -> None:
    """Regression: non-empty datasets keep pre-fix digest compatibility.

    The empty-dir collision fix must not alter historical fingerprints
    for normal (non-empty) workloads, or ``Registry.register`` would
    mint new run IDs after an upgrade.
    """
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_bytes(b"alpha")
    b.write_bytes(b"beta")

    # Legacy algorithm shape:
    # sorted (label, file_sha256) then roll ``label + NUL + hash + NUL``.
    import hashlib

    file_rows = [
        (a.name, hashlib.sha256(a.read_bytes()).hexdigest()),
        (b.name, hashlib.sha256(b.read_bytes()).hexdigest()),
    ]
    file_rows.sort()

    legacy = hashlib.sha256()
    for label, digest in file_rows:
        legacy.update(label.encode("utf-8"))
        legacy.update(b"\x00")
        legacy.update(digest.encode("ascii"))
        legacy.update(b"\x00")

    assert compute_fingerprint([a, b]) == legacy.hexdigest()


# ---- Registry level ------------------------------------------------------


def test_register_with_empty_dir_mints_different_id_than_no_data(
    tmp_path: Path,
) -> None:
    """The bug at the Registry boundary — exactly what the reviewer
    described:

        "In runs where an input directory is temporarily empty (or
        intentionally empty), this can incorrectly reuse a prior run_id
        and collapse distinct experiment registrations."

    Before the fix, both registrations produced an empty-string
    ``data_fingerprint`` and so they hit the same UNIQUE(kind,
    config_hash, data_fingerprint) row. After the fix the empty-dir
    registration carries a real fingerprint and lands its own row.
    """
    config = tmp_path / "config.yaml"
    config.write_text("kind: rl-train\n", encoding="utf-8")
    empty_data = tmp_path / "training_v3"
    empty_data.mkdir()

    reg = Registry(db_path=tmp_path / "registry.db", runs_dir=tmp_path / "runs")

    # First: caller registers without any data dimension.
    no_data_run = reg.register(kind="rl-train", config_path=config)

    # Second: caller registers with an explicitly-empty data directory.
    # (The directory exists; the data hasn't arrived yet, or it was
    # cleared between runs — both real scenarios.)
    empty_dir_run = reg.register(kind="rl-train", config_path=config, data_paths=[empty_data])

    assert no_data_run != empty_dir_run

    # And the empty-dir registration carries a real fingerprint, not
    # the "no data" sentinel.
    record = reg.get(empty_dir_run)
    assert record.data_fingerprint != ""
    assert len(record.data_fingerprint) == 64

    no_data_record = reg.get(no_data_run)
    assert no_data_record.data_fingerprint == ""


def test_register_with_two_distinct_empty_dirs_mints_distinct_ids(
    tmp_path: Path,
) -> None:
    """Empty ``shards/`` and empty ``replays/`` are distinct experiments.

    The manifest fold means same-basename empty dirs *do* still collide
    (an inherent limitation when there's no content to disambiguate);
    the test pins the more useful property — distinct basenames produce
    distinct ids.
    """
    config = tmp_path / "config.yaml"
    config.write_text("kind: rl-train\n", encoding="utf-8")
    shards = tmp_path / "shards"
    replays = tmp_path / "replays"
    shards.mkdir()
    replays.mkdir()

    reg = Registry(db_path=tmp_path / "registry.db", runs_dir=tmp_path / "runs")

    a = reg.register(kind="rl-train", config_path=config, data_paths=[shards])
    b = reg.register(kind="rl-train", config_path=config, data_paths=[replays])

    assert a != b
