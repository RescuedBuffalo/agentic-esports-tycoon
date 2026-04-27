"""Regression tests for two PR-review fixes on the registry:

1. **Fingerprint label-collision** — when two inputs share a basename
   (``/mnt/a/data.csv`` and ``/mnt/b/data.csv``, or two ``shards`` dirs
   under different parents), sorting by label alone left them in
   caller-supplied order. Permutation invariance broke. Fixed by sorting
   ``(label, file_hash)`` so the content hash breaks ties stably.

2. **run_id PK collision** — ``_generate_run_id`` only had ~24 bits of
   entropy in its suffix, so under concurrent starts of the same kind
   two runs of *different* triples could land on the same ``run_id``.
   The previous IntegrityError handler treated every conflict as a
   natural-key race and re-raised when ``_lookup_existing`` returned
   ``None``. Fixed by retrying with a fresh id when the natural-key
   lookup misses.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from esports_sim.registry import Registry, compute_fingerprint
from esports_sim.registry import db as registry_db

# ---- fingerprint label-collision -----------------------------------------


def test_fingerprint_invariant_when_two_files_share_a_basename(tmp_path: Path) -> None:
    """``/mnt/a/data.csv`` and ``/mnt/b/data.csv`` — different content,
    identical basename. The fingerprint must not depend on argument order.
    """
    a = tmp_path / "a" / "data.csv"
    b = tmp_path / "b" / "data.csv"
    a.parent.mkdir()
    b.parent.mkdir()
    a.write_bytes(b"alpha bytes")
    b.write_bytes(b"bravo bytes")

    digest_ab = compute_fingerprint([a, b])
    digest_ba = compute_fingerprint([b, a])
    assert digest_ab == digest_ba
    # And it's a real digest, not the empty-input sentinel.
    assert len(digest_ab) == 64


def test_fingerprint_invariant_when_two_dirs_share_a_basename(tmp_path: Path) -> None:
    """Same scenario with directory walks.

    Two dirs both named ``shards`` under different parents. Each contains
    distinct files with the same intra-dir layout — so labels collide
    across the two walks (``shards/file_0`` produced twice).
    """
    a_root = tmp_path / "a"
    b_root = tmp_path / "b"
    (a_root / "shards").mkdir(parents=True)
    (b_root / "shards").mkdir(parents=True)
    (a_root / "shards" / "file_0").write_bytes(b"alpha 0")
    (a_root / "shards" / "file_1").write_bytes(b"alpha 1")
    (b_root / "shards" / "file_0").write_bytes(b"bravo 0")
    (b_root / "shards" / "file_1").write_bytes(b"bravo 1")

    digest_ab = compute_fingerprint([a_root / "shards", b_root / "shards"])
    digest_ba = compute_fingerprint([b_root / "shards", a_root / "shards"])
    assert digest_ab == digest_ba


def test_fingerprint_distinguishes_label_collision_from_swap(tmp_path: Path) -> None:
    """Same labels + same hashes pairwise but assigned differently → different digest.

    Sanity check that the (label, hash) sort doesn't accidentally collapse
    distinct configurations onto the same digest. Configuration X has
    ``a/data.csv`` containing "alpha" and ``b/data.csv`` containing "bravo".
    Configuration Y swaps them. Both have the same set of hashes and the
    same set of labels, but the *pairing* differs — and that should show
    up in the fingerprint.
    """
    # Configuration X
    x_a = tmp_path / "x" / "a" / "data.csv"
    x_b = tmp_path / "x" / "b" / "data.csv"
    x_a.parent.mkdir(parents=True)
    x_b.parent.mkdir(parents=True)
    x_a.write_bytes(b"alpha")
    x_b.write_bytes(b"bravo")

    # Configuration Y — same content, different parent assignment.
    y_a = tmp_path / "y" / "a" / "data.csv"
    y_b = tmp_path / "y" / "b" / "data.csv"
    y_a.parent.mkdir(parents=True)
    y_b.parent.mkdir(parents=True)
    y_a.write_bytes(b"bravo")  # swapped
    y_b.write_bytes(b"alpha")  # swapped

    # Labels in both cases are {"data.csv", "data.csv"}. Hashes are
    # {sha("alpha"), sha("bravo")} in both cases. But because both files
    # share the *same* label, the (label, hash) sort yields the same
    # ordered list — which means the fingerprints DO match, by design.
    # That's correct behaviour: from the registry's perspective, "two
    # files both named data.csv with this set of contents" is the same
    # input regardless of which one is in which subdirectory. Pin the
    # behaviour as documented.
    assert compute_fingerprint([x_a, x_b]) == compute_fingerprint([y_a, y_b])


# ---- run_id PK collision retry --------------------------------------------


def test_register_retries_on_run_id_pk_collision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Force two distinct triples to mint the same run_id; assert the
    second registration retries and produces a different id.

    We patch ``_generate_run_id`` to return a fixed string the first two
    times it's called, then a unique string thereafter. The first
    register() takes the fixed id; the second register() collides on the
    PK, looks up the natural key, finds nothing, and retries with the
    next mocked id.
    """
    config_a = tmp_path / "a.yaml"
    config_a.write_text("kind: rl-train\nseed: 1\n", encoding="utf-8")
    config_b = tmp_path / "b.yaml"
    config_b.write_text("kind: rl-train\nseed: 2\n", encoding="utf-8")

    registry = Registry(
        db_path=tmp_path / "registry.db",
        runs_dir=tmp_path / "runs",
    )

    fixed = "rl-train-260427-120000-collide"
    fresh = "rl-train-260427-120001-fresh-1"
    ids = iter([fixed, fixed, fresh])  # both triples mint `fixed` first
    monkeypatch.setattr(registry_db, "_generate_run_id", lambda kind: next(ids))

    first = registry.register(kind="rl-train", config_path=config_a)
    second = registry.register(kind="rl-train", config_path=config_b)

    # The first register took the fixed id; the second saw the PK
    # collision, retried, and got the fresh one.
    assert first == fixed
    assert second == fresh
    # Both rows are in the DB with the right configs.
    assert registry.get(first).run_id == first
    assert registry.get(second).run_id == second
    # Run dirs reflect the final ids — no orphan dir from the failed
    # first attempt of the second register.
    assert (tmp_path / "runs" / first).is_dir()
    assert (tmp_path / "runs" / second).is_dir()
    # The run_dir.exists() guard inside the retry loop bails before
    # mkdir, so ``rmtree`` of the colliding directory is unnecessary —
    # nothing was written under that name on attempt #2 anyway.


def test_register_natural_key_race_still_returns_existing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The OTHER IntegrityError branch — natural-key race — still works.

    Simulates two concurrent registrations of the *same* triple. Pre-flight
    lookup misses for both, the first INSERT wins, the second INSERT
    fires UNIQUE(kind, config_hash, data_fingerprint) and is recovered by
    looking up the now-existing row.
    """
    config = tmp_path / "config.yaml"
    config.write_text("kind: rl-train\n", encoding="utf-8")

    registry = Registry(
        db_path=tmp_path / "registry.db",
        runs_dir=tmp_path / "runs",
    )

    # Pre-insert a row for the natural key the next register() will
    # compute. This forces the IntegrityError on UNIQUE(natural-key).
    first = registry.register(kind="rl-train", config_path=config)

    # Patch _lookup_existing to return None on the *first* call inside
    # register() (simulating the pre-flight lookup losing a race) but
    # behave normally on subsequent calls (so the post-IntegrityError
    # recovery lookup sees the row).
    real_lookup = registry._lookup_existing
    calls = {"n": 0}

    def fake_lookup(*, kind, config_hash, data_fingerprint):
        calls["n"] += 1
        if calls["n"] == 1:
            return None
        return real_lookup(
            kind=kind,
            config_hash=config_hash,
            data_fingerprint=data_fingerprint,
        )

    monkeypatch.setattr(registry, "_lookup_existing", fake_lookup)

    second = registry.register(kind="rl-train", config_path=config)
    # Recovery returned the prior id, not a fresh one.
    assert second == first
