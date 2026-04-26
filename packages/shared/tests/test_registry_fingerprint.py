"""Data fingerprint helper.

Pinpoints the properties the registry's idempotency check relies on:

* deterministic across invocations on the same bytes,
* sensitive to *any* content change in the input set,
* permutation-invariant on input order (sort happens inside the helper),
* recursive across directories,
* well-defined empty case (returns ``""`` so the registry can use it as
  "no data fingerprint" without conflating with a real digest).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from esports_sim.registry import compute_fingerprint, hash_file


def test_empty_input_returns_empty_string() -> None:
    """Sentinel value the registry uses for "no data fingerprint"."""
    assert compute_fingerprint([]) == ""


def test_single_file_fingerprint_is_deterministic(tmp_path: Path) -> None:
    f = tmp_path / "data.csv"
    f.write_bytes(b"player_id,kills\nx,10\ny,20\n")
    assert compute_fingerprint([f]) == compute_fingerprint([f])


def test_fingerprint_changes_when_content_changes(tmp_path: Path) -> None:
    f = tmp_path / "data.csv"
    f.write_bytes(b"a")
    digest_a = compute_fingerprint([f])
    f.write_bytes(b"b")
    digest_b = compute_fingerprint([f])
    assert digest_a != digest_b


def test_fingerprint_independent_of_input_argument_order(tmp_path: Path) -> None:
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_bytes(b"alpha")
    b.write_bytes(b"beta")
    assert compute_fingerprint([a, b]) == compute_fingerprint([b, a])


def test_fingerprint_walks_directories_recursively(tmp_path: Path) -> None:
    d = tmp_path / "data"
    (d / "nested").mkdir(parents=True)
    (d / "top.txt").write_bytes(b"top")
    (d / "nested" / "deep.txt").write_bytes(b"deep")

    digest_first = compute_fingerprint([d])
    # Touching a deep file changes the digest.
    (d / "nested" / "deep.txt").write_bytes(b"changed")
    assert compute_fingerprint([d]) != digest_first


def test_fingerprint_distinguishes_renamed_files(tmp_path: Path) -> None:
    """Identical bytes under different names → different fingerprints.

    Catches a subtle bug class: if we hashed only file contents (without
    the relative path), renaming ``train.csv`` → ``test.csv`` would
    silently match the original ``train.csv`` digest.
    """
    a = tmp_path / "train.csv"
    a.write_bytes(b"same")
    digest = compute_fingerprint([a])
    a.rename(tmp_path / "test.csv")
    assert compute_fingerprint([tmp_path / "test.csv"]) != digest


def test_missing_path_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        compute_fingerprint([tmp_path / "does-not-exist"])


def test_hash_file_matches_known_sha256(tmp_path: Path) -> None:
    """Sanity: streaming chunked hash matches the known hex of `b'hello'`."""
    f = tmp_path / "f"
    f.write_bytes(b"hello")
    # echo -n hello | shasum -a 256
    assert hash_file(f) == "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
