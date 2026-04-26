"""Content fingerprints for registry input data.

The fingerprint is the second half of the registry's idempotency key
(the first half is the config-file hash). When the same config runs
against the same data twice, the fingerprints match and the registry
returns the prior ``run_id`` instead of minting a new one.

Algorithm: walk every input path in sorted order, hash each file's bytes
with SHA-256, then hash the concatenation of ``(relative_path, sha256)``
pairs. Stable across machines (no mtime, no inode metadata) and
permutation-invariant on input order.

If callers have a more efficient fingerprint for their input shape
(e.g., a DuckDB row-count + min/max digest for tabular data), they can
compute it themselves and pass it directly to ``Registry.register`` —
this helper is the convenience default for "one or more files/dirs on
disk".
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path

# Streamed in 1 MiB chunks so we don't slurp huge artifacts into memory.
_HASH_CHUNK_BYTES = 1 << 20


def hash_file(path: Path) -> str:
    """SHA-256 hex digest of a single file's bytes."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(_HASH_CHUNK_BYTES):
            h.update(chunk)
    return h.hexdigest()


def _iter_files(paths: Iterable[Path]) -> list[tuple[str, Path]]:
    """Expand directories, return sorted ``(relative_label, path)`` pairs.

    The ``relative_label`` is what gets fed into the rolling hash — for a
    plain file it's just the basename, for a directory it's the
    directory-relative path. Sorting is critical: an unstable iteration
    order would produce a different fingerprint for the same content.
    """
    flat: list[tuple[str, Path]] = []
    for raw in paths:
        p = Path(raw)
        if not p.exists():
            raise FileNotFoundError(p)
        if p.is_dir():
            for child in sorted(p.rglob("*")):
                if child.is_file():
                    rel = child.relative_to(p).as_posix()
                    flat.append((f"{p.name}/{rel}", child))
        else:
            flat.append((p.name, p))
    flat.sort(key=lambda pair: pair[0])
    return flat


def compute_fingerprint(paths: Iterable[Path | str]) -> str:
    """Return a deterministic SHA-256 over the contents of *paths*.

    * Empty input → empty string. The registry uses ``""`` as a sentinel
      for "no data fingerprint, only config matters" so don't conflate
      it with a real digest.
    * Otherwise → 64-character hex digest.

    Stable across runs as long as the bytes are unchanged. Caller is
    responsible for picking ``paths`` that meaningfully describe the run
    inputs — this function makes no judgement about which files matter.
    """
    paths_list = [Path(p) for p in paths]
    if not paths_list:
        return ""

    files = _iter_files(paths_list)
    if not files:
        return ""

    rolling = hashlib.sha256()
    for label, path in files:
        # Hash the label *and* the file digest so two files swapping
        # places (or being renamed) produce different fingerprints.
        rolling.update(label.encode("utf-8"))
        rolling.update(b"\x00")
        rolling.update(hash_file(path).encode("ascii"))
        rolling.update(b"\x00")
    return rolling.hexdigest()
