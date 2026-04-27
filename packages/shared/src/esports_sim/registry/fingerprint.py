"""Content fingerprints for registry input data.

The fingerprint is the second half of the registry's idempotency key
(the first half is the config-file hash). When the same config runs
against the same data twice, the fingerprints match and the registry
returns the prior ``run_id`` instead of minting a new one.

Algorithm:

1. Walk every input path; for each file, compute ``(label, sha256(bytes))``
   where ``label`` is the basename (single file) or the directory-relative
   POSIX path (directory walk).
2. Sort the resulting list by ``(label, sha256)``.
3. Fold the **input manifest** (sorted basenames of the top-level
   inputs) into the rolling hash first, then the sorted file-content
   pairs. The manifest prefix is what makes
   ``compute_fingerprint([empty_dir])`` distinct from
   ``compute_fingerprint([])`` — without it both walked to zero files
   and produced the empty-string sentinel, collapsing the natural-key
   for "explicitly empty dataset" onto "no data provided at all".

The sort key for files is ``(label, sha256)`` — *not* just ``label`` —
so two distinct files with the same basename (``/mnt/a/data.csv`` and
``/mnt/b/data.csv``, or two directories both named ``shards``) produce
a canonical, content-derived ordering. Sorting by label alone would
leave duplicates in caller-provided order, breaking permutation
invariance.

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
    """Expand directories, return ``(label, path)`` pairs in caller order.

    The ``label`` is what gets fed into the rolling hash — for a plain
    file it's just the basename, for a directory it's the
    directory-relative POSIX path. We do *not* sort here; the caller
    sorts later by ``(label, file_hash)`` so duplicate labels (same
    basename across distinct inputs) are broken stably by content.
    """
    flat: list[tuple[str, Path]] = []
    for raw in paths:
        p = Path(raw)
        if not p.exists():
            raise FileNotFoundError(p)
        if p.is_dir():
            # rglob order is filesystem-dependent — sort the *intra-input*
            # walk so the relative labels are deterministic. Inter-input
            # ordering is handled by the (label, hash) sort below.
            for child in sorted(p.rglob("*")):
                if child.is_file():
                    rel = child.relative_to(p).as_posix()
                    flat.append((f"{p.name}/{rel}", child))
        else:
            flat.append((p.name, p))
    return flat


def compute_fingerprint(paths: Iterable[Path | str]) -> str:
    """Return a deterministic SHA-256 over the contents of *paths*.

    * Empty input list (``compute_fingerprint([])``) → empty string. The
      registry uses ``""`` as the "no data provided" sentinel; this is
      reserved for callers who genuinely have no data dimension to fold
      into the natural key.
    * Any non-empty input list → 64-character hex digest, even when the
      walk yields zero files (e.g. the caller passed an empty directory).
      An empty dataset is *different* from no dataset; collapsing them
      onto the same fingerprint would let an explicitly-empty dataset
      reuse a prior run_id that was registered with no data at all.

    Stable across runs as long as the bytes are unchanged. Caller is
    responsible for picking ``paths`` that meaningfully describe the run
    inputs — this function makes no judgement about which files matter.

    Permutation invariance: ``compute_fingerprint([a, b])`` always equals
    ``compute_fingerprint([b, a])``, even when ``a`` and ``b`` share a
    basename, because we sort by ``(label, sha256)`` before rolling.
    """
    paths_list = [Path(p) for p in paths]
    if not paths_list:
        # Reserved sentinel for "no data provided at all". An empty
        # *directory* still produces a real digest — see below.
        return ""

    files = _iter_files(paths_list)

    # Hash up front so the sort can use the file digest as the tie-breaker
    # for label collisions. Without this, two files named ``data.csv``
    # under different parents would sort by stable Python order — which
    # depends on the caller-supplied input order, breaking permutation
    # invariance.
    hashed: list[tuple[str, str]] = [(label, hash_file(path)) for label, path in files]
    hashed.sort()

    rolling = hashlib.sha256()

    # Input manifest: the *basenames* of the top-level inputs, sorted.
    # This is what gives ``compute_fingerprint([empty_dir])`` a non-empty,
    # content-derived digest (and one that's distinct from the
    # ``compute_fingerprint([])`` sentinel). The "INPUT:" prefix
    # disambiguates the manifest section from the file section that
    # follows so a path basename can never collide with a file label.
    for label in sorted(p.name for p in paths_list):
        rolling.update(b"INPUT:")
        rolling.update(label.encode("utf-8"))
        rolling.update(b"\x00")

    for label, file_hash in hashed:
        rolling.update(b"FILE:")
        rolling.update(label.encode("utf-8"))
        rolling.update(b"\x00")
        rolling.update(file_hash.encode("ascii"))
        rolling.update(b"\x00")
    return rolling.hexdigest()
