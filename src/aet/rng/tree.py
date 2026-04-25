"""Hierarchical splittable RNG.

The simulation needs reproducible randomness across many concurrent consumers
(matches, agents, training jobs, ...). Threading a single ``np.random.Generator``
through everything makes draw order significant and is fragile. Instead, every
consumer asks the tree for a child node identified by a stable path; each node
produces an independent, deterministic stream.

Two trees built from the same root seed and walked to the same path always
return the same Generator output, regardless of the order in which other
siblings are visited.
"""

from __future__ import annotations

import hashlib

import numpy as np

_SEP = "/"
_ENTROPY_BYTES = 32


def _hash(parent: bytes, label: str) -> bytes:
    return hashlib.blake2b(
        parent + b"\x00" + label.encode("utf-8"),
        digest_size=_ENTROPY_BYTES,
    ).digest()


def _entropy_from_int(seed: int) -> bytes:
    if seed < 0:
        raise ValueError("seed must be non-negative")
    raw = seed.to_bytes((seed.bit_length() + 7) // 8 or 1, "big")
    return hashlib.blake2b(raw, digest_size=_ENTROPY_BYTES).digest()


class RngTree:
    """Hierarchical splittable RNG keyed by stable path labels.

    Examples
    --------
    >>> root = RngTree(seed=42)
    >>> a = root.child("matches").child("m001")
    >>> b = root.at("matches/m001")
    >>> int(a.generator().integers(0, 1_000_000)) == int(b.generator().integers(0, 1_000_000))
    True
    """

    __slots__ = ("_entropy", "_path", "_children")

    def __init__(
        self,
        seed: int | bytes | None = None,
        *,
        _entropy: bytes | None = None,
        _path: tuple[str, ...] = (),
    ) -> None:
        if _entropy is not None:
            if len(_entropy) != _ENTROPY_BYTES:
                raise ValueError(f"_entropy must be exactly {_ENTROPY_BYTES} bytes")
            self._entropy = _entropy
        elif isinstance(seed, (bytes, bytearray)):
            if len(seed) != _ENTROPY_BYTES:
                raise ValueError(f"byte seed must be exactly {_ENTROPY_BYTES} bytes")
            self._entropy = bytes(seed)
        elif isinstance(seed, int) and not isinstance(seed, bool):
            self._entropy = _entropy_from_int(seed)
        elif seed is None:
            raise ValueError("seed must be provided for the root node")
        else:
            raise TypeError(f"unsupported seed type: {type(seed).__name__}")
        self._path = _path
        self._children: dict[str, RngTree] = {}

    @property
    def path(self) -> tuple[str, ...]:
        return self._path

    def child(self, label: str) -> RngTree:
        if not label or _SEP in label:
            raise ValueError(f"invalid label: {label!r}")
        cached = self._children.get(label)
        if cached is not None:
            return cached
        node = RngTree(
            _entropy=_hash(self._entropy, label),
            _path=self._path + (label,),
        )
        self._children[label] = node
        return node

    def at(self, path: str) -> RngTree:
        """Walk a slash-separated path. Empty segments are ignored."""
        node = self
        for label in path.split(_SEP):
            if not label:
                continue
            node = node.child(label)
        return node

    def generator(self) -> np.random.Generator:
        """Return a fresh Generator seeded from this node's entropy.

        Calling ``generator()`` twice on the same node yields two Generators
        with the same starting state. Long-running consumers should call this
        once and reuse the result.
        """
        seed_int = int.from_bytes(self._entropy, "big")
        return np.random.default_rng(np.random.SeedSequence(seed_int))

    def __repr__(self) -> str:
        printable = "/".join(self._path) if self._path else "<root>"
        return f"RngTree(path={printable})"
