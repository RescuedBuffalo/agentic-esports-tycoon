"""GraphSnapshot — the in-memory, framework-agnostic HeteroData payload.

The snapshot is what the builder produces and what the validator and
the registry write to disk. It is intentionally numpy-only so that:

* Tests don't need ``torch`` / ``torch_geometric`` on the import path
  (CI image stays small).
* The serialised ``snapshot.npz`` is a stable, language-portable
  artifact — a downstream Rust trainer or a Jupyter notebook can load
  it without the PyG runtime.

A trainer that does want a real :class:`torch_geometric.data.HeteroData`
calls :meth:`GraphSnapshot.to_pyg`, which lazily imports torch /
torch_geometric. ``to_pyg`` is the one place the library boundary
lives; everything upstream is numpy.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from ecosystem.graph.schema import EDGE_TYPES, NODE_TYPES, SCHEMA_VERSION

if TYPE_CHECKING:  # pragma: no cover - import-only
    import torch_geometric.data as pyg_data


# Versioning is on the snapshot, not on the schema constants alone, so
# a future format-only change (e.g. switching from npz to safetensors)
# can bump the snapshot version while leaving the schema untouched.
SNAPSHOT_FORMAT_VERSION = "1.0.0"

# Default float precision for feature matrices. fp32 is the trainer's
# native dtype; emitting fp64 here would double the snapshot size on
# disk for no model-side benefit.
_FLOAT_DTYPE = np.float32

# Edge indices are int64 to match torch_geometric.HeteroData's contract
# (torch.long). int32 would save bytes but force a cast on every load.
_INDEX_DTYPE = np.int64


@dataclass(slots=True)
class NodeBlock:
    """One heterogeneous node type's data on a snapshot.

    ``ids`` holds the canonical entity ids (UUID hex strings or any
    stable string). ``x`` is the dense feature matrix; row ``i``
    describes node ``ids[i]``. ``column_names`` is the manifest the
    schema declared for this node type at build time — recorded on the
    snapshot so a later trainer never has to look up the schema by
    column index.
    """

    node_type: str
    ids: list[str]
    x: np.ndarray
    column_names: tuple[str, ...]

    def __post_init__(self) -> None:
        n = len(self.ids)
        if self.x.shape[0] != n:
            raise ValueError(
                f"NodeBlock[{self.node_type}]: x.shape[0]={self.x.shape[0]} "
                f"but len(ids)={n}"
            )
        if self.x.shape[1] != len(self.column_names):
            raise ValueError(
                f"NodeBlock[{self.node_type}]: x.shape[1]={self.x.shape[1]} "
                f"but len(column_names)={len(self.column_names)}"
            )
        if self.x.dtype != _FLOAT_DTYPE:
            # Tolerate fp64 from numpy.array(...) inferences by casting
            # on the way in. A stronger error here would break naive
            # callers without buying anything.
            self.x = self.x.astype(_FLOAT_DTYPE, copy=False)

    @property
    def num_nodes(self) -> int:
        return len(self.ids)

    @property
    def feature_dim(self) -> int:
        return int(self.x.shape[1])


@dataclass(slots=True)
class EdgeBlock:
    """One heterogeneous edge type's data on a snapshot.

    ``edge_index`` is shape ``(2, E)`` with row 0 = source node-local
    index, row 1 = destination node-local index. ``edge_attr`` is shape
    ``(E, K)`` or ``None`` when the schema declares no attributes.
    """

    src: str
    relation: str
    dst: str
    edge_index: np.ndarray
    edge_attr: np.ndarray | None
    edge_attr_columns: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.edge_index.ndim != 2 or self.edge_index.shape[0] != 2:
            raise ValueError(
                f"EdgeBlock[{self.key}]: edge_index must be shape (2, E), "
                f"got {self.edge_index.shape}"
            )
        if self.edge_index.dtype != _INDEX_DTYPE:
            self.edge_index = self.edge_index.astype(_INDEX_DTYPE, copy=False)
        if self.edge_attr is not None:
            e = self.edge_index.shape[1]
            if self.edge_attr.shape[0] != e:
                raise ValueError(
                    f"EdgeBlock[{self.key}]: edge_attr.shape[0]={self.edge_attr.shape[0]} "
                    f"!= edge_index.shape[1]={e}"
                )
            if self.edge_attr.shape[1] != len(self.edge_attr_columns):
                raise ValueError(
                    f"EdgeBlock[{self.key}]: edge_attr.shape[1]={self.edge_attr.shape[1]} "
                    f"!= len(edge_attr_columns)={len(self.edge_attr_columns)}"
                )
            if self.edge_attr.dtype != _FLOAT_DTYPE:
                self.edge_attr = self.edge_attr.astype(_FLOAT_DTYPE, copy=False)

    @property
    def key(self) -> tuple[str, str, str]:
        return (self.src, self.relation, self.dst)

    @property
    def num_edges(self) -> int:
        return int(self.edge_index.shape[1])


@dataclass(slots=True)
class GraphSnapshot:
    """A built, validation-ready heterogeneous graph for one era.

    The snapshot is opaque to the validator and the writer — they go
    through :meth:`nodes`/:meth:`edges` rather than the underlying
    dicts. That makes adding a new node type a one-line dict
    addition, not a multi-touch refactor.
    """

    era_slug: str
    schema_version: str = SCHEMA_VERSION
    snapshot_format_version: str = SNAPSHOT_FORMAT_VERSION
    node_blocks: dict[str, NodeBlock] = field(default_factory=dict)
    edge_blocks: dict[tuple[str, str, str], EdgeBlock] = field(default_factory=dict)
    # Free-form metadata: era window, build timestamp, normaliser
    # parameters keyed by (node_type, column). Persisted verbatim into
    # ``manifest.json`` so a later trainer can re-apply normalisation
    # to a held-out evaluation row without re-running the pipeline.
    metadata: dict[str, Any] = field(default_factory=dict)

    # ---- accessors -------------------------------------------------------

    def nodes(self, node_type: str) -> NodeBlock:
        if node_type not in self.node_blocks:
            raise KeyError(f"snapshot has no node block for {node_type!r}")
        return self.node_blocks[node_type]

    def edges(self, key: tuple[str, str, str]) -> EdgeBlock:
        if key not in self.edge_blocks:
            raise KeyError(f"snapshot has no edge block for {key!r}")
        return self.edge_blocks[key]

    def node_types(self) -> tuple[str, ...]:
        return tuple(self.node_blocks)

    def edge_types(self) -> tuple[tuple[str, str, str], ...]:
        return tuple(self.edge_blocks)

    # ---- summaries -------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """Compact dict for log lines and the run manifest."""
        return {
            "era_slug": self.era_slug,
            "schema_version": self.schema_version,
            "snapshot_format_version": self.snapshot_format_version,
            "node_counts": {
                nt: blk.num_nodes for nt, blk in self.node_blocks.items()
            },
            "node_feature_dims": {
                nt: blk.feature_dim for nt, blk in self.node_blocks.items()
            },
            "edge_counts": {
                _edge_key_str(k): blk.num_edges
                for k, blk in self.edge_blocks.items()
            },
        }

    # ---- IO --------------------------------------------------------------

    def write(self, run_dir: Path) -> dict[str, Path]:
        """Write ``snapshot.npz`` + ``manifest.json`` under ``run_dir``.

        Returns a dict of {role: path} for the caller's convenience —
        the registry stamps these into the run record's notes.
        """
        run_dir.mkdir(parents=True, exist_ok=True)
        npz_path = run_dir / "snapshot.npz"
        manifest_path = run_dir / "manifest.json"

        arrays = self._to_npz_dict()
        # ``np.savez`` is platform-portable and decompresses lazily;
        # gzip-compressing the matrices would shave ~30% off disk but
        # add a load-time cost we don't need under the 2-minutes-per-
        # era acceptance budget. The numpy stub for ``savez`` types
        # ``**kwds`` as ``bool`` (likely a stale stub picking up
        # ``allow_pickle``); the runtime accepts ``str -> ndarray``
        # cleanly, which is the documented use.
        np.savez(npz_path, **arrays)  # type: ignore[arg-type]
        manifest_path.write_text(
            json.dumps(
                self._build_manifest(),
                indent=2,
                sort_keys=True,
                default=_jsonable_default,
            ),
            encoding="utf-8",
        )
        return {"snapshot": npz_path, "manifest": manifest_path}

    @classmethod
    def read(cls, run_dir: Path) -> GraphSnapshot:
        """Round-trip counterpart to :meth:`write`."""
        manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
        with np.load(run_dir / "snapshot.npz", allow_pickle=False) as npz:
            arrays = {k: npz[k].copy() for k in npz.files}
        return cls._from_arrays_and_manifest(arrays, manifest)

    # ---- internals -------------------------------------------------------

    def _to_npz_dict(self) -> dict[str, np.ndarray]:
        """Pack all blocks into a flat dict of (str → ndarray) for ``savez``.

        Keys are namespaced so a bug that picked the wrong dtype in
        one block can't silently overwrite another. ``ids`` are stored
        as fixed-width unicode arrays — they're identifiers, not
        floats, and roundtrip cleanly through npz.
        """
        out: dict[str, np.ndarray] = {}
        for nt, node_blk in self.node_blocks.items():
            out[f"node:{nt}:x"] = node_blk.x
            out[f"node:{nt}:ids"] = np.asarray(node_blk.ids, dtype=np.str_)
        for k, edge_blk in self.edge_blocks.items():
            tag = _edge_key_str(k)
            out[f"edge:{tag}:edge_index"] = edge_blk.edge_index
            if edge_blk.edge_attr is not None:
                out[f"edge:{tag}:edge_attr"] = edge_blk.edge_attr
        return out

    def _build_manifest(self) -> dict[str, Any]:
        return {
            "era_slug": self.era_slug,
            "schema_version": self.schema_version,
            "snapshot_format_version": self.snapshot_format_version,
            "summary": self.summary(),
            "node_columns": {
                nt: list(blk.column_names) for nt, blk in self.node_blocks.items()
            },
            "edge_attr_columns": {
                _edge_key_str(k): list(blk.edge_attr_columns)
                for k, blk in self.edge_blocks.items()
            },
            "metadata": self.metadata,
        }

    @classmethod
    def _from_arrays_and_manifest(
        cls,
        arrays: dict[str, np.ndarray],
        manifest: dict[str, Any],
    ) -> GraphSnapshot:
        node_blocks: dict[str, NodeBlock] = {}
        edge_blocks: dict[tuple[str, str, str], EdgeBlock] = {}

        node_columns = manifest.get("node_columns", {})
        for nt, cols in node_columns.items():
            x = arrays[f"node:{nt}:x"]
            ids = arrays[f"node:{nt}:ids"].tolist()
            node_blocks[nt] = NodeBlock(
                node_type=nt,
                ids=[str(i) for i in ids],
                x=x,
                column_names=tuple(cols),
            )

        edge_attr_columns = manifest.get("edge_attr_columns", {})
        for tag, cols in edge_attr_columns.items():
            key = _edge_key_from_str(tag)
            edge_index = arrays[f"edge:{tag}:edge_index"]
            edge_attr = arrays.get(f"edge:{tag}:edge_attr")
            edge_blocks[key] = EdgeBlock(
                src=key[0],
                relation=key[1],
                dst=key[2],
                edge_index=edge_index,
                edge_attr=edge_attr,
                edge_attr_columns=tuple(cols),
            )

        return cls(
            era_slug=manifest["era_slug"],
            schema_version=manifest["schema_version"],
            snapshot_format_version=manifest["snapshot_format_version"],
            node_blocks=node_blocks,
            edge_blocks=edge_blocks,
            metadata=manifest.get("metadata", {}),
        )

    # ---- adapters --------------------------------------------------------

    def to_pyg(self) -> pyg_data.HeteroData:
        """Return a :class:`torch_geometric.data.HeteroData` view.

        Lazy import: ``torch_geometric`` is *only* required when a
        consumer calls this method. Unit tests and the export pipeline
        itself stay numpy-only.
        """
        # Local import keeps torch_geometric out of the cold-import
        # path; the trainer that needs this already pays the import
        # cost on its own boot.
        import torch  # noqa: PLC0415  (lazy by design)
        from torch_geometric.data import HeteroData  # noqa: PLC0415

        data = HeteroData()
        for nt, node_blk in self.node_blocks.items():
            data[nt].x = torch.from_numpy(node_blk.x)
            data[nt].num_nodes = node_blk.num_nodes
            # PyG accepts arbitrary attributes; keep ids around for
            # debugging-level joins.
            data[nt].ids = list(node_blk.ids)
        for k, edge_blk in self.edge_blocks.items():
            data[k].edge_index = torch.from_numpy(edge_blk.edge_index)
            if edge_blk.edge_attr is not None:
                data[k].edge_attr = torch.from_numpy(edge_blk.edge_attr)
        return data


# ---- helpers ---------------------------------------------------------------


def _jsonable_default(obj: Any) -> Any:
    """Coerce non-native types (datetime, numpy scalars, ...) for ``json.dumps``.

    The snapshot's metadata block carries whatever the data source
    handed back from ``patch_meta`` — typically era window timestamps
    as ``datetime`` instances. Without this default, ``json.dumps``
    raises ``TypeError`` when the manifest is written.

    The handler is deliberately permissive: anything ISO-stringifiable
    is converted, anything else falls through to ``str()``. The
    manifest is a human-readable record, not a typed contract — losing
    a millisecond of precision on a stored timestamp is a fair price
    for not failing the export.
    """
    # ``datetime`` and ``date`` carry ``.isoformat()``; numpy datetimes
    # do not, but ``str(np.datetime64(...))`` returns ISO-like text.
    iso = getattr(obj, "isoformat", None)
    if callable(iso):
        return iso()
    # numpy scalars implement ``.item()`` to fall back to a Python type.
    item = getattr(obj, "item", None)
    if callable(item):
        try:
            return item()
        except (TypeError, ValueError):
            pass
    return str(obj)


def _edge_key_str(key: tuple[str, str, str]) -> str:
    """Stable tag used in npz file keys.

    The ``__`` separator avoids collision with ``_`` in node/edge type
    names ("plays_for", "patch_era") and roundtrips cleanly through
    :func:`_edge_key_from_str`.
    """
    return "__".join(key)


def _edge_key_from_str(tag: str) -> tuple[str, str, str]:
    parts = tag.split("__")
    if len(parts) != 3:
        raise ValueError(f"edge tag {tag!r} did not split into 3 parts on '__'")
    return parts[0], parts[1], parts[2]


# ``_assert_schema_known`` is a tiny self-test the validator can call
# to detect manifests built against an obviously-broken schema (e.g. a
# downstream rename that the runtime didn't pick up). It lives here
# because the snapshot owns the schema contract.
def assert_schema_known(snapshot: GraphSnapshot) -> None:
    """Raise ValueError if a node/edge type isn't in the runtime schema."""
    for nt in snapshot.node_blocks:
        if nt not in NODE_TYPES:
            raise ValueError(
                f"snapshot carries unknown node type {nt!r}; runtime knows "
                f"{sorted(NODE_TYPES)}"
            )
    for k in snapshot.edge_blocks:
        if k not in EDGE_TYPES:
            raise ValueError(
                f"snapshot carries unknown edge type {k!r}; runtime knows "
                f"{sorted(EDGE_TYPES)}"
            )


__all__ = [
    "EdgeBlock",
    "GraphSnapshot",
    "NodeBlock",
    "SNAPSHOT_FORMAT_VERSION",
    "assert_schema_known",
]
