"""Pluggable data sources for the graph builder.

Decoupling the builder from any concrete database (BUF-6 SQLAlchemy,
DuckDB, parquet, fixture dicts) buys two things:

* Tests can construct deterministic graphs from fixtures without a
  Postgres in CI.
* A future migration to a different feature store (parquet on s3, a
  feast online store, ...) is one new ``GraphDataSource`` class — the
  builder doesn't notice.

Source rows are dicts of ``column_name -> value``. Missing columns are
filled by the builder according to each :class:`FeatureColumn`'s
``fill_policy``; sources don't have to know the schema. Edge rows
carry ``src_id`` / ``dst_id`` keyed against the same string ids the
node iterator yields, plus the optional attribute columns the edge
spec declares.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class NodeRow:
    """One node's raw inputs.

    ``id`` is the canonical entity id (UUID hex, slug, anything stable
    for the era). ``features`` is the loose dict of column values; any
    column the schema lists but the dict misses gets the column's
    declared fill policy applied.
    """

    id: str
    features: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class EdgeRow:
    """One edge's raw inputs."""

    src_id: str
    dst_id: str
    attributes: dict[str, Any] = field(default_factory=dict)


class GraphDataSource(Protocol):
    """Read-side interface the builder consumes.

    Implementations are responsible for filtering to the requested era
    *before* yielding rows — the builder treats the iterators as
    authoritative for the era.
    """

    def iter_nodes(self, node_type: str, *, era_slug: str) -> Iterable[NodeRow]:
        """Yield every node of ``node_type`` active in ``era_slug``."""
        ...

    def iter_edges(self, key: tuple[str, str, str], *, era_slug: str) -> Iterable[EdgeRow]:
        """Yield every edge of canonical type ``key`` active in ``era_slug``."""
        ...

    def patch_meta(self, era_slug: str) -> dict[str, Any]:
        """Return era-window metadata: start/end timestamps, magnitude flags.

        Used by the builder to populate the synthetic patch-node row
        and to stamp the snapshot's metadata block.
        """
        ...


# ---- in-memory reference implementation -----------------------------------


@dataclass
class InMemoryDataSource:
    """Fixture-friendly :class:`GraphDataSource` for tests.

    Rows are stored in nested dicts keyed by era slug. The constructor
    is permissive — missing eras yield empty iterators rather than
    raising, since "this era has no edges of this kind" is a legal
    state during the early backfill.
    """

    # era_slug -> node_type -> list[NodeRow]
    nodes: dict[str, dict[str, list[NodeRow]]] = field(default_factory=dict)
    # era_slug -> edge_key -> list[EdgeRow]
    edges: dict[str, dict[tuple[str, str, str], list[EdgeRow]]] = field(default_factory=dict)
    # era_slug -> meta dict
    patches: dict[str, dict[str, Any]] = field(default_factory=dict)

    def iter_nodes(self, node_type: str, *, era_slug: str) -> Iterable[NodeRow]:
        return iter(self.nodes.get(era_slug, {}).get(node_type, ()))

    def iter_edges(self, key: tuple[str, str, str], *, era_slug: str) -> Iterable[EdgeRow]:
        return iter(self.edges.get(era_slug, {}).get(key, ()))

    def patch_meta(self, era_slug: str) -> dict[str, Any]:
        return dict(self.patches.get(era_slug, {}))

    # ---- builder helpers ------------------------------------------------

    def add_node(self, era_slug: str, node_type: str, row: NodeRow) -> None:
        self.nodes.setdefault(era_slug, {}).setdefault(node_type, []).append(row)

    def add_edge(self, era_slug: str, key: tuple[str, str, str], row: EdgeRow) -> None:
        self.edges.setdefault(era_slug, {}).setdefault(key, []).append(row)

    def set_patch(self, era_slug: str, meta: dict[str, Any]) -> None:
        self.patches[era_slug] = dict(meta)


__all__ = [
    "EdgeRow",
    "GraphDataSource",
    "InMemoryDataSource",
    "NodeRow",
]
