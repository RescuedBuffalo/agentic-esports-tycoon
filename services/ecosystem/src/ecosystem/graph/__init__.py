"""Heterogeneous graph schema + export pipeline (BUF-53, Systems-spec System 08).

The full graph the GNN trains on is a four-node, four-edge ``HeteroData``:

* Nodes: ``player``, ``team``, ``patch``, ``agent``.
* Edges: ``(player, plays_for, team)``, ``(team, relates_to, team)``,
  ``(team, sponsored_by, team)``, ``(patch, affects, agent)``.

A snapshot is built per patch era. The builder pulls rows from a
:class:`~ecosystem.graph.source.GraphDataSource`, the normalisers in
:mod:`ecosystem.graph.normalize` rescale every column to the unit
interval (acceptance: "no raw un-scaled columns"), the validator in
:mod:`ecosystem.graph.validate` runs the System-09 structural checks,
and :mod:`ecosystem.graph.export` registers the run in the BUF-69
registry and writes ``snapshot.npz`` plus ``manifest.json`` under
``runs/{run_id}/``.

Public API::

    from ecosystem.graph import GraphSnapshot, build_snapshot, export_era
    from ecosystem.graph.source import InMemoryDataSource

    snapshot = build_snapshot(source, era_slug="e2024_01")
    snapshot.to_pyg()  # lazy torch_geometric.HeteroData adapter
"""

from ecosystem.graph.builder import build_snapshot
from ecosystem.graph.export import export_era
from ecosystem.graph.schema import (
    EDGE_TYPES,
    NODE_TYPES,
    EdgeSpec,
    FeatureColumn,
    FeatureGroup,
    NodeSpec,
    SchemaError,
    edge_spec,
    node_spec,
)
from ecosystem.graph.snapshot import EdgeBlock, GraphSnapshot, NodeBlock
from ecosystem.graph.source import GraphDataSource, InMemoryDataSource
from ecosystem.graph.validate import (
    StructuralValidationError,
    ValidationReport,
    validate_snapshot,
)

__all__ = [
    "EDGE_TYPES",
    "EdgeBlock",
    "EdgeSpec",
    "FeatureColumn",
    "FeatureGroup",
    "GraphDataSource",
    "GraphSnapshot",
    "InMemoryDataSource",
    "NODE_TYPES",
    "NodeBlock",
    "NodeSpec",
    "SchemaError",
    "StructuralValidationError",
    "ValidationReport",
    "build_snapshot",
    "edge_spec",
    "export_era",
    "node_spec",
    "validate_snapshot",
]
