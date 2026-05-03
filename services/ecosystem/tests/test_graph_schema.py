"""Schema-shape contract tests for the graph exporter (BUF-53).

These pin the declared node + edge inventory so a one-line refactor in
:mod:`ecosystem.graph.schema` doesn't quietly change what the trainer
expects to load.
"""

from __future__ import annotations

import pytest
from ecosystem.graph.schema import (
    EDGE_TYPES,
    NODE_TYPES,
    SchemaError,
    edge_spec,
    node_spec,
)

# ---- node-type inventory ---------------------------------------------------


def test_node_types_match_systems_spec() -> None:
    """Issue mandates feature matrices for player / team / patch (+ agent for affects edge)."""
    assert set(NODE_TYPES) == {"player", "team", "patch", "agent"}


@pytest.mark.parametrize("node_type", sorted(NODE_TYPES))
def test_each_node_type_has_all_four_feature_groups(node_type: str) -> None:
    """Acceptance: acs_norm, derived, inferred, context — every node type."""
    spec = NODE_TYPES[node_type]
    group_names = {g.name for g in spec.groups}
    assert group_names == {"acs_norm", "derived", "inferred", "context"}


def test_node_spec_lookup_round_trips() -> None:
    spec = node_spec("player")
    assert spec.name == "player"
    assert spec.feature_dim() == sum(len(g.columns) for g in spec.groups)


def test_node_spec_unknown_raises() -> None:
    with pytest.raises(SchemaError):
        node_spec("not_a_real_node_type")


# ---- edge-type inventory ---------------------------------------------------


def test_edge_types_match_issue_scope() -> None:
    """Issue mandates: plays_for, relates_to, sponsored_by, affects (patch→agent)."""
    assert set(EDGE_TYPES) == {
        ("player", "plays_for", "team"),
        ("team", "relates_to", "team"),
        ("team", "sponsored_by", "team"),
        ("patch", "affects", "agent"),
    }


def test_affects_edge_runs_patch_to_agent() -> None:
    """The acceptance pin from the issue body."""
    spec = edge_spec(("patch", "affects", "agent"))
    assert spec.src == "patch"
    assert spec.relation == "affects"
    assert spec.dst == "agent"


def test_plays_for_edge_is_time_bounded() -> None:
    """Roster movement is the canonical time-bounded edge."""
    assert edge_spec(("player", "plays_for", "team")).time_bounded is True


def test_edge_spec_unknown_raises() -> None:
    with pytest.raises(SchemaError):
        edge_spec(("does", "not", "exist"))


# ---- column-level invariants ----------------------------------------------


def test_every_column_has_a_normalizer_name() -> None:
    """Acceptance: no raw un-scaled columns. Every column declares one."""
    for spec in NODE_TYPES.values():
        for col in spec.all_columns():
            assert col.normalizer
    for spec in EDGE_TYPES.values():
        for col in spec.edge_attr_columns:
            assert col.normalizer


def test_invalid_fill_policy_rejected() -> None:
    from ecosystem.graph.schema import FeatureColumn

    with pytest.raises(SchemaError):
        FeatureColumn(name="x", group="acs_norm", normalizer="minmax", fill_policy="invalid")
