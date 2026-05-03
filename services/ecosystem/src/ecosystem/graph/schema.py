"""HeteroData schema declarations (Systems-spec System 08).

The schema is the contract between the data pipeline and every consumer
of the exported graph (the GNN trainer, the validator, the snapshot
diffing tools). Changing it is a versioned event: bump
:data:`SCHEMA_VERSION` and add a migration note so an older snapshot
doesn't get loaded into a newer trainer with silently-shifted columns.

Feature columns are grouped into four named blocks per node type:

* ``acs_norm`` — primary headline rate stats (ACS, KAST, ADR, ...)
  rescaled into ``[0, 1]`` against the era's distribution.
* ``derived`` — features computed from raw stats by the
  ``config/derivation.yaml`` stage (first-blood rate, agent-specific
  metrics, economy efficiency).
* ``inferred`` — model-emitted estimates not present in the source
  data (synergy embedding magnitude, role-fit score, latent skill).
* ``context`` — slow-moving structural attributes (region one-hot,
  tier rank, age in days, era ordinal).

Each :class:`FeatureColumn` carries its dtype and a fill policy used
when the source row is missing the column. The builder concatenates
the four groups in declaration order; the manifest carried with each
snapshot records which column ended up at which index so a downstream
trainer never has to guess by position.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

# Bump this when the column list, edge schema, or feature block ordering
# changes. The validator refuses to load a snapshot whose manifest
# version doesn't match the runtime; that's the early-warning that two
# pipeline runs produced incompatible artifacts.
SCHEMA_VERSION = "1.0.0"


class SchemaError(ValueError):
    """Raised when a schema lookup hits an unknown name."""


FeatureGroupName = Literal["acs_norm", "derived", "inferred", "context"]
FILL_POLICIES = frozenset({"zero", "mean", "drop_node"})


@dataclass(frozen=True, slots=True)
class FeatureColumn:
    """One scalar feature column on a node-type's feature matrix.

    ``normalizer`` is the name of a transform registered in
    :mod:`ecosystem.graph.normalize`; the builder looks it up rather
    than embedding a callable here so the schema stays a pure data
    description.

    ``fill_policy`` decides what the builder does when the source row
    is missing the underlying value:

    * ``"zero"`` — emit ``0.0`` (post-normalisation neutral). Most
      stat columns use this since 0 is a safe baseline.
    * ``"mean"`` — emit the era-mean from the normaliser. Use for
      columns where 0 is a meaningful low (e.g. ratings).
    * ``"drop_node"`` — refuse to emit the node at all. Use for
      identity-critical columns (e.g. ``team`` membership) where the
      node would corrupt downstream queries if its row were zero.
    """

    name: str
    group: FeatureGroupName
    normalizer: str
    fill_policy: str = "zero"
    description: str = ""

    def __post_init__(self) -> None:
        if self.fill_policy not in FILL_POLICIES:
            raise SchemaError(
                f"FeatureColumn {self.name!r}: fill_policy={self.fill_policy!r} "
                f"not in {sorted(FILL_POLICIES)}"
            )


@dataclass(frozen=True, slots=True)
class FeatureGroup:
    """A named block of feature columns within a node-type's feature matrix."""

    name: FeatureGroupName
    columns: tuple[FeatureColumn, ...]


@dataclass(frozen=True, slots=True)
class NodeSpec:
    """Schema for one heterogeneous node type."""

    name: str
    groups: tuple[FeatureGroup, ...]

    def all_columns(self) -> tuple[FeatureColumn, ...]:
        """Flatten all groups in declaration order."""
        return tuple(col for grp in self.groups for col in grp.columns)

    def feature_dim(self) -> int:
        """Total width of the ``x`` matrix for this node type."""
        return sum(len(g.columns) for g in self.groups)

    def column_names(self) -> tuple[str, ...]:
        """Flat list of column names — consumed by the manifest writer."""
        return tuple(col.name for col in self.all_columns())


@dataclass(frozen=True, slots=True)
class EdgeSpec:
    """One heterogeneous edge type.

    The PyG-conventional canonical key is ``(src, relation, dst)``.
    ``edge_attr_columns`` is the (optional) list of scalar attributes
    carried per edge — the builder stacks these into ``edge_attr`` on
    the snapshot the same way it builds node features.
    ``time_bounded`` means each edge carries ``valid_from`` /
    ``valid_to`` columns that the era filter respects.
    """

    src: str
    relation: str
    dst: str
    edge_attr_columns: tuple[FeatureColumn, ...] = field(default_factory=tuple)
    time_bounded: bool = False

    @property
    def key(self) -> tuple[str, str, str]:
        """PyG-style canonical triple, also used as the dict key on snapshots."""
        return (self.src, self.relation, self.dst)


# ---- node specs ------------------------------------------------------------
#
# The exact column inventory is derived from
# Systems-spec System 08 §node_features and from the headline stats VLR
# already exposes in MapResult.team{1,2}_stats. Adding a column here is
# a schema-version bump (see SCHEMA_VERSION above).

_PLAYER = NodeSpec(
    name="player",
    groups=(
        FeatureGroup(
            name="acs_norm",
            columns=(
                FeatureColumn("acs", "acs_norm", "minmax", description="combat score / round"),
                FeatureColumn(
                    "kast",
                    "acs_norm",
                    "minmax",
                    description="kill/assist/survive/trade %",
                ),
                FeatureColumn("adr", "acs_norm", "minmax", description="damage / round"),
                FeatureColumn("hs_pct", "acs_norm", "minmax", description="headshot %"),
                FeatureColumn("rating", "acs_norm", "minmax", description="VLR composite rating"),
            ),
        ),
        FeatureGroup(
            name="derived",
            columns=(
                FeatureColumn("fb_rate", "derived", "minmax", description="first-blood rate"),
                FeatureColumn("fd_rate", "derived", "minmax", description="first-death rate"),
                FeatureColumn("clutch_rate", "derived", "minmax"),
                FeatureColumn("multikill_rate", "derived", "minmax"),
            ),
        ),
        FeatureGroup(
            name="inferred",
            columns=(
                FeatureColumn(
                    "latent_skill",
                    "inferred",
                    "zscore",
                    description="bayesian skill estimate",
                ),
                FeatureColumn("role_fit", "inferred", "minmax"),
                FeatureColumn("synergy_mag", "inferred", "minmax"),
            ),
        ),
        FeatureGroup(
            name="context",
            columns=(
                FeatureColumn("age_days", "context", "log1p_minmax"),
                FeatureColumn("region_americas", "context", "passthrough"),
                FeatureColumn("region_emea", "context", "passthrough"),
                FeatureColumn("region_pacific", "context", "passthrough"),
                FeatureColumn("region_china", "context", "passthrough"),
                FeatureColumn("tier_rank", "context", "minmax"),
            ),
        ),
    ),
)


_TEAM = NodeSpec(
    name="team",
    groups=(
        FeatureGroup(
            name="acs_norm",
            columns=(
                FeatureColumn("avg_acs", "acs_norm", "minmax"),
                FeatureColumn("avg_kast", "acs_norm", "minmax"),
                FeatureColumn("win_rate", "acs_norm", "minmax"),
                FeatureColumn("map_win_rate", "acs_norm", "minmax"),
            ),
        ),
        FeatureGroup(
            name="derived",
            columns=(
                FeatureColumn(
                    "attack_rwr",
                    "derived",
                    "minmax",
                    description="attack-side round win rate",
                ),
                FeatureColumn("defense_rwr", "derived", "minmax"),
                FeatureColumn("eco_efficiency", "derived", "minmax"),
                FeatureColumn("comeback_rate", "derived", "minmax"),
            ),
        ),
        FeatureGroup(
            name="inferred",
            columns=(
                FeatureColumn("strength_rating", "inferred", "zscore"),
                FeatureColumn("style_aggression", "inferred", "minmax"),
                FeatureColumn("style_utility", "inferred", "minmax"),
            ),
        ),
        FeatureGroup(
            name="context",
            columns=(
                FeatureColumn("region_americas", "context", "passthrough"),
                FeatureColumn("region_emea", "context", "passthrough"),
                FeatureColumn("region_pacific", "context", "passthrough"),
                FeatureColumn("region_china", "context", "passthrough"),
                FeatureColumn("tier_rank", "context", "minmax"),
                FeatureColumn("roster_age_days", "context", "log1p_minmax"),
            ),
        ),
    ),
)


_PATCH = NodeSpec(
    name="patch",
    groups=(
        FeatureGroup(
            name="acs_norm",
            # Patch nodes don't have ACS-style stats; the slot is kept
            # for symmetry with the other node types so a single
            # contiguous matrix layout works in the trainer. Two
            # placeholder summary columns are computed by the builder.
            columns=(
                FeatureColumn("agg_match_acs", "acs_norm", "minmax"),
                FeatureColumn("agg_match_kast", "acs_norm", "minmax"),
            ),
        ),
        FeatureGroup(
            name="derived",
            columns=(
                FeatureColumn("duration_days", "derived", "log1p_minmax"),
                FeatureColumn("matches_played", "derived", "log1p_minmax"),
                FeatureColumn("agents_added", "derived", "minmax"),
                FeatureColumn("maps_added", "derived", "minmax"),
            ),
        ),
        FeatureGroup(
            name="inferred",
            columns=(
                FeatureColumn("meta_magnitude", "inferred", "passthrough"),
                FeatureColumn("is_major_shift", "inferred", "passthrough"),
            ),
        ),
        FeatureGroup(
            name="context",
            columns=(
                FeatureColumn("era_ordinal", "context", "minmax"),
                FeatureColumn("season_year_2024", "context", "passthrough"),
                FeatureColumn("season_year_2025", "context", "passthrough"),
                FeatureColumn("season_year_2026", "context", "passthrough"),
            ),
        ),
    ),
)


_AGENT = NodeSpec(
    name="agent",
    groups=(
        FeatureGroup(
            name="acs_norm",
            # Agent ACS_norm = era-relative pick performance.
            columns=(
                FeatureColumn("avg_acs_when_picked", "acs_norm", "minmax"),
                FeatureColumn("avg_rating_when_picked", "acs_norm", "minmax"),
            ),
        ),
        FeatureGroup(
            name="derived",
            columns=(
                FeatureColumn("pick_rate", "derived", "minmax"),
                FeatureColumn("win_rate_when_picked", "derived", "minmax"),
                FeatureColumn("ban_rate", "derived", "minmax"),
            ),
        ),
        FeatureGroup(
            name="inferred",
            columns=(
                FeatureColumn("strength_score", "inferred", "minmax"),
                FeatureColumn("synergy_index", "inferred", "minmax"),
            ),
        ),
        FeatureGroup(
            name="context",
            columns=(
                FeatureColumn("role_duelist", "context", "passthrough"),
                FeatureColumn("role_controller", "context", "passthrough"),
                FeatureColumn("role_initiator", "context", "passthrough"),
                FeatureColumn("role_sentinel", "context", "passthrough"),
                FeatureColumn("age_days", "context", "log1p_minmax"),
            ),
        ),
    ),
)


NODE_TYPES: dict[str, NodeSpec] = {
    "player": _PLAYER,
    "team": _TEAM,
    "patch": _PATCH,
    "agent": _AGENT,
}


# ---- edge specs ------------------------------------------------------------

_PLAYS_FOR = EdgeSpec(
    src="player",
    relation="plays_for",
    dst="team",
    edge_attr_columns=(
        FeatureColumn("tenure_days", "context", "log1p_minmax"),
        FeatureColumn("role_slot", "context", "minmax"),
    ),
    time_bounded=True,
)

_RELATES_TO = EdgeSpec(
    src="team",
    relation="relates_to",
    dst="team",
    edge_attr_columns=(
        FeatureColumn("rivalry_strength", "inferred", "minmax"),
        FeatureColumn("head_to_head_count", "context", "log1p_minmax"),
    ),
)

_SPONSORED_BY = EdgeSpec(
    src="team",
    relation="sponsored_by",
    dst="team",
    edge_attr_columns=(
        FeatureColumn("annual_value_usd", "context", "log1p_minmax"),
        FeatureColumn("tenure_days", "context", "log1p_minmax"),
    ),
    time_bounded=True,
)

_AFFECTS = EdgeSpec(
    src="patch",
    relation="affects",
    dst="agent",
    edge_attr_columns=(
        FeatureColumn("change_magnitude", "inferred", "minmax"),
        FeatureColumn("buff_direction", "inferred", "passthrough"),
    ),
)


EDGE_TYPES: dict[tuple[str, str, str], EdgeSpec] = {
    spec.key: spec for spec in (_PLAYS_FOR, _RELATES_TO, _SPONSORED_BY, _AFFECTS)
}


def node_spec(name: str) -> NodeSpec:
    try:
        return NODE_TYPES[name]
    except KeyError as e:
        raise SchemaError(f"unknown node type {name!r}; known: {sorted(NODE_TYPES)}") from e


def edge_spec(key: tuple[str, str, str]) -> EdgeSpec:
    try:
        return EDGE_TYPES[key]
    except KeyError as e:
        raise SchemaError(f"unknown edge type {key!r}; known: {sorted(EDGE_TYPES)}") from e


__all__ = [
    "EDGE_TYPES",
    "EdgeSpec",
    "FILL_POLICIES",
    "FeatureColumn",
    "FeatureGroup",
    "FeatureGroupName",
    "NODE_TYPES",
    "NodeSpec",
    "SCHEMA_VERSION",
    "SchemaError",
    "edge_spec",
    "node_spec",
]
