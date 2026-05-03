"""Builder + validator integration tests on synthetic eras (BUF-53).

The fixture in ``graph_fixtures.py`` produces a deterministic
three-era source; these tests exercise the builder against it and
assert the structural invariants the System-09 validator pins.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from ecosystem.graph import (
    EDGE_TYPES,
    NODE_TYPES,
    GraphSnapshot,
    build_snapshot,
    validate_snapshot,
)
from ecosystem.graph.snapshot import assert_schema_known
from ecosystem.graph.source import EdgeRow, InMemoryDataSource, NodeRow
from graph_fixtures import ERA_SLUGS, build_three_era_source

# ---- shape & determinism --------------------------------------------------


@pytest.fixture(scope="module")
def source() -> InMemoryDataSource:
    return build_three_era_source()


@pytest.fixture(params=list(ERA_SLUGS))
def snapshot(source: InMemoryDataSource, request: pytest.FixtureRequest) -> GraphSnapshot:
    return build_snapshot(source, era_slug=request.param)


def test_every_node_type_present(snapshot: GraphSnapshot) -> None:
    assert set(snapshot.node_types()) == set(NODE_TYPES)


def test_every_edge_type_present(snapshot: GraphSnapshot) -> None:
    assert set(snapshot.edge_types()) == set(EDGE_TYPES)


def test_node_feature_dims_match_schema(snapshot: GraphSnapshot) -> None:
    for nt, spec in NODE_TYPES.items():
        block = snapshot.nodes(nt)
        assert block.feature_dim == spec.feature_dim()
        assert block.column_names == spec.column_names()


def test_builder_is_deterministic(source: InMemoryDataSource) -> None:
    """Same source + same era → byte-identical x matrices."""
    a = build_snapshot(source, era_slug="e2024_01")
    b = build_snapshot(source, era_slug="e2024_01")
    for nt in a.node_types():
        np.testing.assert_array_equal(a.nodes(nt).x, b.nodes(nt).x)
        assert a.nodes(nt).ids == b.nodes(nt).ids
    for k in a.edge_types():
        np.testing.assert_array_equal(a.edges(k).edge_index, b.edges(k).edge_index)


def test_node_ids_are_sorted(snapshot: GraphSnapshot) -> None:
    """Sorted ids → deterministic node-local indices regardless of source order."""
    for nt in snapshot.node_types():
        ids = snapshot.nodes(nt).ids
        assert ids == sorted(ids)


# ---- normalisation invariant ----------------------------------------------


def test_no_raw_unscaled_columns(snapshot: GraphSnapshot) -> None:
    """Acceptance: feature matrices are fully normalized."""
    for nt in snapshot.node_types():
        x = snapshot.nodes(nt).x
        if x.size == 0:
            continue
        assert float(x.min()) >= -1e-5
        assert float(x.max()) <= 1.0 + 1e-5


def test_edge_attrs_are_unit_clamped(snapshot: GraphSnapshot) -> None:
    for k in snapshot.edge_types():
        block = snapshot.edges(k)
        if block.edge_attr is None or block.edge_attr.size == 0:
            continue
        assert float(block.edge_attr.min()) >= -1e-5
        assert float(block.edge_attr.max()) <= 1.0 + 1e-5


# ---- structural validation (System 09) ------------------------------------


def test_three_era_export_passes_validation(source: InMemoryDataSource) -> None:
    """Issue acceptance: full graph export for 3 eras passes validation."""
    for slug in ERA_SLUGS:
        snap = build_snapshot(source, era_slug=slug)
        report = validate_snapshot(snap)
        assert report.passed, "; ".join(i.message for i in report.errors)


def test_validation_catches_dropped_edge_endpoints() -> None:
    """A dangling edge (endpoint not in node table) should be filtered, not raise."""
    src = InMemoryDataSource()
    src.set_patch("e2024_01", {"era_ordinal": 0.0})
    # One team, no players, but an edge claiming a player exists.
    src.add_node(
        "e2024_01",
        "team",
        NodeRow(
            id="t-0",
            features={
                "avg_acs": 200.0,
                "avg_kast": 0.65,
                "win_rate": 0.5,
                "map_win_rate": 0.5,
                "attack_rwr": 0.5,
                "defense_rwr": 0.5,
                "eco_efficiency": 0.5,
                "comeback_rate": 0.1,
                "strength_rating": 0.0,
                "style_aggression": 0.5,
                "style_utility": 0.5,
                "region_americas": 1.0,
                "region_emea": 0.0,
                "region_pacific": 0.0,
                "region_china": 0.0,
                "tier_rank": 0.5,
                "roster_age_days": 365.0,
            },
        ),
    )
    src.add_edge(
        "e2024_01",
        ("player", "plays_for", "team"),
        EdgeRow(src_id="ghost-player", dst_id="t-0"),
    )

    snap = build_snapshot(src, era_slug="e2024_01")
    # The orphan edge should be silently dropped — passing validation.
    plays_for = snap.edges(("player", "plays_for", "team"))
    assert plays_for.num_edges == 0


def test_edge_block_rejects_non_integer_edge_index() -> None:
    """Codex P2 (PR #24): a float edge_index must raise, not be coerced.

    Truncating ``0.9`` to ``0`` would still pass the bounds check and
    produce a "valid" snapshot with rewired edges — exactly the
    silent corruption the validator exists to catch.
    """
    from ecosystem.graph.snapshot import EdgeBlock

    bad_index = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
    with pytest.raises(ValueError, match="edge_index dtype"):
        EdgeBlock(
            src="player",
            relation="plays_for",
            dst="team",
            edge_index=bad_index,
            edge_attr=None,
            edge_attr_columns=(),
        )


def test_validator_flags_edge_attr_column_mismatch() -> None:
    """Codex P1 (PR #24): renamed/swapped edge_attr columns must fail.

    Same width, same dtype, same numeric range — a silent semantic
    swap that only column-name parity catches.
    """
    src = build_three_era_source()
    snap = build_snapshot(src, era_slug="e2024_01")
    block = snap.edges(("player", "plays_for", "team"))
    # Swap the two declared column names.
    block.edge_attr_columns = ("role_slot", "tenure_days")
    report = validate_snapshot(snap)
    assert not report.passed
    assert any(i.code == "edge_attr_column_mismatch" for i in report.errors)


def test_duplicate_edge_sort_ignores_non_schema_attributes() -> None:
    """Codex P2 (PR #24): non-schema keys must not perturb the sort key.

    Two sources that emit the same modeled features but disagree on a
    bookkeeping field (debug timestamps, per-source metadata) must
    still produce identical snapshot fingerprints. The
    duplicate-edge tie-breaker now restricts itself to
    ``spec.edge_attr_columns``; any other key on the row's
    ``attributes`` dict gets ignored.

    Also pins the missing-vs-None equivalence: a source that omits
    an attribute and one that sets it to ``None`` agree on order
    (both already produce the same ``edge_attr`` row via
    ``_extract_raw`` → NaN → ``_apply_fill_policy``).
    """
    base_attrs = {"tenure_days": 30.0, "role_slot": 0.2}

    def _build(extra_per_row: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
        src = InMemoryDataSource()
        src.set_patch("e_t", {"era_ordinal": 0.0})
        src.add_node("e_t", "player", NodeRow(id="p-0", features=_player_features_full(0.0)))
        src.add_node(
            "e_t",
            "team",
            NodeRow(
                id="t-0",
                features={
                    "avg_acs": 210.0,
                    "avg_kast": 0.68,
                    "win_rate": 0.5,
                    "map_win_rate": 0.5,
                    "attack_rwr": 0.5,
                    "defense_rwr": 0.5,
                    "eco_efficiency": 0.5,
                    "comeback_rate": 0.1,
                    "strength_rating": 0.0,
                    "style_aggression": 0.5,
                    "style_utility": 0.5,
                    "region_americas": 1.0,
                    "region_emea": 0.0,
                    "region_pacific": 0.0,
                    "region_china": 0.0,
                    "tier_rank": 0.5,
                    "roster_age_days": 365.0,
                },
            ),
        )
        # Three duplicate-endpoint edges; the per-row ``extra`` dict
        # holds non-schema keys that two sources might disagree on.
        for extra in extra_per_row:
            src.add_edge(
                "e_t",
                ("player", "plays_for", "team"),
                EdgeRow(src_id="p-0", dst_id="t-0", attributes={**base_attrs, **extra}),
            )
        snap = build_snapshot(src, era_slug="e_t")
        block = snap.edges(("player", "plays_for", "team"))
        assert block.edge_attr is not None
        return block.edge_index, block.edge_attr

    # Source A includes a "fetched_at" timestamp; source B omits it.
    a_index, a_attr = _build(
        [
            {"fetched_at": "2024-01-01T00:00:00Z"},
            {"fetched_at": "2024-01-02T00:00:00Z"},
            {"fetched_at": "2024-01-03T00:00:00Z"},
        ]
    )
    b_index, b_attr = _build([{}, {}, {}])
    np.testing.assert_array_equal(a_index, b_index)
    np.testing.assert_array_equal(a_attr, b_attr)


def test_validator_flags_missing_edge_attr_when_schema_declares_columns() -> None:
    """Codex P2 (PR #24): a typed-attribute edge with edge_attr=None must fail.

    Without this check, a corrupted ``snapshot.npz`` (or a producer
    that forgot to write the matrix) would pass validation, and the
    trainer would silently lose every attribute on relations like
    ``plays_for`` / ``affects``.
    """
    src = build_three_era_source()
    snap = build_snapshot(src, era_slug="e2024_01")
    # ``plays_for`` declares two edge_attr columns; blanking the matrix
    # post-build models the corrupted-snapshot scenario.
    snap.edges(("player", "plays_for", "team")).edge_attr = None
    report = validate_snapshot(snap)
    assert not report.passed
    assert any(i.code == "missing_edge_attr" for i in report.errors)


def test_validator_flags_out_of_range_node_features() -> None:
    """If we hand-mutate the snapshot post-build, the validator should catch it."""
    src = build_three_era_source()
    snap = build_snapshot(src, era_slug="e2024_01")
    # Inject a bug.
    snap.nodes("player").x[0, 0] = 5.0
    report = validate_snapshot(snap)
    assert not report.passed
    assert any(i.code == "feature_range_violation" for i in report.errors)


def test_assert_schema_known_passes_on_clean_snapshot() -> None:
    snap = build_snapshot(build_three_era_source(), era_slug="e2024_01")
    assert_schema_known(snap)  # does not raise


# ---- per-era fit independence ---------------------------------------------


def test_fits_differ_across_eras(source: InMemoryDataSource) -> None:
    """The fixture varies stats per era, so per-era min/max should differ."""
    a = build_snapshot(source, era_slug="e2024_01")
    c = build_snapshot(source, era_slug="e2024_03")
    fit_a = a.metadata["node_normalizer_params"]["player"]["acs"]["params"]
    fit_c = c.metadata["node_normalizer_params"]["player"]["acs"]["params"]
    assert fit_a != fit_c


# ---- empty era handling ---------------------------------------------------


def test_empty_era_produces_zero_row_blocks() -> None:
    src = InMemoryDataSource()
    src.set_patch("e_empty", {"era_ordinal": 0.0})
    snap = build_snapshot(src, era_slug="e_empty")
    # Patch is synthesised even when nothing else exists, so the
    # patch block carries 1 node. Other blocks are empty.
    assert snap.nodes("patch").num_nodes == 1
    assert snap.nodes("player").num_nodes == 0
    assert snap.nodes("team").num_nodes == 0
    # All edges empty.
    for k in snap.edge_types():
        assert snap.edges(k).num_edges == 0
    # Validator still passes — empty is structurally valid.
    assert validate_snapshot(snap).passed


# ---- fill_policy on missing values ---------------------------------------


def _player_features_full(seed: float) -> dict[str, float]:
    """Helper: a full player feature dict so we can blank one column at a time."""
    return {
        "acs": 200.0 + seed,
        "kast": 0.65,
        "adr": 130.0,
        "hs_pct": 0.22,
        "rating": 1.0,
        "fb_rate": 0.10,
        "fd_rate": 0.10,
        "clutch_rate": 0.10,
        "multikill_rate": 0.05,
        "latent_skill": 0.0,
        "role_fit": 0.5,
        "synergy_mag": 0.5,
        "age_days": 365.0,
        "region_americas": 1.0,
        "region_emea": 0.0,
        "region_pacific": 0.0,
        "region_china": 0.0,
        "tier_rank": 0.5,
    }


def test_missing_feature_does_not_propagate_nan() -> None:
    """Codex P1 (PR #24): omitted source columns must be back-filled.

    Without ``_apply_fill_policy``, a NaN raw value would survive the
    normaliser's arithmetic and trip the validator's
    ``non_finite_features`` check on the first real source row that
    happens to be missing a column.
    """
    src = InMemoryDataSource()
    src.set_patch("e_t", {"era_ordinal": 0.0})
    feats_complete = _player_features_full(seed=0.0)
    feats_missing = _player_features_full(seed=10.0)
    del feats_missing["acs"]  # simulate an upstream gap
    src.add_node("e_t", "player", NodeRow(id="p-0", features=feats_complete))
    src.add_node("e_t", "player", NodeRow(id="p-1", features=feats_missing))

    snap = build_snapshot(src, era_slug="e_t")
    player = snap.nodes("player")
    # No NaN cells anywhere.
    assert np.all(np.isfinite(player.x))
    # The missing ``acs`` cell got back-filled to 0.0 (the column's
    # declared fill_policy="zero" default).
    acs_idx = list(player.column_names).index("acs")
    p1_idx = player.ids.index("p-1")
    assert player.x[p1_idx, acs_idx] == 0.0
    # And the snapshot still validates.
    assert validate_snapshot(snap).passed


def test_duplicate_edges_get_deterministic_order() -> None:
    """Codex P2 (PR #24): duplicate-endpoint edges must sort deterministically.

    Two ``InMemoryDataSource`` instances with the same edges in two
    different insertion orders should produce identical snapshots —
    otherwise the registry's content fingerprint changes on every
    re-run and the idempotency contract breaks.
    """

    def _build_one(order: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
        src = InMemoryDataSource()
        src.set_patch("e_t", {"era_ordinal": 0.0})
        src.add_node("e_t", "player", NodeRow(id="p-0", features=_player_features_full(0.0)))
        # One team to anchor the dst endpoint.
        src.add_node(
            "e_t",
            "team",
            NodeRow(
                id="t-0",
                features={
                    "avg_acs": 210.0,
                    "avg_kast": 0.68,
                    "win_rate": 0.5,
                    "map_win_rate": 0.5,
                    "attack_rwr": 0.5,
                    "defense_rwr": 0.5,
                    "eco_efficiency": 0.5,
                    "comeback_rate": 0.1,
                    "strength_rating": 0.0,
                    "style_aggression": 0.5,
                    "style_utility": 0.5,
                    "region_americas": 1.0,
                    "region_emea": 0.0,
                    "region_pacific": 0.0,
                    "region_china": 0.0,
                    "tier_rank": 0.5,
                    "roster_age_days": 365.0,
                },
            ),
        )
        # Three duplicate-endpoint edges with distinct attributes.
        edges = [
            EdgeRow(src_id="p-0", dst_id="t-0", attributes={"tenure_days": 30.0, "role_slot": 0.2}),
            EdgeRow(src_id="p-0", dst_id="t-0", attributes={"tenure_days": 60.0, "role_slot": 0.4}),
            EdgeRow(src_id="p-0", dst_id="t-0", attributes={"tenure_days": 90.0, "role_slot": 0.6}),
        ]
        for i in order:
            src.add_edge("e_t", ("player", "plays_for", "team"), edges[i])
        snap = build_snapshot(src, era_slug="e_t")
        block = snap.edges(("player", "plays_for", "team"))
        assert block.edge_attr is not None
        return block.edge_index, block.edge_attr

    a_index, a_attr = _build_one((0, 1, 2))
    b_index, b_attr = _build_one((2, 0, 1))
    np.testing.assert_array_equal(a_index, b_index)
    np.testing.assert_array_equal(a_attr, b_attr)


def test_missing_edge_attribute_does_not_propagate_nan() -> None:
    """Edge attributes go through the same fill path; protect that too."""
    src = InMemoryDataSource()
    src.set_patch("e_t", {"era_ordinal": 0.0})
    src.add_node("e_t", "player", NodeRow(id="p-0", features=_player_features_full(0.0)))
    # One team with full feature dict so the team block validates.
    team_features = {
        "avg_acs": 210.0,
        "avg_kast": 0.68,
        "win_rate": 0.5,
        "map_win_rate": 0.5,
        "attack_rwr": 0.5,
        "defense_rwr": 0.5,
        "eco_efficiency": 0.5,
        "comeback_rate": 0.1,
        "strength_rating": 0.0,
        "style_aggression": 0.5,
        "style_utility": 0.5,
        "region_americas": 1.0,
        "region_emea": 0.0,
        "region_pacific": 0.0,
        "region_china": 0.0,
        "tier_rank": 0.5,
        "roster_age_days": 365.0,
    }
    src.add_node("e_t", "team", NodeRow(id="t-0", features=team_features))
    # Edge with one attribute deliberately omitted.
    src.add_edge(
        "e_t",
        ("player", "plays_for", "team"),
        EdgeRow(src_id="p-0", dst_id="t-0", attributes={"role_slot": 0.4}),
    )

    snap = build_snapshot(src, era_slug="e_t")
    plays_for = snap.edges(("player", "plays_for", "team"))
    assert plays_for.edge_attr is not None
    assert np.all(np.isfinite(plays_for.edge_attr))
    assert validate_snapshot(snap).passed
