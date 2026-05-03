"""Build a :class:`GraphSnapshot` for one era from a :class:`GraphDataSource`.

The builder is the only place that talks to both the schema and the
data source. It:

1. Pulls every node row of every declared node type from the source.
2. For each declared :class:`FeatureColumn`, extracts the raw values,
   applies the column's fill policy to the missing entries, runs the
   declared normaliser, and stacks the result into the node-type's
   ``x`` matrix.
3. Pulls every edge row of every declared edge type, indexes the
   ``src_id`` / ``dst_id`` pairs against the node id tables, drops
   edges whose endpoints didn't survive a ``drop_node`` policy, and
   normalises any edge attributes the schema declares.
4. Wraps the result in a :class:`GraphSnapshot` with normaliser
   parameters and per-era metadata stamped into ``metadata``.

The builder is pure: same source + same era → byte-identical
snapshot. Determinism matters because the registry's data
fingerprint is folded into the run id; a non-deterministic builder
would mint a fresh run id on every re-run and defeat idempotency.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ecosystem.graph.normalize import FitParams, normalize_column
from ecosystem.graph.schema import (
    EDGE_TYPES,
    NODE_TYPES,
    EdgeSpec,
    FeatureColumn,
    NodeSpec,
)
from ecosystem.graph.snapshot import EdgeBlock, GraphSnapshot, NodeBlock
from ecosystem.graph.source import EdgeRow, GraphDataSource, NodeRow


class BuildError(RuntimeError):
    """Raised when the builder can't satisfy the schema for the era."""


def build_snapshot(
    source: GraphDataSource,
    *,
    era_slug: str,
) -> GraphSnapshot:
    """Build the full :class:`GraphSnapshot` for ``era_slug``."""
    snapshot = GraphSnapshot(era_slug=era_slug)
    # patch_meta is consumed both as a node-feature source for the
    # synthetic ``patch`` node and as snapshot-wide metadata.
    patch_meta = source.patch_meta(era_slug)

    normalizer_params: dict[str, dict[str, dict[str, Any]]] = {}

    # Node blocks --------------------------------------------------------
    # ``ids_by_type`` is the post-build id table per node type, used
    # downstream for edge endpoint resolution.
    ids_by_type: dict[str, dict[str, int]] = {}
    for node_type, node_schema in NODE_TYPES.items():
        node_rows = list(source.iter_nodes(node_type, era_slug=era_slug))
        # Patch is the only synthetic node type — if the source didn't
        # emit a row for it, materialise one from the patch metadata.
        if node_type == "patch" and not node_rows:
            node_rows = [_patch_row_from_meta(era_slug, patch_meta)]
        node_block, node_fit = _build_node_block(node_schema, node_rows)
        snapshot.node_blocks[node_type] = node_block
        ids_by_type[node_type] = {nid: idx for idx, nid in enumerate(node_block.ids)}
        normalizer_params[node_type] = {
            col: fit.to_jsonable() for col, fit in node_fit.items()
        }

    # Edge blocks --------------------------------------------------------
    edge_normalizer_params: dict[str, dict[str, dict[str, Any]]] = {}
    for key, edge_schema in EDGE_TYPES.items():
        edge_rows = list(source.iter_edges(key, era_slug=era_slug))
        edge_block, edge_fit = _build_edge_block(edge_schema, edge_rows, ids_by_type)
        snapshot.edge_blocks[key] = edge_block
        edge_normalizer_params["__".join(key)] = {
            col: fit.to_jsonable() for col, fit in edge_fit.items()
        }

    # Stamp deterministic metadata. The build timestamp deliberately
    # is *not* recorded — wall-clock would make the snapshot non-
    # deterministic, breaking the registry's content-hash idempotency.
    snapshot.metadata = {
        "era_slug": era_slug,
        "patch_meta": patch_meta,
        "node_normalizer_params": normalizer_params,
        "edge_normalizer_params": edge_normalizer_params,
    }
    return snapshot


# ---- node-block construction ----------------------------------------------


def _build_node_block(
    spec: NodeSpec,
    rows: list[NodeRow],
) -> tuple[NodeBlock, dict[str, FitParams]]:
    """Return the populated NodeBlock and per-column fit params."""
    columns = spec.all_columns()

    # First pass: drop any rows where a ``drop_node`` column is missing.
    drop_required = [col.name for col in columns if col.fill_policy == "drop_node"]
    if drop_required:
        rows = [
            r for r in rows
            if all(_is_present(r.features.get(name)) for name in drop_required)
        ]

    n = len(rows)
    if n == 0:
        # Empty era is legal (a region with no matches in the window);
        # keep an empty (0, F) matrix so downstream shape handling
        # stays uniform.
        x = np.zeros((0, spec.feature_dim()), dtype=np.float32)
        return (
            NodeBlock(
                node_type=spec.name,
                ids=[],
                x=x,
                column_names=spec.column_names(),
            ),
            {},
        )

    # Stable id ordering. Sorting by id makes the snapshot deterministic
    # regardless of the source's iteration order — a different source
    # implementation can't perturb edge_index integers downstream.
    rows.sort(key=lambda r: r.id)

    cols: list[np.ndarray] = []
    fit_by_col: dict[str, FitParams] = {}
    for col in columns:
        raw = np.array(
            [_extract_raw(r.features, col) for r in rows],
            dtype=np.float64,
        )
        # ``normalize_column`` ignores NaN inputs during fit but leaves
        # them in the output for the missing rows — that's the point at
        # which the column's declared ``fill_policy`` has to step in.
        # Without this fill the snapshot would carry NaN cells and
        # tip the validator's ``non_finite_features`` check.
        normed, fit = normalize_column(raw, normalizer_name=col.normalizer)
        normed = _apply_fill_policy(normed, raw, col)
        cols.append(normed)
        fit_by_col[col.name] = fit

    x = np.stack(cols, axis=1).astype(np.float32, copy=False)
    block = NodeBlock(
        node_type=spec.name,
        ids=[r.id for r in rows],
        x=x,
        column_names=spec.column_names(),
    )
    return block, fit_by_col


def _attr_sort_key(attrs: dict[str, Any]) -> tuple[tuple[str, str], ...]:
    """Stable tertiary sort key for edges with identical endpoints.

    Sorting attribute items by key first canonicalises the dict
    iteration order; rendering values through ``str`` keeps the key
    comparable even when values are mixed types (float / NaN /
    datetime). We don't need the original value back — just a
    deterministic ordering that two equivalent source implementations
    will agree on.
    """
    return tuple(sorted((k, str(attrs[k])) for k in attrs))


def _apply_fill_policy(
    normed: np.ndarray, raw: np.ndarray, col: FeatureColumn
) -> np.ndarray:
    """Replace NaN cells (originally missing values) per the column's policy.

    ``raw`` is the pre-normalisation column; we use it to identify
    which rows were missing rather than re-checking ``normed`` —
    the normalisers themselves can't *introduce* NaN, so the two
    masks are equivalent, but reading from ``raw`` keeps the
    intent visible to readers.

    ``drop_node`` rows have already been filtered upstream, so
    that policy never reaches this function. We still defensively
    no-op on it rather than asserting, so a future refactor can't
    crash here.
    """
    missing = np.isnan(raw)
    if not missing.any():
        return normed
    if col.fill_policy == "zero":
        normed = normed.copy()
        normed[missing] = 0.0
        return normed
    if col.fill_policy == "mean":
        # Post-normalisation mean is well-defined for every kind: it's
        # the mean of the surviving (non-NaN) cells in ``normed`` itself.
        # If every cell is missing, fall back to 0.5 — the "neutral"
        # post-normalisation centre — rather than propagating NaN.
        normed = normed.copy()
        finite = normed[~missing]
        fill_value = float(finite.mean()) if finite.size else 0.5
        normed[missing] = fill_value
        return normed
    # ``drop_node`` reaches here only if the upstream filter let a
    # row through anyway (e.g. a passthrough column with the policy
    # nominally set). Treat as zero rather than failing — the
    # validator catches anything genuinely broken.
    normed = normed.copy()
    normed[missing] = 0.0
    return normed


def _extract_raw(features: dict[str, Any], col: FeatureColumn) -> float:
    """Pull a raw scalar for ``col`` from a row's features dict.

    Missing / None / NaN values are *not* substituted here — they are
    preserved as NaN so the normaliser's fit phase can ignore them.
    The post-normalisation NaN-fill below applies the column's
    fill_policy on the normalised side, where 0 / mean have a
    well-defined meaning.
    """
    val = features.get(col.name)
    if val is None:
        return float("nan")
    try:
        f = float(val)
    except (TypeError, ValueError):
        return float("nan")
    if math.isnan(f) or math.isinf(f):
        return float("nan")
    return f


def _is_present(value: Any) -> bool:
    if value is None:
        return False
    try:
        f = float(value)
    except (TypeError, ValueError):
        return True  # non-numeric but present (e.g. a label)
    return not math.isnan(f)


# ---- edge-block construction ----------------------------------------------


def _build_edge_block(
    spec: EdgeSpec,
    rows: list[EdgeRow],
    ids_by_type: dict[str, dict[str, int]],
) -> tuple[EdgeBlock, dict[str, FitParams]]:
    src_lookup = ids_by_type.get(spec.src, {})
    dst_lookup = ids_by_type.get(spec.dst, {})

    # Resolve endpoint ids to node-local indices; drop edges whose
    # endpoints didn't survive the node build (e.g. dropped by
    # drop_node policy, or simply absent from the era).
    src_idx: list[int] = []
    dst_idx: list[int] = []
    kept_rows: list[EdgeRow] = []
    for row in rows:
        s = src_lookup.get(row.src_id)
        d = dst_lookup.get(row.dst_id)
        if s is None or d is None:
            continue
        src_idx.append(s)
        dst_idx.append(d)
        kept_rows.append(row)

    # Determinism: sort by (src, dst) so a reorder in the source can't
    # shuffle the snapshot's edge_index columns. Two rows with the
    # same endpoints are legal (a time-bounded relation re-asserted at
    # different windows; an upstream double-write) and we break the
    # tie on a canonicalised attribute fingerprint so identical
    # source contents from two implementations still produce
    # byte-identical snapshots — preserving the registry's
    # idempotency contract.
    if kept_rows:
        order = sorted(
            range(len(kept_rows)),
            key=lambda i: (
                src_idx[i],
                dst_idx[i],
                _attr_sort_key(kept_rows[i].attributes),
            ),
        )
        src_idx = [src_idx[i] for i in order]
        dst_idx = [dst_idx[i] for i in order]
        kept_rows = [kept_rows[i] for i in order]

    edge_index = np.array([src_idx, dst_idx], dtype=np.int64)
    if edge_index.size == 0:
        edge_index = np.zeros((2, 0), dtype=np.int64)

    edge_attr: np.ndarray | None = None
    fit_by_col: dict[str, FitParams] = {}
    if spec.edge_attr_columns:
        if not kept_rows:
            edge_attr = np.zeros(
                (0, len(spec.edge_attr_columns)), dtype=np.float32
            )
        else:
            cols: list[np.ndarray] = []
            for col in spec.edge_attr_columns:
                raw = np.array(
                    [_extract_raw(r.attributes, col) for r in kept_rows],
                    dtype=np.float64,
                )
                normed, fit = normalize_column(raw, normalizer_name=col.normalizer)
                # Same fill story as the node-block path — without
                # this, an edge row missing one attribute would
                # leave a NaN in ``edge_attr`` and trip
                # ``non_finite_edge_attr``.
                normed = _apply_fill_policy(normed, raw, col)
                cols.append(normed)
                fit_by_col[col.name] = fit
            edge_attr = np.stack(cols, axis=1).astype(np.float32, copy=False)

    block = EdgeBlock(
        src=spec.src,
        relation=spec.relation,
        dst=spec.dst,
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_attr_columns=tuple(c.name for c in spec.edge_attr_columns),
    )
    return block, fit_by_col


# ---- patch synthetic node --------------------------------------------------


def _patch_row_from_meta(era_slug: str, patch_meta: dict[str, Any]) -> NodeRow:
    """Build a single patch node from the source's era metadata.

    The data source is *allowed* to emit its own patch row (e.g. via a
    SqlDataSource that joins with patch_era and the in-era match
    aggregates), but for fixture-driven runs and for fresh eras we
    derive a minimal row from whatever the source's
    :meth:`patch_meta` returns.
    """
    season_year = patch_meta.get("season_year")
    return NodeRow(
        id=era_slug,
        features={
            "agg_match_acs": patch_meta.get("agg_match_acs", float("nan")),
            "agg_match_kast": patch_meta.get("agg_match_kast", float("nan")),
            "duration_days": patch_meta.get("duration_days", float("nan")),
            "matches_played": patch_meta.get("matches_played", float("nan")),
            "agents_added": patch_meta.get("agents_added", 0.0),
            "maps_added": patch_meta.get("maps_added", 0.0),
            "meta_magnitude": patch_meta.get("meta_magnitude", 0.0),
            "is_major_shift": 1.0 if patch_meta.get("is_major_shift") else 0.0,
            "era_ordinal": patch_meta.get("era_ordinal", 0.0),
            "season_year_2024": 1.0 if season_year == 2024 else 0.0,
            "season_year_2025": 1.0 if season_year == 2025 else 0.0,
            "season_year_2026": 1.0 if season_year == 2026 else 0.0,
        },
    )


__all__ = ["BuildError", "build_snapshot"]
