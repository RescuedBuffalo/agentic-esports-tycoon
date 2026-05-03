"""Structural validation of a :class:`GraphSnapshot` (Systems-spec System 09).

The validator is the gate between ``build_snapshot`` and
``Registry.finalize``: a snapshot that fails any check is a
``failed`` run, not a ``completed`` one. Acceptance:

> Full graph export for 3 eras passes all structural validation.

Checks fall into three buckets:

1. **Schema consistency** — the snapshot only references node/edge
   types the runtime knows; column counts / dtypes match the schema.
2. **Numerical health** — no NaN/Inf values; feature matrices are in
   the unit interval (the "no raw un-scaled columns" acceptance);
   ids are non-empty unique strings.
3. **Edge integrity** — every edge endpoint refers to a valid index;
   no orphan edges; no self-loops on relations that forbid them
   (currently none, but the hook is here).

The validator returns a :class:`ValidationReport` rather than raising
on the first failure — operators want the full picture, not the first
broken row. The export orchestrator inspects ``report.passed`` and
either finalises the registry row as ``completed`` or ``failed``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ecosystem.graph.schema import (
    EDGE_TYPES,
    NODE_TYPES,
    SCHEMA_VERSION,
)
from ecosystem.graph.snapshot import GraphSnapshot

# Tolerance for unit-interval checks. The normalisers explicitly clip
# their output, so anything outside this window indicates a bypassed
# normaliser, not a rounding artefact.
_UNIT_SLACK = 1e-5


class StructuralValidationError(RuntimeError):
    """Raised when :func:`assert_valid` is called on a failed report."""


@dataclass(slots=True)
class ValidationIssue:
    """One thing the validator found wrong with the snapshot."""

    code: str
    severity: str  # "error" | "warn"
    location: str  # human-readable: "node:player", "edge:patch__affects__agent", ...
    message: str

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "severity": self.severity,
            "location": self.location,
            "message": self.message,
        }


@dataclass(slots=True)
class ValidationReport:
    """The full output of :func:`validate_snapshot`."""

    era_slug: str
    schema_version: str
    issues: list[ValidationIssue] = field(default_factory=list)

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == "warn"]

    @property
    def passed(self) -> bool:
        return not self.errors

    def to_jsonable(self) -> dict[str, Any]:
        return {
            "era_slug": self.era_slug,
            "schema_version": self.schema_version,
            "passed": self.passed,
            "error_count": len(self.errors),
            "warning_count": len(self.warnings),
            "issues": [i.to_jsonable() for i in self.issues],
        }

    def assert_passed(self) -> None:
        if self.passed:
            return
        joined = "; ".join(
            f"[{i.code} @ {i.location}] {i.message}" for i in self.errors[:5]
        )
        more = f" (+{len(self.errors) - 5} more)" if len(self.errors) > 5 else ""
        raise StructuralValidationError(
            f"snapshot for era {self.era_slug!r} failed structural validation: "
            f"{joined}{more}"
        )


def validate_snapshot(snapshot: GraphSnapshot) -> ValidationReport:
    """Run every structural check; return the full report."""
    report = ValidationReport(
        era_slug=snapshot.era_slug,
        schema_version=snapshot.schema_version,
    )
    _check_schema_version(snapshot, report)
    _check_node_blocks(snapshot, report)
    _check_edge_blocks(snapshot, report)
    return report


# ---- individual check groups ----------------------------------------------


def _check_schema_version(
    snapshot: GraphSnapshot, report: ValidationReport
) -> None:
    if snapshot.schema_version != SCHEMA_VERSION:
        report.issues.append(
            ValidationIssue(
                code="schema_version_mismatch",
                severity="error",
                location="snapshot",
                message=(
                    f"snapshot schema_version={snapshot.schema_version!r} "
                    f"!= runtime SCHEMA_VERSION={SCHEMA_VERSION!r}"
                ),
            )
        )


def _check_node_blocks(
    snapshot: GraphSnapshot, report: ValidationReport
) -> None:
    # Cover every declared node type — missing is an error, not a warn,
    # since the trainer's fixed-shape ingestion would index off the end.
    for node_type, spec in NODE_TYPES.items():
        loc = f"node:{node_type}"
        block = snapshot.node_blocks.get(node_type)
        if block is None:
            report.issues.append(
                ValidationIssue(
                    "missing_node_block",
                    "error",
                    loc,
                    "snapshot has no block for this declared node type",
                )
            )
            continue

        # Column-count parity with the schema.
        expected_cols = spec.column_names()
        if block.column_names != expected_cols:
            report.issues.append(
                ValidationIssue(
                    "column_mismatch",
                    "error",
                    loc,
                    f"column_names={block.column_names!r} != "
                    f"expected={expected_cols!r}",
                )
            )

        # Shape sanity — NodeBlock.__post_init__ catches construction-
        # time mismatches; this catches a snapshot that was loaded
        # from disk with an inconsistent npz/manifest pair.
        if block.x.shape != (block.num_nodes, len(expected_cols)):
            report.issues.append(
                ValidationIssue(
                    "shape_mismatch",
                    "error",
                    loc,
                    f"x.shape={block.x.shape} but expected "
                    f"({block.num_nodes}, {len(expected_cols)})",
                )
            )

        # Numerical health: NaN / Inf would silently propagate through
        # the GNN's matmul and corrupt the loss; catch them here.
        if block.num_nodes > 0:
            if not np.all(np.isfinite(block.x)):
                bad = int(np.sum(~np.isfinite(block.x)))
                report.issues.append(
                    ValidationIssue(
                        "non_finite_features",
                        "error",
                        loc,
                        f"{bad} non-finite values in x",
                    )
                )
            # "Fully normalised" — the acceptance requires every column
            # in [0, 1]. Let _UNIT_SLACK absorb fp32-precision drift
            # from the normaliser's internal fp64 arithmetic.
            lo = float(block.x.min())
            hi = float(block.x.max())
            if lo < -_UNIT_SLACK or hi > 1.0 + _UNIT_SLACK:
                report.issues.append(
                    ValidationIssue(
                        "feature_range_violation",
                        "error",
                        loc,
                        f"x range=[{lo:.6f}, {hi:.6f}] outside [0, 1]",
                    )
                )

        # ID hygiene.
        if len(set(block.ids)) != len(block.ids):
            report.issues.append(
                ValidationIssue(
                    "duplicate_node_ids",
                    "error",
                    loc,
                    f"{len(block.ids) - len(set(block.ids))} duplicates",
                )
            )
        if any((not isinstance(i, str)) or not i for i in block.ids):
            report.issues.append(
                ValidationIssue(
                    "empty_node_ids",
                    "error",
                    loc,
                    "found empty / non-string node id",
                )
            )

    # Reverse direction: snapshot carrying an unknown node type.
    for nt in snapshot.node_blocks:
        if nt not in NODE_TYPES:
            report.issues.append(
                ValidationIssue(
                    "unknown_node_type",
                    "error",
                    f"node:{nt}",
                    f"snapshot block {nt!r} not in runtime schema {sorted(NODE_TYPES)}",
                )
            )


def _check_edge_blocks(
    snapshot: GraphSnapshot, report: ValidationReport
) -> None:
    for key, spec in EDGE_TYPES.items():
        loc = f"edge:{'__'.join(key)}"
        block = snapshot.edge_blocks.get(key)
        if block is None:
            report.issues.append(
                ValidationIssue(
                    "missing_edge_block",
                    "error",
                    loc,
                    "snapshot has no block for this declared edge type",
                )
            )
            continue

        if block.edge_index.shape[0] != 2:
            report.issues.append(
                ValidationIssue(
                    "edge_index_shape",
                    "error",
                    loc,
                    f"edge_index must be (2, E); got {block.edge_index.shape}",
                )
            )
            continue  # subsequent checks would raise on the bad shape

        # Endpoint integrity. The builder filters orphans, but a hand-
        # constructed snapshot or a loaded npz could still violate.
        src_block = snapshot.node_blocks.get(spec.src)
        dst_block = snapshot.node_blocks.get(spec.dst)
        if src_block is None or dst_block is None:
            # The missing-block error is already reported above; skip
            # endpoint checks since we can't compute bounds.
            continue
        if block.num_edges:
            src_max = int(block.edge_index[0].max(initial=-1))
            dst_max = int(block.edge_index[1].max(initial=-1))
            src_min = int(block.edge_index[0].min(initial=0))
            dst_min = int(block.edge_index[1].min(initial=0))
            if src_min < 0 or src_max >= src_block.num_nodes:
                report.issues.append(
                    ValidationIssue(
                        "src_index_out_of_bounds",
                        "error",
                        loc,
                        f"src indices in [{src_min}, {src_max}] but "
                        f"{spec.src} has {src_block.num_nodes} nodes",
                    )
                )
            if dst_min < 0 or dst_max >= dst_block.num_nodes:
                report.issues.append(
                    ValidationIssue(
                        "dst_index_out_of_bounds",
                        "error",
                        loc,
                        f"dst indices in [{dst_min}, {dst_max}] but "
                        f"{spec.dst} has {dst_block.num_nodes} nodes",
                    )
                )

        # If the schema declares attribute columns for this edge, the
        # block must carry an ``edge_attr`` matrix — a corrupted
        # ``snapshot.npz`` (or a producer that forgot to write the
        # array) would otherwise sail through and the trainer would
        # silently lose every attribute on relations like
        # ``plays_for`` / ``affects``.
        if spec.edge_attr_columns and block.edge_attr is None:
            report.issues.append(
                ValidationIssue(
                    "missing_edge_attr",
                    "error",
                    loc,
                    f"schema declares {len(spec.edge_attr_columns)} edge_attr "
                    f"column(s) but block carries no edge_attr matrix",
                )
            )

        # Column-name parity. A snapshot with the right *width* but
        # swapped or renamed attribute columns would otherwise pass
        # validation and the trainer would consume the wrong feature
        # at every position — silent semantic corruption is exactly
        # what System 09 should catch.
        expected_attr_cols = tuple(c.name for c in spec.edge_attr_columns)
        if block.edge_attr_columns != expected_attr_cols:
            report.issues.append(
                ValidationIssue(
                    "edge_attr_column_mismatch",
                    "error",
                    loc,
                    f"edge_attr_columns={block.edge_attr_columns!r} != "
                    f"expected={expected_attr_cols!r}",
                )
            )

        # Edge-attribute health: same NaN / range checks as nodes.
        if block.edge_attr is not None and block.num_edges:
            if not np.all(np.isfinite(block.edge_attr)):
                bad = int(np.sum(~np.isfinite(block.edge_attr)))
                report.issues.append(
                    ValidationIssue(
                        "non_finite_edge_attr",
                        "error",
                        loc,
                        f"{bad} non-finite values in edge_attr",
                    )
                )
            lo = float(block.edge_attr.min())
            hi = float(block.edge_attr.max())
            if lo < -_UNIT_SLACK or hi > 1.0 + _UNIT_SLACK:
                report.issues.append(
                    ValidationIssue(
                        "edge_attr_range_violation",
                        "error",
                        loc,
                        f"edge_attr range=[{lo:.6f}, {hi:.6f}] outside [0, 1]",
                    )
                )

    # Snapshot edge type the runtime doesn't know.
    for k in snapshot.edge_blocks:
        if k not in EDGE_TYPES:
            report.issues.append(
                ValidationIssue(
                    "unknown_edge_type",
                    "error",
                    f"edge:{'__'.join(k)}",
                    f"snapshot block {k!r} not in runtime schema {sorted(EDGE_TYPES)}",
                )
            )


__all__ = [
    "StructuralValidationError",
    "ValidationIssue",
    "ValidationReport",
    "validate_snapshot",
]
