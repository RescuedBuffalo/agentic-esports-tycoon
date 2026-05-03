"""Run-the-export orchestrator.

Glues :func:`build_snapshot` → :func:`validate_snapshot` → BUF-69
:class:`Registry` together. The exit contract is binary: a passing
validation produces a ``completed`` registry row and ``snapshot.npz``
+ ``manifest.json`` + ``validation.json`` under the run directory; a
failing validation produces a ``failed`` row plus the same artifacts
(so an operator can read ``validation.json`` to see why).

The function returns the :class:`~esports_sim.registry.RunRecord` so
the caller can read paths off it without re-querying the registry.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path

from esports_sim.registry import Registry, RunStatus

from ecosystem.graph.builder import build_snapshot
from ecosystem.graph.schema import SCHEMA_VERSION
from ecosystem.graph.snapshot import GraphSnapshot
from ecosystem.graph.source import GraphDataSource
from ecosystem.graph.validate import (
    StructuralValidationError,
    ValidationReport,
    validate_snapshot,
)

_log = logging.getLogger(__name__)

# The registry's ``kind`` field is part of the run id and a CLI
# filter; keep it aligned with the System-08 vocabulary so future
# digest tools can find graph runs without a substring search.
GRAPH_SNAPSHOT_KIND = "graph-snapshot"


@dataclass(frozen=True, slots=True)
class ExportResult:
    """Compact summary of one ``export_era`` invocation.

    Returned to CLI / batch callers so they can log or aggregate
    without re-reading the registry. Carries the full
    :class:`ValidationReport` for ad-hoc inspection.
    """

    run_id: str
    run_dir: Path
    snapshot_path: Path
    manifest_path: Path
    validation_path: Path
    report: ValidationReport
    snapshot: GraphSnapshot

    @property
    def passed(self) -> bool:
        return self.report.passed


def export_era(
    source: GraphDataSource,
    *,
    era_slug: str,
    config_path: Path | str,
    registry: Registry | None = None,
) -> ExportResult:
    """Build, validate, register, and persist one era's snapshot.

    ``config_path`` is the YAML/JSON config that drove this build; the
    registry hashes its bytes for idempotency. Re-running with the
    same config + same source produces the same run_id (no-op).

    Pass ``registry`` to share a Registry instance across multiple
    eras (typical for a batch run); otherwise the function constructs
    a default one wired through the standard env vars.
    """
    config_path = Path(config_path)
    registry = registry or Registry()

    # Building the snapshot first means the data fingerprint reflects
    # the actual graph contents, not just the config — two runs of the
    # same config against different upstream snapshots get different
    # run ids, which is the right semantics for the registry's
    # idempotency contract.
    snapshot = build_snapshot(source, era_slug=era_slug)
    fingerprint = _fingerprint_snapshot(snapshot)

    notes = f"era={era_slug} schema={SCHEMA_VERSION}"
    run_id = registry.register(
        kind=GRAPH_SNAPSHOT_KIND,
        config_path=config_path,
        data_fingerprint=fingerprint,
        notes=notes,
    )
    record = registry.get(run_id)

    # Idempotent re-runs: the registry returned an existing run_id.
    # If the on-disk artifacts are still there, skip the rebuild — the
    # contract is "same input → same output", and writing the same
    # bytes twice is wasted I/O. If they're missing (operator deleted
    # them), rebuild them under the existing run_id.
    snapshot_path = record.run_dir / "snapshot.npz"
    manifest_path = record.run_dir / "manifest.json"
    validation_path = record.run_dir / "validation.json"
    if (
        record.status is RunStatus.COMPLETED
        and snapshot_path.exists()
        and manifest_path.exists()
        and validation_path.exists()
    ):
        report = _load_report(validation_path, era_slug)
        _log.info(
            "graph-snapshot.idempotent_hit",
            extra={"run_id": run_id, "era_slug": era_slug},
        )
        return ExportResult(
            run_id=run_id,
            run_dir=record.run_dir,
            snapshot_path=snapshot_path,
            manifest_path=manifest_path,
            validation_path=validation_path,
            report=report,
            snapshot=snapshot,
        )

    # Validate before writing so a fail-on-validation run still has
    # the npz / manifest on disk for inspection (writing happens
    # below regardless of pass/fail), but we know the verdict before
    # finalising.
    report = validate_snapshot(snapshot)

    # Write artifacts. The npz + manifest go through GraphSnapshot's
    # writer for round-trip parity with snapshot.read; validation.json
    # is independent.
    snapshot.write(record.run_dir)
    validation_path.write_text(
        json.dumps(report.to_jsonable(), indent=2, sort_keys=True),
        encoding="utf-8",
    )

    final_status = RunStatus.COMPLETED if report.passed else RunStatus.FAILED
    finalize_notes = (
        f"validation={'pass' if report.passed else 'fail'} "
        f"errors={len(report.errors)} warnings={len(report.warnings)}"
    )
    registry.finalize(run_id, status=final_status, notes=finalize_notes)

    return ExportResult(
        run_id=run_id,
        run_dir=record.run_dir,
        snapshot_path=snapshot_path,
        manifest_path=manifest_path,
        validation_path=validation_path,
        report=report,
        snapshot=snapshot,
    )


def export_eras(
    source: GraphDataSource,
    *,
    era_slugs: list[str],
    config_path: Path | str,
    registry: Registry | None = None,
    fail_fast: bool = False,
) -> list[ExportResult]:
    """Sequential batch wrapper around :func:`export_era`.

    The acceptance scenario explicitly mentions "Full graph export for
    3 eras passes all structural validation" — this is the helper
    that does that. ``fail_fast`` controls whether a failed validation
    raises immediately (operator pipelines) or continues to the next
    era (CI batch jobs collecting a full report).
    """
    registry = registry or Registry()
    results: list[ExportResult] = []
    for slug in era_slugs:
        result = export_era(
            source, era_slug=slug, config_path=config_path, registry=registry
        )
        results.append(result)
        if fail_fast and not result.passed:
            raise StructuralValidationError(
                f"era {slug!r} failed structural validation; aborting batch "
                f"(see {result.validation_path})"
            )
    return results


# ---- helpers ---------------------------------------------------------------


def _fingerprint_snapshot(snapshot: GraphSnapshot) -> str:
    """Stable SHA-256 over the snapshot's tensors + structural metadata.

    This is what the registry uses to decide idempotency. We *do not*
    include normaliser fit parameters in the fingerprint: they are a
    pure function of the tensors, so folding them in would be
    redundant. We *do* include the schema version so a schema bump
    forces a re-register even if the underlying ids are unchanged.
    """
    h = hashlib.sha256()
    h.update(SCHEMA_VERSION.encode("utf-8"))
    h.update(snapshot.snapshot_format_version.encode("utf-8"))
    h.update(snapshot.era_slug.encode("utf-8"))
    for nt in sorted(snapshot.node_blocks):
        node_block = snapshot.node_blocks[nt]
        h.update(f"node:{nt}".encode())
        h.update(b"\x00".join(i.encode("utf-8") for i in node_block.ids))
        h.update(node_block.x.tobytes())
    for k in sorted(snapshot.edge_blocks):
        edge_block = snapshot.edge_blocks[k]
        h.update(("edge:" + "__".join(k)).encode("utf-8"))
        h.update(edge_block.edge_index.tobytes())
        if edge_block.edge_attr is not None:
            h.update(edge_block.edge_attr.tobytes())
    return h.hexdigest()


def _load_report(path: Path, era_slug: str) -> ValidationReport:
    """Re-hydrate a ``validation.json`` for the idempotent-hit path.

    Only the fields :class:`ExportResult` actually exposes are
    repopulated — the export path returns the original snapshot
    object, so a callsite that wants the full issue list can read it
    off the file directly.
    """
    raw = json.loads(path.read_text(encoding="utf-8"))
    from ecosystem.graph.validate import ValidationIssue  # noqa: PLC0415 (local)

    issues = [
        ValidationIssue(
            code=i["code"],
            severity=i["severity"],
            location=i["location"],
            message=i["message"],
        )
        for i in raw.get("issues", [])
    ]
    return ValidationReport(
        era_slug=raw.get("era_slug", era_slug),
        schema_version=raw.get("schema_version", SCHEMA_VERSION),
        issues=issues,
    )


__all__ = [
    "ExportResult",
    "GRAPH_SNAPSHOT_KIND",
    "export_era",
    "export_eras",
]
