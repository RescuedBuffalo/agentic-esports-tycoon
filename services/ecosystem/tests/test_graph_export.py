"""Snapshot IO + export-orchestrator tests (BUF-53).

The acceptance scenario is "full graph export for 3 eras passes
structural validation"; these tests run that flow end-to-end through
the real BUF-69 :class:`Registry` and assert the on-disk artifacts
are present, well-formed, and round-trip cleanly back into a
:class:`GraphSnapshot`.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pytest
from ecosystem.graph import GraphSnapshot, build_snapshot, validate_snapshot
from ecosystem.graph.export import (
    GRAPH_SNAPSHOT_KIND,
    export_era,
    export_eras,
)
from ecosystem.graph.snapshot import assert_schema_known
from esports_sim.registry import Registry, RunStatus
from graph_fixtures import ERA_SLUGS, build_three_era_source

# ---- fixtures --------------------------------------------------------------


@pytest.fixture
def registry(tmp_path: Path) -> Registry:
    return Registry(
        db_path=tmp_path / "state" / "registry.db",
        runs_dir=tmp_path / "runs",
    )


@pytest.fixture
def config_path(tmp_path: Path) -> Path:
    p = tmp_path / "graph_snapshot.yaml"
    p.write_text(
        "kind: graph-snapshot\nschema_version: 1.0.0\n", encoding="utf-8"
    )
    return p


# ---- snapshot round-trip ---------------------------------------------------


def test_snapshot_round_trips_through_npz(tmp_path: Path) -> None:
    src = build_three_era_source()
    snap = build_snapshot(src, era_slug="e2024_01")
    out_dir = tmp_path / "snap"
    snap.write(out_dir)

    loaded = GraphSnapshot.read(out_dir)
    assert loaded.era_slug == snap.era_slug
    for nt in snap.node_types():
        np.testing.assert_array_equal(
            loaded.nodes(nt).x, snap.nodes(nt).x
        )
        assert loaded.nodes(nt).ids == snap.nodes(nt).ids
        assert loaded.nodes(nt).column_names == snap.nodes(nt).column_names
    for k in snap.edge_types():
        np.testing.assert_array_equal(
            loaded.edges(k).edge_index, snap.edges(k).edge_index
        )
    assert_schema_known(loaded)


def test_manifest_records_per_node_columns(tmp_path: Path) -> None:
    src = build_three_era_source()
    snap = build_snapshot(src, era_slug="e2024_01")
    snap.write(tmp_path)
    manifest = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))

    assert "node_columns" in manifest
    assert "player" in manifest["node_columns"]
    # acs / kast / adr / hs_pct / rating headline + derived + inferred + context
    assert "acs" in manifest["node_columns"]["player"]
    # Edge columns surfaced too.
    assert "edge_attr_columns" in manifest
    assert "player__plays_for__team" in manifest["edge_attr_columns"]


# ---- export orchestrator ---------------------------------------------------


def test_export_era_creates_registered_completed_run(
    registry: Registry, config_path: Path
) -> None:
    src = build_three_era_source()
    result = export_era(
        src, era_slug="e2024_01", config_path=config_path, registry=registry
    )

    assert result.passed
    assert result.snapshot_path.exists()
    assert result.manifest_path.exists()
    assert result.validation_path.exists()
    record = registry.get(result.run_id)
    assert record.status is RunStatus.COMPLETED
    assert record.kind == GRAPH_SNAPSHOT_KIND


def test_export_eras_three_era_acceptance(
    registry: Registry, config_path: Path
) -> None:
    """BUF-53 acceptance: full graph export for 3 eras passes structural validation."""
    src = build_three_era_source()
    results = export_eras(
        src,
        era_slugs=list(ERA_SLUGS),
        config_path=config_path,
        registry=registry,
    )

    assert len(results) == 3
    assert all(r.passed for r in results)
    # Three distinct run ids — different fingerprints per era.
    assert len({r.run_id for r in results}) == 3


def test_export_eras_runs_in_under_two_minutes(
    registry: Registry, config_path: Path
) -> None:
    """BUF-53 acceptance: export runs in under 2 minutes per era.

    The fixture is small enough that the per-era cost should be in
    the 10s of milliseconds; this test asserts the budget for the
    *whole batch* with two orders of magnitude of headroom so a
    legitimate slowdown trips it before a real era ever does.
    """
    src = build_three_era_source()
    start = time.monotonic()
    export_eras(
        src,
        era_slugs=list(ERA_SLUGS),
        config_path=config_path,
        registry=registry,
    )
    duration = time.monotonic() - start
    assert duration < 30.0, f"3-era export took {duration:.2f}s, budget is 30s"


def test_export_era_is_idempotent(
    registry: Registry, config_path: Path
) -> None:
    """Re-exporting the same era + same source returns the same run_id (no-op)."""
    src = build_three_era_source()
    first = export_era(
        src, era_slug="e2024_01", config_path=config_path, registry=registry
    )
    second = export_era(
        src, era_slug="e2024_01", config_path=config_path, registry=registry
    )
    assert first.run_id == second.run_id
    assert second.passed


def test_manifest_serializes_datetime_metadata(tmp_path: Path) -> None:
    """Codex P1 (PR #24): datetime values from ``patch_meta`` must serialize.

    The ``GraphDataSource.patch_meta`` contract documents that era
    window timestamps may flow through; ``json.dumps`` would otherwise
    raise ``TypeError`` on the manifest write.
    """
    from datetime import UTC, datetime

    from ecosystem.graph.source import InMemoryDataSource

    src = InMemoryDataSource()
    src.set_patch(
        "e_t",
        {
            "era_ordinal": 0.0,
            "starts_at": datetime(2024, 1, 9, tzinfo=UTC),
            "ends_at": datetime(2024, 4, 15, tzinfo=UTC),
        },
    )
    snap = build_snapshot(src, era_slug="e_t")
    out_dir = tmp_path / "snap"
    snap.write(out_dir)

    manifest = json.loads((out_dir / "manifest.json").read_text(encoding="utf-8"))
    patch_meta = manifest["metadata"]["patch_meta"]
    # Datetimes round-trip as ISO strings; the exact format is whatever
    # ``datetime.isoformat()`` produces, so we just confirm it landed
    # as a string rather than crashing the dump.
    assert isinstance(patch_meta["starts_at"], str)
    assert "2024-01-09" in patch_meta["starts_at"]


def test_metadata_change_changes_run_id(
    registry: Registry, config_path: Path
) -> None:
    """Codex P2 (PR #24): a metadata-only source change must mint a new run_id.

    The fingerprint previously hashed only tensors, so a source that
    shifted ``patch_meta.starts_at`` (or any other manifest-only
    field) would short-circuit to the prior run_id and the manifest
    on disk would silently disagree with the source — broken
    auditability.
    """
    from datetime import UTC, datetime

    from ecosystem.graph.source import InMemoryDataSource

    def _build_source(starts_at: datetime) -> InMemoryDataSource:
        s = InMemoryDataSource()
        s.set_patch(
            "e_t",
            {"era_ordinal": 0.0, "starts_at": starts_at},
        )
        return s

    first = export_era(
        _build_source(datetime(2024, 1, 9, tzinfo=UTC)),
        era_slug="e_t",
        config_path=config_path,
        registry=registry,
    )
    second = export_era(
        _build_source(datetime(2024, 1, 10, tzinfo=UTC)),
        era_slug="e_t",
        config_path=config_path,
        registry=registry,
    )
    assert first.run_id != second.run_id, (
        "metadata-only source change must produce a new fingerprint"
    )


def test_failed_run_short_circuits_on_re_export(
    registry: Registry, config_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Codex P1 (PR #24): re-exporting after a FAILED run must not relabel.

    First export forces a validation failure → row finalises FAILED
    with artifacts. Second export hits the same idempotency triple
    so ``register`` returns the existing FAILED row. The orchestrator
    must short-circuit to the cached report (passed=False) rather
    than rebuild and silently report passed=True against a registry
    row that ``Registry.finalize`` cannot transition out of FAILED.
    """
    src = build_three_era_source()

    real_validate = validate_snapshot

    def force_failing_report(snapshot: GraphSnapshot):
        report = real_validate(snapshot)
        from ecosystem.graph.validate import ValidationIssue

        report.issues.append(
            ValidationIssue("injected", "error", "test", "forced fail")
        )
        return report

    monkeypatch.setattr(
        "ecosystem.graph.export.validate_snapshot", force_failing_report
    )
    first = export_era(
        src, era_slug="e2024_01", config_path=config_path, registry=registry
    )
    assert not first.passed
    assert registry.get(first.run_id).status is RunStatus.FAILED

    # Lift the injected failure for the second pass — were the
    # orchestrator to rebuild instead of short-circuiting, the
    # rebuild would now pass and contradict the FAILED row.
    monkeypatch.setattr(
        "ecosystem.graph.export.validate_snapshot", real_validate
    )
    second = export_era(
        src, era_slug="e2024_01", config_path=config_path, registry=registry
    )
    assert second.run_id == first.run_id
    assert not second.passed, (
        "re-export should return the cached FAILED report, not relabel"
    )
    assert registry.get(second.run_id).status is RunStatus.FAILED


def test_terminal_run_with_missing_artifacts_raises(
    registry: Registry, config_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Codex P1 (PR #24): missing artifacts on a terminal row → operator action.

    Rebuilding silently would leave the registry stuck on the
    pre-existing terminal status (since ``Registry.finalize`` no-ops
    on non-RUNNING rows), so the orchestrator surfaces the conflict
    instead of producing contradictory state.
    """
    from ecosystem.graph.export import GraphExportStateError

    src = build_three_era_source()
    first = export_era(
        src, era_slug="e2024_01", config_path=config_path, registry=registry
    )
    assert first.passed
    # Operator manually deleted the artifacts but left the row.
    first.snapshot_path.unlink()
    first.manifest_path.unlink()
    first.validation_path.unlink()

    with pytest.raises(GraphExportStateError):
        export_era(
            src, era_slug="e2024_01", config_path=config_path, registry=registry
        )


def test_failed_validation_writes_failed_status(
    registry: Registry, config_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A snapshot that fails validation still writes artifacts but finalises FAILED."""
    src = build_three_era_source()

    real_validate = validate_snapshot

    def force_failing_report(snapshot: GraphSnapshot):
        report = real_validate(snapshot)
        # Inject a synthetic error so the orchestrator records FAILED.
        from ecosystem.graph.validate import ValidationIssue

        report.issues.append(
            ValidationIssue(
                code="injected",
                severity="error",
                location="test",
                message="forced failure for orchestrator coverage",
            )
        )
        return report

    monkeypatch.setattr(
        "ecosystem.graph.export.validate_snapshot", force_failing_report
    )
    result = export_era(
        src, era_slug="e2024_01", config_path=config_path, registry=registry
    )
    assert not result.passed
    assert result.snapshot_path.exists()
    assert result.validation_path.exists()
    record = registry.get(result.run_id)
    assert record.status is RunStatus.FAILED
