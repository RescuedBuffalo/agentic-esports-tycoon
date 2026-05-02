"""``seed_patch_eras()`` — bootstrap the temporal-partition table (BUF-13).

One-shot loader run once per environment to populate
:class:`~esports_sim.db.models.PatchEra` with the historical Valorant
patch-release timeline (2020 → present). The dataset is hand-curated
in :data:`_VALORANT_ERAS` rather than scraped: the BUF-83 patch-notes
scraper would take many runs to cover six years of history, and the
BUF-13 acceptance ("100% of historical matches successfully assigned")
needs the table populated *before* any historical match lands. So the
seed is the source of truth, the steady-state path is :func:`roll_era`
(called from a future BUF-24 patch-intent extractor when a new patch
notes article lands).

Idempotency contract: the second invocation is a no-op. Each row is
keyed on its ``era_slug``; the seed reads what's already there before
inserting, and the manifest distinguishes ``inserted`` from
``existing`` so an operator can prove the second run added zero rows.

Major-shift policy: a patch_era row is marked ``is_major_shift=True``
when the underlying patch ships ANY of:

* a new agent (Iso, Clove, Tejo, Waylay, etc.)
* a new map or a major map rework
* an episode rollover (X.0 patches)

These are the boundaries the
:func:`esports_sim.eras.assert_no_temporal_bleed` guard treats as
hard splits — aggregating across one is forbidden because the meta
moved enough that pre-/post- stats are measuring different games.
The 2024+ subset of these dates aligns with the hand-curated
``config/patch_eras.yaml`` boundaries; the 2020-2023 subset comes
from the Valorant patch history (Riot's release notes archive).

Out of scope:

* No raw_record / staging_record writes. Era rows aren't entities;
  they're partition metadata, and audit replay of the seed itself
  lives on the manifest.
* No automatic boundary detection from BUF-83 patch notes. That's
  BUF-24's job — when a new patch notes article lands and the
  intent extractor recognises a major-shift signal, it calls
  :func:`esports_sim.eras.roll_era` to close the current era and
  open a new one. The seed only handles the historical backfill.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from esports_sim.db.models import PatchEra
from esports_sim.eras import current_era, open_new_era
from sqlalchemy import select
from sqlalchemy.orm import Session

_logger = logging.getLogger("data_pipeline.seeds.patch_eras")

# Default seed manifest location; mirrors the VLR seed.
DEFAULT_SEEDS_DIR = Path("seeds")


@dataclass(frozen=True)
class _EraSpec:
    """One row of the seeded patch-era timeline.

    ``end_date`` is computed from the *next* spec's ``start_date`` at
    seed time — keeping the spec list as start-only avoids the silent
    bug where the operator updates one ``start_date`` but forgets to
    update the previous spec's ``end_date``.
    """

    era_slug: str
    patch_version: str
    start_date: date
    is_major_shift: bool
    meta_magnitude: float


# Hand-curated Valorant patch-era timeline. Each entry's ``end_date``
# is the *next* entry's ``start_date`` (computed in
# :func:`_build_planned_rows`), so the timeline tiles
# [first.start, +infinity) with no gap and no overlap. The last entry
# has ``end_date=None`` — that's the open era.
#
# Slugs follow the ``eYYYY_NN`` pattern from
# ``config/patch_eras.yaml`` so the 2024+ rows land at the same slugs
# operators are already grepping for. Pre-2024 slugs use the same
# pattern (eYYYY_NN, NN incrementing per calendar year).
#
# ``meta_magnitude`` is a 0..1 estimate of how much the meta shifted
# at the start of this era. Conventions:
#   - Game launch / Episode rollover: 0.95
#   - New agent or major map: 0.80–0.85
#   - Mid-act balance patch: 0.30–0.50
# These are deliberately coarse — the BUF-13 guard is binary
# (is_major_shift), the magnitude is a hint for downstream
# magnitude-weighted aggregations.
_VALORANT_ERAS: tuple[_EraSpec, ...] = (
    # --- Episode 1: launch through Skye -------------------------------
    _EraSpec("e2020_01", "1.0", date(2020, 6, 2), True, 0.95),
    _EraSpec("e2020_02", "1.05", date(2020, 8, 4), True, 0.80),  # Killjoy
    _EraSpec("e2020_03", "1.09", date(2020, 9, 29), True, 0.75),  # Icebox
    _EraSpec("e2020_04", "1.11", date(2020, 10, 27), True, 0.80),  # Skye
    # --- Episode 2: Yoru / Astra / Breeze -----------------------------
    _EraSpec("e2021_01", "2.0", date(2021, 1, 12), True, 0.85),  # Yoru
    _EraSpec("e2021_02", "2.04", date(2021, 3, 2), True, 0.80),  # Astra
    _EraSpec("e2021_03", "2.08", date(2021, 4, 27), True, 0.80),  # Breeze
    # --- Episode 3: KAY/O / Fracture / Chamber ------------------------
    _EraSpec("e2021_04", "3.0", date(2021, 6, 22), True, 0.85),  # KAY/O
    _EraSpec("e2021_05", "3.05", date(2021, 9, 8), True, 0.80),  # Fracture
    _EraSpec("e2021_06", "3.10", date(2021, 11, 16), True, 0.80),  # Chamber
    # --- Episode 4: Neon / Pearl prep ---------------------------------
    _EraSpec("e2022_01", "4.0", date(2022, 1, 11), True, 0.85),  # Neon
    _EraSpec("e2022_02", "4.04", date(2022, 3, 8), False, 0.40),
    _EraSpec("e2022_03", "4.08", date(2022, 5, 11), False, 0.30),
    # --- Episode 5: Fade / Pearl / Harbor -----------------------------
    _EraSpec("e2022_04", "5.0", date(2022, 6, 22), True, 0.85),  # Fade + Pearl
    _EraSpec("e2022_05", "5.05", date(2022, 8, 24), False, 0.40),
    _EraSpec("e2022_06", "5.08", date(2022, 10, 18), True, 0.80),  # Harbor
    # --- Episode 6: Lotus ---------------------------------------------
    _EraSpec("e2023_01", "6.0", date(2023, 1, 10), True, 0.85),  # Lotus
    _EraSpec("e2023_02", "6.04", date(2023, 2, 28), False, 0.40),
    _EraSpec("e2023_03", "6.08", date(2023, 4, 25), False, 0.30),
    # --- Episode 7: Deadlock + Sunset prep ---------------------------
    _EraSpec("e2023_04", "7.0", date(2023, 6, 27), True, 0.85),  # Deadlock
    _EraSpec("e2023_05", "7.04", date(2023, 8, 29), False, 0.40),
    _EraSpec("e2023_06", "7.08", date(2023, 10, 31), True, 0.80),  # Sunset
    _EraSpec("e2023_07", "7.12", date(2023, 12, 12), False, 0.30),
    # --- Episode 8 / 2024 (aligned with config/patch_eras.yaml) -------
    _EraSpec("e2024_01", "8.0", date(2024, 1, 9), True, 0.85),  # Iso
    _EraSpec("e2024_02", "8.08", date(2024, 4, 15), True, 0.85),  # Clove
    _EraSpec("e2024_03", "9.02", date(2024, 8, 5), True, 0.80),  # Abyss / Champions cycle
    # --- 2025 (aligned with config/patch_eras.yaml) -------------------
    _EraSpec("e2025_01", "10.00", date(2025, 1, 6), True, 0.85),  # Tejo
    _EraSpec("e2025_02", "10.07", date(2025, 6, 23), True, 0.85),  # Waylay
    # --- 2026 (current era — open) ------------------------------------
    _EraSpec("e2026_01", "11.03", date(2026, 1, 12), True, 0.80),
)


@dataclass
class _CurrentCounters:
    """Running totals for the seed manifest."""

    planned: int = 0
    inserted: int = 0
    existing: int = 0
    updated: int = 0


@dataclass
class PatchEraSeedManifest:
    """Auditable record of one ``seed_patch_eras`` invocation.

    Persisted to ``{seeds_dir}/patch_eras_seed_{seed_date}.json`` so
    an operator can diff manifests across runs and prove the BUF-13
    acceptance ("no null era_id") held the first time and that
    subsequent re-runs added zero rows.
    """

    seed_date: str
    started_at: str
    finished_at: str
    counters: _CurrentCounters = field(default_factory=_CurrentCounters)
    earliest_start_date: str | None = None
    latest_start_date: str | None = None
    open_era_slug: str | None = None

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def seed_patch_eras(
    session: Session,
    *,
    seeds_dir: Path | None = None,
    today: date | None = None,
    write_manifest: bool = True,
) -> PatchEraSeedManifest:
    """Bootstrap the patch_era table from the curated Valorant timeline.

    Idempotent: each spec is keyed on its ``era_slug``. If the row
    already exists, it's left alone (no UPDATE, even if ``end_date``
    or ``meta_magnitude`` differs — the steady-state path is
    :func:`roll_era`, not the seed). The manifest's
    ``existing`` counter records the no-op count so a re-run is
    self-documenting.

    Caller owns the transaction. The seed flushes after each insert
    so an EXCLUDE-constraint violation on a malformed dataset surfaces
    on the offending row, not on the outer commit.

    Returns :class:`PatchEraSeedManifest`. The manifest's
    ``open_era_slug`` is the slug of the row whose ``end_date`` is
    null after the seed completes — exactly one if the dataset is
    well-formed.
    """
    seed_date = today or datetime.now(UTC).date()
    started_at = datetime.now(UTC)
    manifest = PatchEraSeedManifest(
        seed_date=seed_date.isoformat(),
        started_at=started_at.isoformat(),
        finished_at="",
    )

    planned = _build_planned_rows(_VALORANT_ERAS)
    manifest.counters.planned = len(planned)
    manifest.earliest_start_date = planned[0]["start_date"].isoformat() if planned else None
    manifest.latest_start_date = planned[-1]["start_date"].isoformat() if planned else None

    # Pre-load existing slugs in one query so the per-spec branch is
    # an in-memory lookup, not N round-trips. The seed dataset is
    # tiny (<100 rows) so the upper bound on memory is trivial.
    existing_by_slug: dict[str, PatchEra] = {
        row.era_slug: row for row in session.execute(select(PatchEra)).scalars()
    }

    for spec in planned:
        slug = spec["era_slug"]
        if slug in existing_by_slug:
            manifest.counters.existing += 1
            continue

        # New row. ``open_new_era`` is for the open-current path; for
        # historical rows we want to set end_date directly, so
        # construct the row inline. The EXCLUDE constraint will catch
        # a mistakenly-overlapping insert at flush time.
        row = PatchEra(
            era_id=uuid.uuid4(),
            era_slug=slug,
            patch_version=spec["patch_version"],
            start_date=spec["start_date"],
            end_date=spec["end_date"],
            meta_magnitude=spec["meta_magnitude"],
            is_major_shift=spec["is_major_shift"],
        )
        session.add(row)
        session.flush()
        manifest.counters.inserted += 1

    # Report the open era from the DB after the seed completes, not from
    # whichever planned spec happened to land with ``end_date=None``.
    # In steady state, ``roll_era`` opens new eras *not* present in the
    # curated ``_VALORANT_ERAS`` list — a re-run of the seed would then
    # see all planned slugs as ``existing`` and report
    # ``open_era_slug=None`` even though an open row clearly exists,
    # misleading any operational check that grep'd the manifest. Query
    # the partial-unique-indexed open row directly so the manifest is
    # always truthful.
    open_row = current_era(session)
    manifest.open_era_slug = open_row.era_slug if open_row is not None else None
    manifest.finished_at = datetime.now(UTC).isoformat()

    if write_manifest:
        target_dir = seeds_dir if seeds_dir is not None else DEFAULT_SEEDS_DIR
        _persist_manifest(manifest, target_dir)

    _logger.info(
        "patch_eras_seed.done planned=%d inserted=%d existing=%d open_era=%s",
        manifest.counters.planned,
        manifest.counters.inserted,
        manifest.counters.existing,
        manifest.open_era_slug,
    )
    return manifest


def _build_planned_rows(specs: Sequence[_EraSpec]) -> list[dict[str, Any]]:
    """Materialise the spec list with ``end_date`` derived from neighbours.

    Each entry's ``end_date`` is the next entry's ``start_date``,
    with the last entry left open (``end_date=None``). The dates land
    as midnight UTC so the timezone-aware column accepts them
    directly. Rejects a non-monotonically-increasing dataset early —
    a silently-misordered spec list would otherwise trip the EXCLUDE
    constraint at flush with a less actionable error.
    """
    if not specs:
        return []
    sorted_specs = sorted(specs, key=lambda s: s.start_date)
    if [s.start_date for s in sorted_specs] != [s.start_date for s in specs]:
        raise ValueError(
            "patch_eras seed dataset is not strictly increasing by start_date — "
            "the spec list must be hand-ordered so re-runs are deterministic."
        )

    rows: list[dict[str, Any]] = []
    for i, spec in enumerate(sorted_specs):
        start = datetime.combine(spec.start_date, datetime.min.time(), tzinfo=UTC)
        if i + 1 < len(sorted_specs):
            next_start = sorted_specs[i + 1].start_date
            if next_start <= spec.start_date:
                raise ValueError(
                    f"patch_eras seed dataset has duplicate or non-monotonic "
                    f"start_date at index {i}: {spec.start_date} >= "
                    f"{next_start}"
                )
            end: datetime | None = datetime.combine(next_start, datetime.min.time(), tzinfo=UTC)
        else:
            end = None
        rows.append(
            {
                "era_slug": spec.era_slug,
                "patch_version": spec.patch_version,
                "start_date": start,
                "end_date": end,
                "is_major_shift": spec.is_major_shift,
                "meta_magnitude": spec.meta_magnitude,
            }
        )
    return rows


def _persist_manifest(manifest: PatchEraSeedManifest, seeds_dir: Path) -> Path:
    """Write the manifest JSON; return the file path."""
    seeds_dir.mkdir(parents=True, exist_ok=True)
    target = seeds_dir / f"patch_eras_seed_{manifest.seed_date}.json"
    with target.open("w", encoding="utf-8") as fh:
        json.dump(manifest.to_json(), fh, indent=2, sort_keys=True)
        fh.write("\n")
    return target


# Re-export ``open_new_era`` for the rare seed caller that wants to
# *append* a brand-new open era after the curated dataset (e.g. when
# Riot ships a patch a few days before this module's spec list is
# updated). The steady-state caller should still use
# :func:`esports_sim.eras.roll_era`.
__all__ = [
    "DEFAULT_SEEDS_DIR",
    "PatchEraSeedManifest",
    "open_new_era",
    "seed_patch_eras",
]
