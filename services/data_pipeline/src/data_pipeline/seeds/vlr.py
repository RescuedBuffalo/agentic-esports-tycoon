"""``seed_from_vlr_csv()`` — bootstrap match history from a VLR.gg scrape (BUF-8 v2).

One-shot import that reads a community-scraped VLR.gg map-level CSV
and lands canonical TEAM + TOURNAMENT entities plus every match and
map row into the relational store. Replaces the original Liquipedia
seed (which targeted a fictional REST API).

Why a CSV path: VLR has no public API and live scraping is gated by
Cloudflare; the steady-state :class:`~data_pipeline.connectors.vlr.VLRConnector`
uses a headless browser at polite cadence. A from-scratch backfill of
~129k map rows over ~5 years of VAL competitive history would take
weeks at that cadence, so the bootstrap consumes a one-shot offline
snapshot. Subsequent incremental work runs against the live VLR
connector and only catches up the gap since the snapshot.

Decision tree per row:

* The bootstrap **does not** consult the fuzzy matcher in
  :mod:`esports_sim.resolver`. Two distinct VLR ids are by definition
  two distinct entities — there's no cross-platform identity to
  consolidate yet. Running fuzzy on placeholder names (e.g.
  ``"vlr-event-2158"`` vs ``"vlr-event-2159"``) would misfire because
  the strings are 90%+ identical and would auto-merge unrelated
  events. We use a direct create-or-get path keyed on
  ``(Platform.VLR, vlr_id)`` instead, which is what the resolver's
  exact-alias lookup would do anyway in the steady state.
* Match-level idempotency is on ``vlr_match_id``; map-level on
  ``vlr_game_id``. Pre-loading both sets up front lets a re-run no-op
  on existing rows without paying a per-row SELECT.
* Sentinel rows (date 1970-01-01, an upstream null marker) are
  dropped. ``Team{1,2}ID == "0"`` is treated as a TBD/forfeit
  opponent — the row still lands but its team FK is null.

Out of scope:

* Player canonical entities. The CSV's ``Team1Game1..Team2Game5``
  columns are recent-form match-id history, not roster slots; per-map
  player participation will come from a follow-up VLR ``/match/<id>``
  scraper.
* ``RawRecord`` / ``StagingRecord`` writes. The seed predates the
  steady-state pipeline; its only purpose is to populate canonical
  entities + match history. Staging tables are for incremental
  traffic only.
* Fuzzy retry / cross-source rebrand handling. A team rename keeps
  the same VLR numeric id, so the (platform_id) anchor stays put;
  the alias's ``platform_name`` will lag the live UI until the next
  enrichment pass overwrites it.
"""

from __future__ import annotations

import csv
import json
import logging
import uuid
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from esports_sim.db.enums import EntityType, Platform
from esports_sim.db.models import Entity, EntityAlias, MapResult, Match
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from data_pipeline.connectors.vlr import vlr_alias_platform_id

_logger = logging.getLogger("data_pipeline.seeds.vlr")

# Where the manifest file lands. Operators can override; default is
# the top-level ``seeds/`` directory the README pins as the seed-
# artifact location.
DEFAULT_SEEDS_DIR = Path("seeds")

# CSV columns that carry per-team aggregate stats. Names follow the
# upstream CSV header verbatim (``Team1 ACS``, ``Team2 Kills``, …).
# Centralised here so the JSONB blob shape is easy to read off, and
# so a future schema change in the upstream CSV surfaces as a single
# KeyError rather than 18 silent string mismatches.
_TEAM_STAT_FIELDS: tuple[str, ...] = (
    "ACS",
    "Kills",
    "Deaths",
    "Assists",
    "DeltaK/D",
    "KAST",
    "ADR",
    "HS",
    "FK",
    "FD",
    "DeltaFK/FD",
    "Pistols",
    "EcosWon",
    "Ecos",
    "Semibuys Won",
    "Semibuys",
    "Fullbuys Won",
    "Fullbuys",
)

# Date sentinels we treat as upstream nulls and skip outright. The
# 1970-01-01 epoch zero is what the original scraper emits when the
# match page lacks a parseable date.
_SENTINEL_DATES: frozenset[str] = frozenset({"1970-01-01"})

# Treated as a TBD opponent or unresolved tournament. We still land
# the row but leave the FK null rather than drop the data.
_NULL_VLR_ID: frozenset[str] = frozenset({"0", ""})


@dataclass
class _EntityCounters:
    """Per-entity-type running totals for the manifest."""

    discovered: int = 0
    created: int = 0
    existing: int = 0


@dataclass
class _MatchCounters:
    """Match + map row totals."""

    matches_seen: int = 0
    matches_inserted: int = 0
    matches_existing: int = 0
    maps_seen: int = 0
    maps_inserted: int = 0
    maps_existing: int = 0
    rows_skipped_sentinel: int = 0
    rows_skipped_malformed: int = 0


@dataclass
class VlrSeedManifest:
    """Auditable record of one ``seed_from_vlr_csv`` invocation.

    Persisted to ``{seeds_dir}/vlr_seed_{seed_date}.json`` so an
    operator can diff manifests across runs and confirm idempotency
    (a second pass adds zero rows; ``inserted`` zeros out and
    ``existing`` carries the totals).
    """

    seed_date: str
    started_at: str
    finished_at: str
    csv_path: str
    teams: _EntityCounters = field(default_factory=_EntityCounters)
    tournaments: _EntityCounters = field(default_factory=_EntityCounters)
    matches: _MatchCounters = field(default_factory=_MatchCounters)

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def seed_from_vlr_csv(
    session: Session,
    *,
    csv_path: Path,
    seeds_dir: Path | None = None,
    today: date | None = None,
    write_manifest: bool = True,
) -> VlrSeedManifest:
    """Bootstrap canonical entities + match history from a VLR CSV.

    The walk is two-pass: the first pass collects distinct team and
    tournament ids (so we can resolve canonicals once each before
    touching match rows); the second pass walks the CSV again and
    writes :class:`Match` / :class:`MapResult` rows. Matches are
    grouped by ``MatchID`` so each series gets exactly one ``match``
    row regardless of how many maps it spans.

    Caller owns the transaction. The seed ``flush()``es as it goes
    so existing-row checks within a run are correct; it does not
    ``commit``. Wrap in :meth:`Session.begin` to get all-or-nothing
    semantics; the operator script default is "commit at end of
    seed".
    """
    seed_date = today or datetime.now(UTC).date()
    started_at = datetime.now(UTC)

    manifest = VlrSeedManifest(
        seed_date=seed_date.isoformat(),
        started_at=started_at.isoformat(),
        finished_at="",  # filled in after the second pass
        csv_path=str(csv_path),
    )

    # Pass 1: distinct entity ids. Keep the most-recently-seen team
    # display name as the platform_name (CSV rows for the same team
    # id usually agree, but minor casing tweaks happen across years
    # and the latest snapshot is the freshest source).
    teams_to_resolve: dict[str, str] = {}
    tournaments_to_resolve: set[str] = set()
    for row in _iter_rows(csv_path):
        if _is_sentinel(row):
            continue
        for vlr_id_key, name_key in (("Team1ID", "Team1 Name"), ("Team2ID", "Team2 Name")):
            vlr_id = row[vlr_id_key]
            if vlr_id not in _NULL_VLR_ID:
                teams_to_resolve[vlr_id] = row[name_key]
        event_id = row["EventID"]
        if event_id not in _NULL_VLR_ID:
            tournaments_to_resolve.add(event_id)
    manifest.teams.discovered = len(teams_to_resolve)
    manifest.tournaments.discovered = len(tournaments_to_resolve)

    # Pre-load existing VLR aliases so we can decide create-or-get
    # in O(1) per row rather than a SELECT-per-id round-trip.
    existing_alias_canonical = _load_existing_vlr_aliases(session)

    # Tuple-keyed lookup so the team and tournament namespaces stay
    # disjoint by entity type. VLR's per-resource id spaces overlap on
    # the integer axis (a team and an event can share the same raw
    # numeric id), so a single ``vlr_id -> canonical`` dict would be
    # ambiguous. The platform_id we hand the alias store goes through
    # :func:`vlr_alias_platform_id` for the same reason at the row
    # level — the alias unique constraint is on ``(platform, platform_id)``
    # alone, and we'd silently overwrite one entity with the other if
    # we let raw VLR ids land directly.
    canonical_for: dict[tuple[EntityType, str], uuid.UUID] = {}
    for vlr_id, display_name in teams_to_resolve.items():
        cid, was_created = _create_or_get_canonical(
            session,
            platform=Platform.VLR,
            platform_id=vlr_alias_platform_id(EntityType.TEAM, vlr_id),
            platform_name=display_name,
            entity_type=EntityType.TEAM,
            existing=existing_alias_canonical,
        )
        canonical_for[(EntityType.TEAM, vlr_id)] = cid
        if was_created:
            manifest.teams.created += 1
        else:
            manifest.teams.existing += 1
    for vlr_id in tournaments_to_resolve:
        # The CSV doesn't carry tournament names — those land on a
        # later enrichment pass via the live VLR scraper. Use a
        # placeholder so the resolver invariant (non-empty name) is
        # met without faking real data.
        cid, was_created = _create_or_get_canonical(
            session,
            platform=Platform.VLR,
            platform_id=vlr_alias_platform_id(EntityType.TOURNAMENT, vlr_id),
            platform_name=f"vlr-event-{vlr_id}",
            entity_type=EntityType.TOURNAMENT,
            existing=existing_alias_canonical,
        )
        canonical_for[(EntityType.TOURNAMENT, vlr_id)] = cid
        if was_created:
            manifest.tournaments.created += 1
        else:
            manifest.tournaments.existing += 1
    session.flush()

    # Pass 2: match + map_result rows. Pre-load the full
    # ``vlr_match_id -> match_id`` mapping (not just the set of ids)
    # so a re-run that lands a new map under an already-seeded match
    # has its parent UUID in hand without an extra SELECT. Same for
    # the existing-game-id set, which only needs membership checks.
    # SQLAlchemy 2.0 ``Row`` objects are tuple-like at runtime but
    # typed as ``Row[tuple[str, UUID]]``, which mypy refuses to feed
    # to ``dict()``. Iterate explicitly so the runtime tuple-unpack
    # is independent of the typed-result wrapper — no risk of a
    # ``.tuples()`` quirk in a particular SQLAlchemy minor version
    # changing semantics.
    match_canonical_by_vlr_id: dict[str, uuid.UUID] = {
        row[0]: row[1] for row in session.execute(select(Match.vlr_match_id, Match.match_id)).all()
    }
    pre_existing_match_ids: frozenset[str] = frozenset(match_canonical_by_vlr_id)
    existing_game_ids: set[str] = set(
        session.execute(select(MapResult.vlr_game_id)).scalars().all()
    )
    # Track which match ids we've already counted toward the
    # per-match counters this run. Without this, every map row in a
    # Bo3 series would re-increment matches_seen / matches_existing.
    seen_match_ids: set[str] = set()

    for row in _iter_rows(csv_path):
        if _is_sentinel(row):
            manifest.matches.rows_skipped_sentinel += 1
            continue
        try:
            match_record, map_record = _build_match_and_map(
                row,
                canonical_for=canonical_for,
                match_canonical_by_vlr_id=match_canonical_by_vlr_id,
                pre_existing_match_ids=pre_existing_match_ids,
                seen_match_ids=seen_match_ids,
                existing_game_ids=existing_game_ids,
                counters=manifest.matches,
            )
        except _RowMalformed as exc:
            _logger.warning(
                "vlr_seed.row_malformed match_id=%s game_id=%s detail=%s",
                row.get("MatchID", "<unknown>"),
                row.get("GameID", "<unknown>"),
                exc,
            )
            manifest.matches.rows_skipped_malformed += 1
            continue
        wrote_anything = match_record is not None or map_record is not None
        if match_record is not None:
            session.add(match_record)
            match_canonical_by_vlr_id[match_record.vlr_match_id] = match_record.match_id
            manifest.matches.matches_inserted += 1
        if map_record is not None:
            session.add(map_record)
            existing_game_ids.add(map_record.vlr_game_id)
            manifest.matches.maps_inserted += 1
        # Flush every ~1000 inserts keeps the in-session state
        # consistent for the next iteration's existing-id checks
        # without paying a flush per row. The ``wrote_anything``
        # gate is what makes a full-idempotent re-run cheap: with
        # both counters stuck at 0 the modulo would be true every
        # iteration and we'd flush on every CSV row even though
        # nothing was added. Skipping when nothing was written
        # keeps re-runs at ~zero DB round-trips beyond the
        # initial pre-load SELECTs.
        total_inserted = manifest.matches.maps_inserted + manifest.matches.matches_inserted
        if wrote_anything and total_inserted % 1000 == 0:
            session.flush()
    session.flush()

    manifest.finished_at = datetime.now(UTC).isoformat()

    if write_manifest:
        target_dir = seeds_dir if seeds_dir is not None else DEFAULT_SEEDS_DIR
        _persist_manifest(manifest, target_dir)

    _logger.info(
        "vlr_seed.done csv=%s teams_created=%d tournaments_created=%d "
        "matches_inserted=%d maps_inserted=%d",
        csv_path,
        manifest.teams.created,
        manifest.tournaments.created,
        manifest.matches.matches_inserted,
        manifest.matches.maps_inserted,
    )
    return manifest


# --- CSV iteration --------------------------------------------------------


def _iter_rows(csv_path: Path) -> Iterator[dict[str, str]]:
    """Yield CSV rows as dicts, streaming from disk.

    Streaming (rather than loading everything into memory) keeps
    peak RSS modest on the full ~129k-row CSV — the seeder runs
    inside an operator process whose other workloads we shouldn't
    crowd out.
    """
    # ``utf-8-sig`` strips a possible BOM the upstream scraper may
    # emit on Windows. Plain ``utf-8`` would leave it embedded in
    # the first column name and cause a downstream KeyError.
    with csv_path.open("r", encoding="utf-8-sig", newline="") as fh:
        yield from csv.DictReader(fh)


def _is_sentinel(row: dict[str, str]) -> bool:
    """True for rows we treat as upstream null-equivalent.

    Currently just the 1970-01-01 epoch sentinel; if a future CSV
    revision adds another, list it in :data:`_SENTINEL_DATES`.
    """
    return row.get("Date", "") in _SENTINEL_DATES


# --- canonical entity creation -------------------------------------------


def _load_existing_vlr_aliases(session: Session) -> dict[str, uuid.UUID]:
    """Map every existing ``(Platform.VLR, platform_id) -> canonical_id``.

    Loaded once before the per-row create-or-get loop so the inner
    loop is a hashmap lookup rather than a SELECT per id.
    """
    rows = session.execute(
        select(EntityAlias.platform_id, EntityAlias.canonical_id).where(
            EntityAlias.platform == Platform.VLR
        )
    ).all()
    return {platform_id: canonical_id for platform_id, canonical_id in rows}


def _create_or_get_canonical(
    session: Session,
    *,
    platform: Platform,
    platform_id: str,
    platform_name: str,
    entity_type: EntityType,
    existing: dict[str, uuid.UUID],
) -> tuple[uuid.UUID, bool]:
    """Return ``(canonical_id, was_created)`` for one (platform, id).

    Bypasses :func:`esports_sim.resolver.resolve_entity` deliberately:
    the bootstrap has nothing to gain from fuzzy matching (placeholder
    names like ``vlr-event-2158`` vs ``vlr-event-2159`` would WRatio
    >0.9 and incorrectly auto-merge), and exact-alias lookup is what
    the resolver would do in the no-fuzzy-needed path anyway.

    Race-safety: a savepoint scopes the insert so a concurrent worker
    that already inserted the same alias raises an IntegrityError on
    flush; we catch the (platform, platform_id) violation, re-read the
    alias to learn its canonical_id, and degrade to "existing".
    """
    cached = existing.get(platform_id)
    if cached is not None:
        return cached, False

    try:
        with session.begin_nested():
            entity = Entity(entity_type=entity_type)
            session.add(entity)
            session.flush()  # populate canonical_id
            session.add(
                EntityAlias(
                    canonical_id=entity.canonical_id,
                    platform=platform,
                    platform_id=platform_id,
                    platform_name=platform_name,
                    confidence=1.0,
                )
            )
            session.flush()
    except IntegrityError as exc:
        # Only the alias unique constraint is the race we know how to
        # absorb; any other constraint surfaces (different table,
        # CHECK violation we don't model yet) and propagates.
        if "uq_entity_alias_platform_platform_id" not in str(exc):
            raise
        existing_alias = session.execute(
            select(EntityAlias.canonical_id).where(
                EntityAlias.platform == platform,
                EntityAlias.platform_id == platform_id,
            )
        ).scalar_one()
        existing[platform_id] = existing_alias
        return existing_alias, False

    existing[platform_id] = entity.canonical_id
    return entity.canonical_id, True


# --- match + map_result construction -------------------------------------


class _RowMalformed(ValueError):
    """The row is structurally bad enough that we should skip it.

    Distinct from a sentinel — sentinels are upstream null markers
    we know to drop; malformed rows are unexpected drift the operator
    should see in the manifest's ``rows_skipped_malformed`` bucket.
    """


def _build_match_and_map(
    row: dict[str, str],
    *,
    canonical_for: dict[tuple[EntityType, str], uuid.UUID],
    match_canonical_by_vlr_id: dict[str, uuid.UUID],
    pre_existing_match_ids: frozenset[str],
    seen_match_ids: set[str],
    existing_game_ids: set[str],
    counters: _MatchCounters,
) -> tuple[Match | None, MapResult | None]:
    """Project one CSV row into the ``(Match?, MapResult?)`` it adds.

    Returns ``(None, None)`` only when the row is entirely a no-op
    (idempotent re-run where both the match and the map already
    exist). The match is ``None`` when this isn't the first map of
    the series we encounter — only the first map produces the
    parent ``Match`` row.

    Per-match counters (``matches_seen`` / ``matches_existing``) fire
    exactly once per unique MatchID in the CSV regardless of how many
    map rows the series spans, gated by ``seen_match_ids``.
    """
    counters.maps_seen += 1
    vlr_match_id = row["MatchID"]
    vlr_game_id = row["GameID"]
    if not vlr_match_id or not vlr_game_id:
        raise _RowMalformed("missing MatchID or GameID")

    if vlr_match_id not in seen_match_ids:
        seen_match_ids.add(vlr_match_id)
        counters.matches_seen += 1
        if vlr_match_id in pre_existing_match_ids:
            counters.matches_existing += 1

    match_record: Match | None = None
    if vlr_match_id in match_canonical_by_vlr_id:
        # Either inserted earlier this run or pre-existing in the
        # DB — either way, we have its canonical UUID in the map.
        match_canonical_id = match_canonical_by_vlr_id[vlr_match_id]
    else:
        # First time we've seen this MatchID — build the Match row.
        # The dict update happens at the call-site after the row is
        # added to the session so the next map row in the same series
        # finds it.
        match_record = _construct_match(row, canonical_for=canonical_for)
        match_canonical_id = match_record.match_id

    if vlr_game_id in existing_game_ids:
        counters.maps_existing += 1
        return match_record, None

    map_record = _construct_map_result(row, match_id=match_canonical_id)
    return match_record, map_record


def _construct_match(
    row: dict[str, str],
    *,
    canonical_for: dict[tuple[EntityType, str], uuid.UUID],
) -> Match:
    match_date = _parse_date(row["Date"])
    if match_date is None:
        raise _RowMalformed(f"unparseable date {row['Date']!r}")
    # Look up by (entity_type, raw_vlr_id). The seeder populates
    # canonical_for with disjoint TEAM and TOURNAMENT keys so the same
    # numeric id under each type resolves to its own canonical row —
    # see the namespacing rationale in :func:`seed_from_vlr_csv`.
    return Match(
        match_id=uuid.uuid4(),
        vlr_match_id=row["MatchID"],
        match_date=match_date,
        team1_canonical_id=canonical_for.get((EntityType.TEAM, row["Team1ID"])),
        team2_canonical_id=canonical_for.get((EntityType.TEAM, row["Team2ID"])),
        tournament_canonical_id=canonical_for.get((EntityType.TOURNAMENT, row["EventID"])),
        series_odds=_parse_float(row.get("Series Odds", "")),
        team1_map_odds=_parse_float(row.get("Team1 Map Odds", "")),
    )


def _construct_map_result(row: dict[str, str], *, match_id: uuid.UUID) -> MapResult:
    return MapResult(
        map_result_id=uuid.uuid4(),
        match_id=match_id,
        vlr_game_id=row["GameID"],
        # ``_require_int`` rather than ``_parse_int(...) or 0``: ``0``
        # is a real upstream sentinel for an unplayed/forfeit map, so
        # silently coercing a blank or non-numeric Map column to 0
        # would erase the distinction between "VLR's id space says
        # unplayed" and "the upstream row was corrupt". Routing
        # malformed Map values to ``rows_skipped_malformed`` keeps the
        # data-quality counter truthful.
        vlr_map_id=_require_int(row, "Map"),
        team1_rounds=_require_int(row, "Team1 Rounds"),
        team2_rounds=_require_int(row, "Team2 Rounds"),
        team1_atk_rounds=_require_int(row, "Team1 Atk Rounds"),
        team1_def_rounds=_require_int(row, "Team1 Def Rounds"),
        team2_atk_rounds=_require_int(row, "Team2 Atk Rounds"),
        team2_def_rounds=_require_int(row, "Team2 Def Rounds"),
        team1_rating=_parse_float(row.get("Team1 Rating", "")),
        team2_rating=_parse_float(row.get("Team2 Rating", "")),
        team1_stats=_extract_team_stats(row, prefix="Team1"),
        team2_stats=_extract_team_stats(row, prefix="Team2"),
        round_breakdown=row.get("Round Breakdown") or None,
        vod_url=row.get("VOD Link") or None,
    )


def _extract_team_stats(row: dict[str, str], *, prefix: str) -> dict[str, Any]:
    """Pull every per-team aggregate stat into the JSONB blob.

    Stat keys in the output dict use the raw upstream names (e.g.
    ``"ACS"``, ``"DeltaK/D"``) without the ``Team1``/``Team2``
    prefix — the column belongs to a per-team JSONB blob, so the
    side is implicit. Values stay as floats; missing/blank values
    become ``None`` rather than zero so the absence is preserved.
    """
    stats: dict[str, Any] = {}
    for field_name in _TEAM_STAT_FIELDS:
        raw = row.get(f"{prefix} {field_name}", "")
        stats[field_name] = _parse_float(raw)
    return stats


# --- parsers --------------------------------------------------------------


def _parse_date(value: str) -> datetime | None:
    """Parse a CSV date column into a tz-aware UTC datetime.

    The CSV ships ``YYYY-MM-DD`` (date only). We standardise to
    midnight UTC so downstream era assignment (``assign_era`` is
    half-open ``[start, end)``) treats every match in a day as
    falling in the same era.
    """
    candidate = value.strip()
    if not candidate:
        return None
    try:
        parsed = date.fromisoformat(candidate)
    except ValueError:
        return None
    return datetime(parsed.year, parsed.month, parsed.day, tzinfo=UTC)


def _parse_float(value: str) -> float | None:
    """Permissive float parser: blanks and non-numeric become None."""
    candidate = (value or "").strip()
    if not candidate:
        return None
    try:
        return float(candidate)
    except ValueError:
        return None


def _parse_int(value: str) -> int | None:
    candidate = (value or "").strip()
    if not candidate:
        return None
    try:
        return int(candidate)
    except ValueError:
        # Some CSV rows ship integer columns as floats (e.g.
        # "13.0"). Fall back via float() so we don't reject those —
        # but ONLY when the value is integer-valued. ``int(float(...))``
        # alone would silently truncate ``"13.5"`` to ``13`` and
        # corrupt round totals; we'd rather route the malformed row
        # to ``rows_skipped_malformed`` than ingest a wrong number.
        try:
            as_float = float(candidate)
        except ValueError:
            return None
        if not as_float.is_integer():
            return None
        return int(as_float)


def _require_int(row: dict[str, str], column: str) -> int:
    """Like :func:`_parse_int` but raises :class:`_RowMalformed` on miss.

    Used for round columns where a missing value is genuine corruption
    (a played map without a round count is uninterpretable). The
    explicit raise lands the row in ``rows_skipped_malformed`` rather
    than silently coercing to zero.
    """
    parsed = _parse_int(row.get(column, ""))
    if parsed is None:
        raise _RowMalformed(f"missing or non-numeric {column!r}")
    return parsed


# --- manifest persistence ------------------------------------------------


def _persist_manifest(manifest: VlrSeedManifest, seeds_dir: Path) -> Path:
    """Write the manifest JSON; return the file path.

    Filename is ``vlr_seed_{YYYY-MM-DD}.json`` so an operator can
    ``ls seeds/`` and read off run history at a glance. Two runs on
    the same date overwrite — that's intentional: the file represents
    the *most recent* attempt, and the structured logs carry the
    per-run detail.
    """
    seeds_dir.mkdir(parents=True, exist_ok=True)
    target = seeds_dir / f"vlr_seed_{manifest.seed_date}.json"
    with target.open("w", encoding="utf-8") as fh:
        json.dump(manifest.to_json(), fh, indent=2, sort_keys=True)
        fh.write("\n")
    return target


# Public surface for callers writing
# ``from data_pipeline.seeds.vlr import ...``.
__all__ = [
    "DEFAULT_SEEDS_DIR",
    "VlrSeedManifest",
    "seed_from_vlr_csv",
]
