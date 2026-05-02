"""Tests for the BUF-8 v2 VLR CSV seed.

Two layers, mirroring the patch-eras seed:

* Pure-function tests against the parser helpers — exercise date /
  float / int / stats extraction without touching Postgres.
* Integration tests (gated on ``TEST_DATABASE_URL``) prove the full
  seed flow end-to-end: canonical entity creation through the
  resolver-bypass path, match + map_result row inserts, and the
  BUF-8 idempotency acceptance (re-running the seed adds zero rows).
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from data_pipeline.seeds.vlr import (
    _TEAM_STAT_FIELDS,
    _extract_team_stats,
    _is_sentinel,
    _parse_date,
    _parse_float,
    _parse_int,
    seed_from_vlr_csv,
)
from esports_sim.db.enums import EntityType, Platform
from esports_sim.db.models import Entity, EntityAlias, MapResult, Match
from sqlalchemy import select


# A tiny VLR-shaped CSV exercising:
# * two matches (381258, 381287) — one with two maps (Bo3 partial),
#   the other with one map (Bo1 / partial Bo3);
# * two distinct teams in each match (4 unique team ids);
# * one event id (2158) shared across the first match, a second event
#   id (2159) for the second match — so the manifest's
#   tournament counter is non-trivial;
# * a sentinel 1970-01-01 row that must drop;
# * a row with Team1ID=0 to exercise the TBD-opponent branch (FK
#   stays null but the row still lands).
#
# Field count in each row matches the production CSV header so the
# DictReader keys line up. Most numeric fields are filler — only the
# columns the seed reads are interesting; the rest stays as 0 / "".
_HEADER = (
    "MatchID,GameID,EventID,Date,Team1ID,Team2ID,Team1 Name,Team2 Name,"
    "Series Odds,Team1 Map Odds,Map,Team1 Rounds,Team2 Rounds,"
    "Team1 Atk Rounds,Team2 Atk Rounds,Team1 Def Rounds,Team2 Def Rounds,"
    "Team1 Rating,Team2 Rating,Team1 ACS,Team2 ACS,Team1 Kills,Team2 Kills,"
    "Team1 Deaths,Team2 Deaths,Team1 Assists,Team2 Assists,"
    "Team1 DeltaK/D,Team2 DeltaK/D,Team1 KAST,Team2 KAST,Team1 ADR,Team2 ADR,"
    "Team1 HS,Team2 HS,Team1 FK,Team2 FK,Team1 FD,Team2 FD,"
    "Team1 DeltaFK/FD,Team2 DeltaFK/FD,Team1 Pistols,Team2 Pistols,"
    "Team1 EcosWon,Team2 EcosWon,Team1 Ecos,Team2 Ecos,"
    "Team1 Semibuys Won,Team2 Semibuys Won,Team1 Semibuys,Team2 Semibuys,"
    "Team1 Fullbuys Won,Team2 Fullbuys Won,Team1 Fullbuys,Team2 Fullbuys,"
    "Round Breakdown,Team1Game1,Team1Game2,Team1Game3,Team1Game4,Team1Game5,"
    "Team2Game1,Team2Game2,Team2Game3,Team2Game4,Team2Game5,VOD Link"
)
_ROW_VALORANT_BO3_MAP1 = (
    "381258,180991,2158,2024-09-04,15139,15138,MYVRA,KS Hunters,"
    "0,0,11,13,9,7,6,6,3,1.134,0.87,211.4,178.2,83,70,70,83,39,32,"
    "13,-13,72.8,66.6,134.8,115.8,29.4,29.8,12,10,10,12,2,-2,2,0,"
    "0,0,2,3,4,2,7,4,7,7,11,13,enc,381254,381252,381247,381242,365378,"
    "381253,381251,381249,381243,365377,https://example.com/m1g1"
)
_ROW_VALORANT_BO3_MAP2 = (
    "381258,180992,2158,2024-09-04,15139,15138,MYVRA,KS Hunters,"
    "0,0,1,1,13,10,4,2,9,1.066,0.95,211.2,198.2,87,82,82,87,42,27,"
    "5,-5,73.8,68.8,136.8,132.0,26,30,11,12,12,11,-1,1,1,1,2,1,2,4,"
    "2,0,5,5,8,8,14,12,enc,381254,381252,381247,381242,365378,"
    "381253,381251,381249,381243,365377,https://example.com/m1g2"
)
_ROW_SECOND_MATCH = (
    "381287,181075,2159,2024-09-04,7413,9205,FiRePOWER,LEVIATAN GC,"
    "2.4,2.25,11,4,13,1,9,3,4,0.628,1.284,150.2,216.2,41,70,72,41,"
    "8,18,-31,29,53,83.2,105,144,30.6,34.2,7,10,10,7,-3,3,1,1,0,1,"
    "3,1,2,1,7,3,1,10,5,11,enc,0,0,0,0,0,0,0,0,0,0,https://example.com/m2"
)
_ROW_SENTINEL = (
    "999999,999999,9999,1970-01-01,1,2,SentinelTeamA,SentinelTeamB,"
    "0,0,1,0,0,0,0,0,0,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,enc,"
    "0,0,0,0,0,0,0,0,0,0,"
)
_ROW_TBD_OPPONENT = (
    # Team2ID=0 — TBD/forfeit case; row should still land.
    "381999,180999,2158,2024-09-05,15139,0,MYVRA,TBD,"
    "0,0,1,13,0,7,0,6,0,1.0,0.0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,"
    "0,0,0,0,0,0,0,0,0,0,0,0,0,0,enc,0,0,0,0,0,0,0,0,0,0,"
)


def _make_csv(tmp_path: Path, *, rows: list[str]) -> Path:
    """Materialise a mini CSV at ``tmp_path/sample.csv`` and return the path."""
    target = tmp_path / "sample.csv"
    target.write_text("\n".join([_HEADER, *rows]) + "\n", encoding="utf-8")
    return target


# --- pure-function tests -----------------------------------------------------


def test_parse_date_handles_iso_and_blank() -> None:
    assert _parse_date("2024-09-04") == datetime(2024, 9, 4, tzinfo=UTC)
    assert _parse_date("") is None
    assert _parse_date("not-a-date") is None


def test_parse_float_handles_blank_and_garbage() -> None:
    assert _parse_float("1.134") == pytest.approx(1.134)
    assert _parse_float("") is None
    assert _parse_float("   ") is None
    assert _parse_float("nope") is None


def test_parse_int_coerces_floats() -> None:
    assert _parse_int("13") == 13
    # Some upstream rows ship integer-valued floats (e.g. "13.0").
    # ``_parse_int`` should accept them rather than erroring out and
    # losing the row.
    assert _parse_int("13.0") == 13
    assert _parse_int("") is None
    assert _parse_int("garbage") is None


def test_is_sentinel_drops_only_known_dates() -> None:
    assert _is_sentinel({"Date": "1970-01-01"}) is True
    assert _is_sentinel({"Date": "2024-09-04"}) is False
    assert _is_sentinel({"Date": ""}) is False  # blank ≠ sentinel
    assert _is_sentinel({}) is False


def test_extract_team_stats_returns_canonical_key_set() -> None:
    """The seed's JSONB blob shape is documented by ``_TEAM_STAT_FIELDS``.

    Future readers (analytics views, ML feature builders) will pick
    keys out of these blobs; the test pins the key set so a typo in
    the field list surfaces as a fail rather than a quietly-renamed
    column nobody reads anymore.
    """
    row = {
        "Team1 ACS": "211.4",
        "Team1 Kills": "83",
        "Team1 Deaths": "70",
        "Team1 KAST": "72.8",
        # The remaining stat columns are deliberately omitted — the
        # extractor should fill them as ``None`` rather than KeyError.
    }
    stats = _extract_team_stats(row, prefix="Team1")
    assert set(stats.keys()) == set(_TEAM_STAT_FIELDS)
    assert stats["ACS"] == pytest.approx(211.4)
    assert stats["Kills"] == pytest.approx(83.0)
    assert stats["KAST"] == pytest.approx(72.8)
    # Missing column → None, not 0.0; preserves "value absent" vs
    # "value zero" distinction for the analytics layer.
    assert stats["ADR"] is None


# --- integration tests (Postgres) --------------------------------------------

pytestmark_integration = pytest.mark.integration


@pytest.mark.integration
def test_seed_creates_canonical_entities_and_match_rows(
    db_session,
    tmp_path: Path,
) -> None:
    """End-to-end: 3 real rows + 1 sentinel + 1 TBD; entities + matches land.

    Acceptance assertions:

    * TEAM canonical entities exist for every distinct non-zero
      ``Team{1,2}ID`` in the CSV.
    * TOURNAMENT canonicals exist for every distinct ``EventID``.
    * Each unique ``MatchID`` produces exactly one ``match`` row,
      with FK columns wired to the right canonical_ids.
    * Each unique ``GameID`` produces exactly one ``map_result`` row,
      grouped under its parent match.
    * The sentinel row drops (date 1970-01-01).
    * The TBD-opponent row lands with a null ``team2_canonical_id``.
    """
    csv_path = _make_csv(
        tmp_path,
        rows=[
            _ROW_VALORANT_BO3_MAP1,
            _ROW_VALORANT_BO3_MAP2,
            _ROW_SECOND_MATCH,
            _ROW_SENTINEL,
            _ROW_TBD_OPPONENT,
        ],
    )

    manifest = seed_from_vlr_csv(
        db_session,
        csv_path=csv_path,
        seeds_dir=tmp_path / "seeds-out",
    )

    # Distinct teams: {15139, 15138, 7413, 9205} = 4. Team2ID=0 from
    # the TBD row doesn't count (it's the TBD sentinel).
    assert manifest.teams.discovered == 4
    assert manifest.teams.created == 4
    # Distinct events: {2158, 2159}.
    assert manifest.tournaments.discovered == 2
    assert manifest.tournaments.created == 2
    # 3 unique MatchIDs (sentinel skipped, TBD counted).
    assert manifest.matches.matches_seen == 3
    assert manifest.matches.matches_inserted == 3
    # 4 unique GameIDs land (one sentinel drops).
    assert manifest.matches.maps_inserted == 4
    assert manifest.matches.rows_skipped_sentinel == 1

    # Schema-level assertions — direct DB queries, not via the
    # manifest, so a manifest bug can't mask a missing row.
    teams = db_session.execute(
        select(Entity).where(Entity.entity_type == EntityType.TEAM)
    ).scalars().all()
    assert len(teams) == 4

    aliases = db_session.execute(
        select(EntityAlias).where(EntityAlias.platform == Platform.VLR)
    ).scalars().all()
    # 4 teams + 2 events = 6 VLR aliases, all at confidence 1.0
    assert len(aliases) == 6
    assert all(a.confidence == 1.0 for a in aliases)

    matches = db_session.execute(select(Match)).scalars().all()
    assert {m.vlr_match_id for m in matches} == {"381258", "381287", "381999"}
    # The TBD-opponent match: team2 is null, team1 is wired up.
    tbd = next(m for m in matches if m.vlr_match_id == "381999")
    assert tbd.team1_canonical_id is not None
    assert tbd.team2_canonical_id is None

    maps = db_session.execute(select(MapResult)).scalars().all()
    assert {mr.vlr_game_id for mr in maps} == {"180991", "180992", "181075", "180999"}
    # Spot-check a known row's typed columns + JSONB blob.
    map1 = next(mr for mr in maps if mr.vlr_game_id == "180991")
    assert map1.team1_rounds == 13
    assert map1.team2_rounds == 9
    assert map1.team1_rating == pytest.approx(1.134)
    assert map1.vod_url == "https://example.com/m1g1"
    assert map1.team1_stats["ACS"] == pytest.approx(211.4)
    assert map1.team1_stats["Kills"] == pytest.approx(83.0)


@pytest.mark.integration
def test_seed_is_idempotent_on_re_run(db_session, tmp_path: Path) -> None:
    """BUF-8 acceptance: a second pass over the same CSV adds zero rows.

    The resolver bypass path keys on ``(Platform.VLR, platform_id)``;
    pre-existing match rows are detected via the
    ``vlr_match_id`` map preloaded at the start of pass 2; pre-existing
    map rows are caught via ``vlr_game_id``. The second run should
    therefore touch zero new rows in any of the three tables.
    """
    csv_path = _make_csv(
        tmp_path,
        rows=[
            _ROW_VALORANT_BO3_MAP1,
            _ROW_VALORANT_BO3_MAP2,
            _ROW_SECOND_MATCH,
        ],
    )

    first = seed_from_vlr_csv(
        db_session,
        csv_path=csv_path,
        seeds_dir=tmp_path / "seeds-1",
    )
    second = seed_from_vlr_csv(
        db_session,
        csv_path=csv_path,
        seeds_dir=tmp_path / "seeds-2",
    )

    # Round 1: the seed found everything new.
    assert first.teams.created == 4
    assert first.tournaments.created == 2
    assert first.matches.matches_inserted == 2
    assert first.matches.maps_inserted == 3

    # Round 2: every entity is now in the existing bucket, no new
    # rows are inserted into any of the three tables.
    assert second.teams.created == 0
    assert second.teams.existing == 4
    assert second.tournaments.created == 0
    assert second.tournaments.existing == 2
    assert second.matches.matches_inserted == 0
    assert second.matches.matches_existing == 2
    assert second.matches.maps_inserted == 0
    assert second.matches.maps_existing == 3

    # Cross-check at the DB level: total row counts unchanged.
    matches_count = db_session.execute(select(Match)).scalars().all()
    maps_count = db_session.execute(select(MapResult)).scalars().all()
    assert len(matches_count) == 2
    assert len(maps_count) == 3


@pytest.mark.integration
def test_seed_writes_manifest_json(db_session, tmp_path: Path) -> None:
    """The manifest JSON lands at ``{seeds_dir}/vlr_seed_{date}.json``.

    Operators need the file to diff runs; the seed writes it by
    default. We assert the path exists and contains the expected
    top-level shape — every field of :class:`VlrSeedManifest`
    plus the nested counters.
    """
    csv_path = _make_csv(tmp_path, rows=[_ROW_VALORANT_BO3_MAP1])
    seeds_out = tmp_path / "seeds-out"

    manifest = seed_from_vlr_csv(
        db_session,
        csv_path=csv_path,
        seeds_dir=seeds_out,
    )

    written = seeds_out / f"vlr_seed_{manifest.seed_date}.json"
    assert written.exists()

    import json

    payload = json.loads(written.read_text())
    assert payload["csv_path"] == str(csv_path)
    assert payload["teams"]["created"] == 2
    assert payload["tournaments"]["created"] == 1
    assert payload["matches"]["matches_inserted"] == 1
    assert payload["matches"]["maps_inserted"] == 1


def test_csv_header_matches_production_field_count() -> None:
    """Sanity guard: the test fixture's column count tracks the upstream CSV.

    The full production CSV ships 67 columns (per the BUF-8 v2 sample
    we ingested). If a future revision adds or removes a column, this
    test will fail before the integration tests do, pointing the
    operator at the column-list update before they get a confusing
    ``KeyError`` deep inside the seeder.
    """
    expected_columns = 67
    columns = _HEADER.split(",")
    assert len(columns) == expected_columns, (
        f"VLR CSV column count drifted: expected {expected_columns}, "
        f"got {len(columns)}. Update the test header (and the seed if "
        f"a new field needs reading)."
    )
